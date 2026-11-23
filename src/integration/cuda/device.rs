//! Device side of the CUDA 1D integration (feature `cuda`).
//!
//! `Cuda`, `Plan`, `Batch`, `Kernel`, `Accumulation` and `Layout` are the
//! prototype's internal types, still used by the test-only benchmarks; the
//! public [`Cuda1dPlan`] wraps a `Plan` in an [`Engine`].
use std::{
    any::Any,
    borrow::Cow,
    collections::HashMap,
    marker::PhantomData,
    ops::Deref,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex, OnceLock, PoisonError},
    time::Instant,
};

use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, DriverError,
        LaunchConfig, PushKernelArg, ValidAsZeroBits, sys::CUresult,
    },
    nvrtc::{CompileError, CompileOptions, compile_ptx_with_opts},
};

use super::{Cuda1dOptions, Cuda1dPlan, CudaError, CudaInfo, CudaPrecision, sealed::Frames};
use crate::integration::{Fixed1dResult, Integrator};
use crate::lut::prepared::Columns;
#[cfg(test)]
use {super::check_configuration, cudarc::driver::HostSlice};

/// CUDA API version of the cudarc bindings selected in `Cargo.toml`
/// (feature `cuda-13040`).
const BINDINGS_VERSION: (u32, u32) = (13, 4);
/// Default chunk budget cap.
const DEFAULT_BATCH_BYTES: usize = 1 << 30;
/// Frames per kernel launch, which bounds the work of one launch on display
/// GPUs with a watchdog.
const FRAMES_PER_LAUNCH: u32 = 256;
/// Layout of the public plans; the recorded runs picked it most often.
const DEFAULT_LAYOUT: Layout = Layout::Block;

impl From<DriverError> for CudaError {
    fn from(error: DriverError) -> Self {
        Self::Device(error.to_string())
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

/// Runs device work, turning a cudarc panic (for example a symbol missing from
/// an older driver) into [`CudaError::Device`].
fn guarded<T>(work: impl FnOnce() -> Result<T, CudaError>) -> Result<T, CudaError> {
    catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|payload| {
        Err(CudaError::Device(format!(
            "CUDA call panicked: {}",
            panic_message(payload.as_ref())
        )))
    })
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Accumulation {
    Plain,
    Compensated,
    /// Only the prototype tests and benchmarks use it.
    #[cfg_attr(not(test), allow(dead_code))]
    Double,
}

impl Accumulation {
    #[cfg(test)]
    pub(super) const ALL: [Self; 3] = [Self::Plain, Self::Compensated, Self::Double];

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Compensated => "compensated",
            Self::Double => "double",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Layout {
    Block,
    /// Only the prototype tests and benchmarks use it.
    #[cfg_attr(not(test), allow(dead_code))]
    Warp,
}

impl Layout {
    #[cfg(test)]
    pub(super) const ALL: [Self; 2] = [Self::Block, Self::Warp];

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::Warp => "warp",
        }
    }

    pub(super) fn bins_per_block(self) -> u32 {
        match self {
            Self::Block => 1,
            Self::Warp => 4,
        }
    }
}

/// Process-wide state of one initialised device: context and compiled module.
pub(super) struct Device {
    pub(super) context: Arc<CudaContext>,
    pub(super) module: Arc<CudaModule>,
    pub(super) info: CudaInfo,
    /// Read by the benchmarks only.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) compile_ms: f64,
    /// Read by the benchmarks only.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) module_ms: f64,
}

/// Returns the cached state of device `ordinal`, initialising it on first use.
/// On failure the returned description holds the reason and every field read
/// before the failure. Failures are not cached.
pub(super) fn device(ordinal: usize) -> Result<Arc<Device>, CudaInfo> {
    static DEVICES: OnceLock<Mutex<HashMap<usize, Arc<Device>>>> = OnceLock::new();
    // Holding the lock across initialisation compiles each device only once.
    // The panic guard below keeps the lock from being poisoned.
    let mut devices = DEVICES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(device) = devices.get(&ordinal) {
        return Ok(Arc::clone(device));
    }
    let mut info = CudaInfo::unavailable(ordinal, true, String::new());
    info.bindings_version = Some(BINDINGS_VERSION);
    let outcome = catch_unwind(AssertUnwindSafe(|| initialise(ordinal, &mut info)));
    match outcome {
        Ok(Ok(device)) => {
            let device = Arc::new(device);
            devices.insert(ordinal, Arc::clone(&device));
            Ok(device)
        }
        Ok(Err(reason)) => {
            info.available = false;
            info.reason = Some(reason);
            Err(info)
        }
        Err(payload) => {
            info.available = false;
            info.reason = Some(format!(
                "CUDA initialisation panicked: {}",
                panic_message(payload.as_ref())
            ));
            Err(info)
        }
    }
}

fn initialise(ordinal: usize, info: &mut CudaInfo) -> Result<Device, String> {
    // Probing the libraries first avoids cudarc's missing-library panic.
    // SAFETY: only attempts to dlopen the driver library.
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return Err("the CUDA driver library (libcuda) was not found".into());
    }
    let mut driver_version = 0;
    // SAFETY: valid out pointer; allowed before cuInit.
    if unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut driver_version) }
        == CUresult::CUDA_SUCCESS
    {
        info.driver_version = Some(driver_version);
    }
    // SAFETY: only attempts to dlopen the NVRTC library.
    let nvrtc_present = unsafe { cudarc::nvrtc::sys::is_culib_present() };
    if nvrtc_present {
        let (mut major, mut minor) = (0, 0);
        // SAFETY: valid out pointers.
        if unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) }
            == cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS
        {
            info.nvrtc_version = Some((major, minor));
        }
    }
    let count = match CudaContext::device_count() {
        Ok(count) => usize::try_from(count).unwrap_or(0),
        Err(error) if error.0 == CUresult::CUDA_ERROR_NO_DEVICE => 0,
        Err(error) => return Err(format!("CUDA driver initialisation failed: {error}")),
    };
    if count == 0 {
        return Err("no CUDA device found".into());
    }
    if ordinal >= count {
        return Err(format!(
            "CUDA device {ordinal} does not exist; {count} device(s) visible"
        ));
    }
    let context = CudaContext::new(ordinal)
        .map_err(|error| format!("creating a context on CUDA device {ordinal} failed: {error}"))?;
    info.device_name = context.name().ok();
    let (major, minor) = context
        .compute_capability()
        .map_err(|error| format!("reading the compute capability failed: {error}"))?;
    info.compute_capability = Some((major, minor));
    info.total_memory_bytes = context.total_mem().ok();
    if !nvrtc_present {
        return Err(
            "the NVRTC library (libnvrtc) was not found; put a CUDA toolkit's libnvrtc on the library path"
                .into(),
        );
    }
    let start = Instant::now();
    let architecture = format!("compute_{major}{minor}");
    let ptx = compile_ptx_with_opts(
        include_str!("kernels.cu"),
        CompileOptions {
            fmad: Some(false),
            options: vec![
                "--std=c++17".into(),
                format!("--gpu-architecture={architecture}"),
            ],
            ..Default::default()
        },
    )
    .map_err(|error| match error {
        CompileError::CompileError { nvrtc, log, .. } => format!(
            "NVRTC {} could not compile the kernels for {architecture} ({nvrtc:?}): {}",
            info.nvrtc_version
                .map_or_else(|| "?".into(), |(a, b)| format!("{a}.{b}")),
            log.to_string_lossy().trim()
        ),
        other => format!("NVRTC failed for {architecture}: {other}"),
    })?;
    let compile_ms = elapsed_ms(start);
    let start = Instant::now();
    let module = context.load_module(ptx).map_err(|error| {
        format!(
            "loading NVRTC PTX for compute capability {major}.{minor} failed ({error}); check driver and NVRTC compatibility"
        )
    })?;
    let module_ms = elapsed_ms(start);
    info.available = true;
    info.reason = None;
    Ok(Device {
        context,
        module,
        info: info.clone(),
        compile_ms,
        module_ms,
    })
}

/// A device and one stream on it. Cloning shares both.
#[derive(Clone)]
pub(super) struct Cuda {
    pub(super) device: Arc<Device>,
    pub(super) stream: Arc<CudaStream>,
}

impl Deref for Cuda {
    type Target = Device;

    fn deref(&self) -> &Device {
        &self.device
    }
}

impl Cuda {
    /// Opens device `ordinal` (initialised once per process) with a new stream.
    pub(super) fn open(ordinal: usize) -> Result<Self, CudaError> {
        let device = device(ordinal).map_err(|info| {
            CudaError::Unavailable(info.reason.unwrap_or_else(|| "unknown reason".into()))
        })?;
        let stream = guarded(|| Ok(device.context.new_stream()?))?;
        Ok(Self { device, stream })
    }

    /// Device 0 for tests and benchmarks, or `None` (a skip) when it is
    /// unavailable. Panics instead when `DIOPTRIN_REQUIRE_CUDA` is set.
    #[cfg(test)]
    pub(super) fn probe() -> Option<Self> {
        match Self::open(0) {
            Ok(cuda) => Some(cuda),
            Err(error) => {
                assert!(!super::tests::cuda_required(), "{error}");
                eprintln!("Skipped CUDA test: {error}");
                None
            }
        }
    }

    #[cfg(test)]
    pub(super) fn prepare(
        &self,
        integrator: &Integrator,
        shape: (usize, usize),
        bins: u32,
    ) -> Result<Plan, CudaError> {
        check_configuration(integrator, shape, bins as usize, None)?;
        with_table(integrator, shape, bins as usize, |table| {
            self.upload(table, [1.0_f32], encode_f32)
        })
    }

    #[cfg(test)]
    pub(super) fn prepare_f64(
        &self,
        integrator: &Integrator,
        shape: (usize, usize),
        bins: u32,
    ) -> Result<Plan<f64>, CudaError> {
        check_configuration(integrator, shape, bins as usize, None)?;
        with_table(integrator, shape, bins as usize, |table| {
            self.upload(table, [1.0_f64], |weights| Cow::Borrowed(weights))
        })
    }

    fn upload<T: DeviceRepr + Clone>(
        &self,
        table: &HostTable<'_>,
        unit_weight: [T; 1],
        encode_weights: impl for<'w> FnOnce(&'w [f64]) -> Cow<'w, [T]>,
    ) -> Result<Plan<T>, CudaError> {
        let weights = table.weights.map(encode_weights);
        let (columns, w, weighted) = device_arrays(table, weights.as_deref(), &unit_weight);
        let outcome = guarded(|| {
            let start = Instant::now();
            let rows = self.stream.clone_htod(&table.rows)?;
            let columns = self.stream.clone_htod(columns)?;
            let weights_device = self.stream.clone_htod(w)?;
            let denominator = self.stream.clone_htod(table.denominator)?;
            self.stream.synchronize()?;
            Ok(Plan {
                cuda: self.clone(),
                pixels: table.pixels,
                bins: table.bins,
                entries: table.entries,
                rows,
                columns,
                weights: weights_device,
                weighted,
                denominator,
                upload_ms: elapsed_ms(start),
            })
        });
        if outcome.is_err() {
            // Copies queued before the failure may still read `weights` and
            // the table's host vectors, which are dropped after returning.
            settle(&self.stream);
        }
        outcome
    }
}

/// Weights of `f32` storage: each `f64` weight rounded to nearest.
pub(super) fn encode_f32(weights: &[f64]) -> Cow<'_, [f32]> {
    Cow::Owned(weights.iter().map(|&w| w as f32).collect())
}

/// Columns and weights that `Cuda::upload` copies, and the kernels'
/// `weighted` flag. A fully masked table gets one dummy column and implicit
/// unit weights (or an empty weight array) one dummy weight, because device
/// buffers cannot be empty; the kernels never read either, since every row is
/// empty or `weighted` is 0.
pub(super) fn device_arrays<'t, T>(
    table: &'t HostTable<'_>,
    weights: Option<&'t [T]>,
    unit_weight: &'t [T; 1],
) -> (&'t [u32], &'t [T], u32) {
    let columns = if table.columns.is_empty() {
        &[0u32][..]
    } else {
        table.columns.as_ref()
    };
    let stored = weights.filter(|w| !w.is_empty()).unwrap_or(unit_weight);
    (columns, stored, u32::from(weights.is_some()))
}

/// Blocks until all work queued on `stream` has executed, for error paths that
/// must not return while an asynchronous copy may still read host memory.
///
/// `CudaStream::synchronize` cannot be used for this: its `bind_to_thread`
/// first takes any error that a `Drop` recorded on the shared context
/// (`CudaContext::check_err`) and then returns it without synchronising. This
/// calls the driver directly instead, so the recorded error stays pending and
/// is reported by the next checked call on the device, possibly by another
/// plan on the same device (cudarc keeps one error slot per context).
/// Failures here are ignored: the caller already returns an error.
fn settle(stream: &CudaStream) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the stream holds an `Arc` of its context, so both handles
        // are alive, and the stream belongs to that context. Binding the
        // context is what `bind_to_thread` does after its error check.
        unsafe {
            cudarc::driver::result::ctx::set_current(stream.context().cu_ctx())?;
            cudarc::driver::result::stream::synchronize(stream.cu_stream())
        }
    }));
}

/// The host CSR table in device index types, borrowed from the integrator's
/// cached lookup table where possible.
pub(super) struct HostTable<'a> {
    pub(super) pixels: u32,
    pub(super) bins: u32,
    pub(super) entries: u32,
    pub(super) rows: Vec<u32>,
    pub(super) columns: Cow<'a, [u32]>,
    pub(super) weights: Option<&'a [f64]>,
    pub(super) denominator: &'a [f64],
    pub(super) radial: Vec<f64>,
}

/// Builds (or reuses) the integrator's masked 1D CSR table and strict `f64`
/// denominators, and passes them to `use_table`. The configuration must
/// already have passed `check_configuration`.
pub(super) fn with_table<R>(
    integrator: &Integrator,
    shape: (usize, usize),
    bins: usize,
    use_table: impl FnOnce(&HostTable<'_>) -> Result<R, CudaError>,
) -> Result<R, CudaError> {
    let too_large = || CudaError::Unsupported("the CUDA table exceeds 32-bit indices".into());
    let pixels = u32::try_from(shape.0 * shape.1).map_err(|_| too_large())?;
    let n_bins = u32::try_from(bins).map_err(|_| too_large())?;
    let view = integrator.for_shape(shape.0, shape.1);
    let norm = view.normalization();
    let key = integrator.one_d_table_key(bins).ok_or_else(|| {
        CudaError::Unsupported("the integration method has no 1D lookup table".into())
    })?;
    let lut = view.ensure_lut1d(key);
    let mask = integrator.mask.as_ref().map(|mask| mask.data.as_slice());
    let (table, denominator) = lut.fixed_csr_parts(mask, &norm);
    let rows = table
        .row_ptr
        .iter()
        .map(|&row| u32::try_from(row))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| too_large())?;
    let entries = rows
        .last()
        .copied()
        .ok_or_else(|| CudaError::Device("the CUDA table has no rows".into()))?;
    if entries > u32::MAX - 128 {
        return Err(CudaError::Unsupported(format!(
            "the CUDA table has {entries} entries; at most {} are supported",
            u32::MAX - 128
        )));
    }
    if rows.len() != bins + 1 || denominator.len() != bins {
        return Err(CudaError::Device(
            "the CUDA table does not match the requested bins".into(),
        ));
    }
    let columns = match &table.col_idx {
        Columns::Compact(columns) => Cow::Borrowed(columns.as_slice()),
        Columns::Wide(columns) => Cow::Owned(
            columns
                .iter()
                .map(|&column| u32::try_from(column))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| too_large())?,
        ),
    };
    use_table(&HostTable {
        pixels,
        bins: n_bins,
        entries,
        rows,
        columns,
        weights: table.values.as_deref(),
        denominator,
        radial: lut.bins(),
    })
}

pub(super) struct Plan<T = f32> {
    pub(super) cuda: Cuda,
    pub(super) pixels: u32,
    pub(super) bins: u32,
    /// Read by the tests and benchmarks only.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) entries: u32,
    pub(super) rows: CudaSlice<u32>,
    pub(super) columns: CudaSlice<u32>,
    pub(super) weights: CudaSlice<T>,
    pub(super) weighted: u32,
    pub(super) denominator: CudaSlice<f64>,
    /// Read by the benchmarks only.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) upload_ms: f64,
}

/// Prototype batch of the tests and benchmarks; public plans use `Engine`.
#[cfg(test)]
pub(super) struct Batch<'a, T = f32, I = T> {
    pub(super) plan: &'a Plan<T>,
    pub(super) stream: Arc<CudaStream>,
    pub(super) input: CudaSlice<I>,
    pub(super) output: CudaSlice<f64>,
    pub(super) frames: u32,
    pub(super) errors: bool,
}

pub(super) struct Kernel<T = f32, I = T> {
    pub(super) function: CudaFunction,
    pub(super) layout: Layout,
    pub(super) errors: bool,
    storage: PhantomData<(T, I)>,
}

#[cfg(test)]
impl<T: DeviceRepr + ValidAsZeroBits> Plan<T> {
    pub(super) fn batch(&self, frames: u32, errors: bool) -> Result<Batch<'_, T>, CudaError> {
        self.batch_on(frames, errors, Arc::clone(&self.cuda.stream))
    }

    pub(super) fn batch_on(
        &self,
        frames: u32,
        errors: bool,
        stream: Arc<CudaStream>,
    ) -> Result<Batch<'_, T>, CudaError> {
        self.batch_with_input(frames, errors, stream)
    }

    pub(super) fn batch_with_input<I: DeviceRepr + ValidAsZeroBits>(
        &self,
        frames: u32,
        errors: bool,
        stream: Arc<CudaStream>,
    ) -> Result<Batch<'_, T, I>, CudaError> {
        if frames == 0 || frames > i32::MAX as u32 / self.bins {
            return Err(CudaError::InvalidInput(
                "CUDA batch exceeds the grid limit or is empty".into(),
            ));
        }
        let elements = |per_frame: usize, element_bytes: usize| {
            per_frame
                .checked_mul(frames as usize)
                .filter(|&len| len <= isize::MAX as usize / element_bytes)
                .ok_or_else(|| {
                    CudaError::InvalidInput(
                        "CUDA batch allocation exceeds the addressable range".into(),
                    )
                })
        };
        let input_len = elements(self.pixels as usize, size_of::<I>())?;
        let output_len = elements(
            self.bins as usize * (1 + usize::from(errors)),
            size_of::<f64>(),
        )?;
        Ok(Batch {
            plan: self,
            input: stream.alloc_zeros(input_len)?,
            output: stream.alloc_zeros(output_len)?,
            stream,
            frames,
            errors,
        })
    }
}

impl<T> Plan<T> {
    pub(super) fn load_kernel<I>(
        &self,
        name: &str,
        layout: Layout,
        errors: bool,
    ) -> Result<Kernel<T, I>, CudaError> {
        let suffix = if errors { "iv" } else { "i" };
        let name = format!("{name}_{}_{suffix}", layout.name());
        let function = self.cuda.module.load_function(&name).map_err(|error| {
            CudaError::Unavailable(format!("CUDA kernel {name} is missing: {error}"))
        })?;
        Ok(Kernel {
            function,
            layout,
            errors,
            storage: PhantomData,
        })
    }
}

impl<T: DeviceRepr> Plan<T> {
    /// Launches `kernel` on the first `frames` frames of `input`, writing the
    /// start of `output`. This is the only place that launches the gather
    /// kernels: the argument order matches `STORED_KERNEL` in `kernels.cu`.
    fn launch<I: DeviceRepr>(
        &self,
        stream: &Arc<CudaStream>,
        kernel: &Kernel<T, I>,
        input: &CudaSlice<I>,
        output: &mut CudaSlice<f64>,
        frames: u32,
    ) -> Result<(), CudaError> {
        let output_per_frame = self.bins as usize * (1 + usize::from(kernel.errors));
        let pixels = self.pixels as usize;
        if frames == 0
            || input.len() < frames as usize * pixels
            || output.len() < frames as usize * output_per_frame
        {
            return Err(CudaError::InvalidInput(
                "CUDA launch exceeds its buffers or is empty".into(),
            ));
        }
        let groups = self.bins.div_ceil(kernel.layout.bins_per_block());
        // Keep the grid within i32::MAX blocks (groups <= bins <= i32::MAX).
        let per_launch = (i32::MAX as u32 / groups).clamp(1, FRAMES_PER_LAUNCH);
        for first in (0..frames).step_by(per_launch as usize) {
            let end = (first + per_launch).min(frames);
            let input = input.slice(first as usize * pixels..end as usize * pixels);
            let mut output = output
                .slice_mut(first as usize * output_per_frame..end as usize * output_per_frame);
            let config = LaunchConfig {
                grid_dim: (groups * (end - first), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = stream.launch_builder(&kernel.function);
            launch
                .arg(&self.rows)
                .arg(&self.columns)
                .arg(&self.weights)
                .arg(&self.denominator)
                .arg(&input)
                .arg(&mut output)
                .arg(&self.pixels)
                .arg(&self.bins)
                .arg(&self.weighted);
            // SAFETY: the views span exactly the launched frames and the
            // kernel matches the storage type, layout and variance specialization.
            unsafe { launch.launch(config)? };
        }
        Ok(())
    }
}

impl Plan {
    pub(super) fn kernel(
        &self,
        mode: Accumulation,
        layout: Layout,
        errors: bool,
    ) -> Result<Kernel, CudaError> {
        self.load_kernel(mode.name(), layout, errors)
    }
}

impl Plan<f64> {
    pub(super) fn kernel(&self, layout: Layout, errors: bool) -> Result<Kernel<f64>, CudaError> {
        self.load_kernel("full_f64", layout, errors)
    }

    pub(super) fn kernel_f32(
        &self,
        layout: Layout,
        errors: bool,
    ) -> Result<Kernel<f64, f32>, CudaError> {
        self.load_kernel("f32_f64", layout, errors)
    }

    #[cfg(test)]
    pub(super) fn kernel_u32(
        &self,
        layout: Layout,
        errors: bool,
    ) -> Result<Kernel<f64, u32>, CudaError> {
        self.load_kernel("u32_f64", layout, errors)
    }

    #[cfg(test)]
    pub(super) fn kernel_i32(
        &self,
        layout: Layout,
        errors: bool,
    ) -> Result<Kernel<f64, i32>, CudaError> {
        self.load_kernel("i32_f64", layout, errors)
    }
}

/// Rounds the first `len` values of `source` to `f32` into `destination`
/// with the `cast_f64_input` kernel.
fn cast_f64(
    stream: &Arc<CudaStream>,
    cast: &CudaFunction,
    source: &CudaSlice<f64>,
    destination: &mut CudaSlice<f32>,
    len: usize,
) -> Result<(), CudaError> {
    if len == 0 || len > source.len() || len > destination.len() {
        return Err(CudaError::InvalidInput(
            "f64 cast must be non-empty and fit both buffers".into(),
        ));
    }
    let source = source.slice(..len);
    let mut destination = destination.slice_mut(..len);
    let config = LaunchConfig {
        grid_dim: (len.div_ceil(256).min(65_535) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let len = len as u64;
    let mut launch = stream.launch_builder(cast);
    launch.arg(&source).arg(&mut destination).arg(&len);
    // SAFETY: both views cover len elements and cast is the matching
    // f64-to-f32 grid-stride function of this module.
    unsafe { launch.launch(config)? };
    Ok(())
}

#[cfg(test)]
impl<T: DeviceRepr, I: DeviceRepr> Batch<'_, T, I> {
    pub(super) fn upload(
        &mut self,
        offset: usize,
        frames: &(impl HostSlice<I> + ?Sized),
    ) -> Result<(), CudaError> {
        let pixels = self.plan.pixels as usize;
        let end = offset
            .checked_add(frames.len())
            .filter(|&end| end <= self.input.len());
        if !offset.is_multiple_of(pixels) || !frames.len().is_multiple_of(pixels) {
            return Err(CudaError::InvalidInput(
                "CUDA upload must cover whole frames".into(),
            ));
        }
        let end =
            end.ok_or_else(|| CudaError::InvalidInput("CUDA upload exceeds the batch".into()))?;
        let mut destination = self.input.slice_mut(offset..end);
        Ok(self.stream.memcpy_htod(frames, &mut destination)?)
    }
}

#[cfg(test)]
impl Batch<'_> {
    pub(super) fn upload_f64(
        &mut self,
        frames: &(impl HostSlice<f64> + ?Sized),
        staging: &mut CudaSlice<f64>,
        cast: &CudaFunction,
    ) -> Result<(), CudaError> {
        let len = frames.len();
        if len == 0
            || !len.is_multiple_of(self.plan.pixels as usize)
            || len > self.input.len()
            || len > staging.len()
        {
            return Err(CudaError::InvalidInput(
                "f64 upload must fit both buffers and cover whole frames".into(),
            ));
        }
        let mut source = staging.slice_mut(..len);
        self.stream.memcpy_htod(frames, &mut source)?;
        cast_f64(&self.stream, cast, staging, &mut self.input, len)
    }
}

#[cfg(test)]
impl<T: DeviceRepr, I: DeviceRepr> Batch<'_, T, I> {
    pub(super) fn run(&mut self, kernel: &Kernel<T, I>) -> Result<(), CudaError> {
        self.run_frames(self.frames, kernel)
    }

    pub(super) fn run_frames(
        &mut self,
        frames: u32,
        kernel: &Kernel<T, I>,
    ) -> Result<(), CudaError> {
        if frames == 0 || frames > self.frames {
            return Err(CudaError::InvalidInput(
                "CUDA launch frame count exceeds the allocated batch or is empty".into(),
            ));
        }
        if self.errors != kernel.errors {
            return Err(CudaError::InvalidInput(
                "CUDA kernel and output strides differ".into(),
            ));
        }
        self.plan
            .launch(&self.stream, kernel, &self.input, &mut self.output, frames)
    }

    pub(super) fn download(
        &self,
        output: &mut (impl HostSlice<f64> + ?Sized),
    ) -> Result<(), CudaError> {
        self.download_frames(self.frames, output)
    }

    pub(super) fn download_frames(
        &self,
        frames: u32,
        output: &mut (impl HostSlice<f64> + ?Sized),
    ) -> Result<(), CudaError> {
        if frames == 0
            || frames > self.frames
            || output.len()
                != frames as usize * self.plan.bins as usize * (1 + usize::from(self.errors))
        {
            return Err(CudaError::InvalidInput(
                "host output does not match CUDA batch shape".into(),
            ));
        }
        Ok(self
            .stream
            .memcpy_dtoh(&self.output.slice(..output.len()), output)?)
    }
}

/// Device scratch of a public plan, grown on demand up to
/// `max_batch_frames` frames and reused by later calls.
#[derive(Default)]
struct Scratch {
    /// `f32` frames: the kernel input of `f32` storage, or `f32` input of an
    /// `f64` plan.
    narrow: Option<CudaSlice<f32>>,
    /// `f64` frames: staging for the cast to `f32` storage, or the kernel
    /// input of an `f64` plan.
    wide: Option<CudaSlice<f64>>,
    /// Interleaved results, sized for `[I, σ]` per bin.
    output: Option<CudaSlice<f64>>,
}

/// Returns a buffer of at least `len` elements, replacing a smaller one.
fn reserve<'s, X: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    slot: &'s mut Option<CudaSlice<X>>,
    len: usize,
) -> Result<&'s mut CudaSlice<X>, CudaError> {
    if !slot.as_ref().is_some_and(|buffer| buffer.len() >= len) {
        // Free the smaller buffer before allocating its replacement.
        *slot = None;
        *slot = Some(stream.alloc_zeros(len)?);
    }
    slot.as_mut()
        .ok_or_else(|| CudaError::Device("CUDA scratch allocation is missing".into()))
}

/// Loads the kernel variants without and with errors.
fn both<K>(make: impl Fn(bool) -> Result<K, CudaError>) -> Result<[K; 2], CudaError> {
    Ok([make(false)?, make(true)?])
}

/// Number of whole frames in `slices`, each of which holds one or more
/// frames of `pixels` values.
pub(super) fn frame_count<X>(slices: &[&[X]], pixels: usize) -> Result<usize, CudaError> {
    slices.iter().try_fold(0, |count, slice| {
        if slice.is_empty() || !slice.len().is_multiple_of(pixels) {
            return Err(CudaError::InvalidInput(format!(
                "CUDA input of {} values is not a whole number of {pixels}-pixel frames",
                slice.len()
            )));
        }
        Ok(count + slice.len() / pixels)
    })
}

/// Copies `slices` of whole frames (one frame each, or one contiguous stack
/// chunk) to consecutive frame slots at the start of `input`, with one copy
/// per slice.
fn copy_frames<X: DeviceRepr>(
    stream: &Arc<CudaStream>,
    slices: &[&[X]],
    input: &mut CudaSlice<X>,
    pixels: usize,
) -> Result<(), CudaError> {
    // memcpy_htod asserts that the destination is long enough.
    if input.len() < frame_count(slices, pixels)? * pixels {
        return Err(CudaError::InvalidInput(
            "CUDA input buffer is too small".into(),
        ));
    }
    let mut offset = 0;
    for slice in slices {
        let mut destination = input.slice_mut(offset..offset + slice.len());
        stream.memcpy_htod(*slice, &mut destination)?;
        offset += slice.len();
    }
    Ok(())
}

/// Number of slices in `frames`.
fn frames_len(frames: &Frames<'_, '_>) -> usize {
    match frames {
        Frames::F32(frames) => frames.len(),
        Frames::F64(frames) => frames.len(),
    }
}

/// Uploaded table and the kernels a public plan launches, by storage type.
enum Tables {
    /// `f32` weights: gather kernels by `errors`, and the `f64` input cast.
    Single {
        plan: Plan<f32>,
        gather: [Kernel<f32>; 2],
        cast: CudaFunction,
    },
    /// `f64` weights: kernels for `f64` and for `f32` input, by `errors`.
    Double {
        plan: Plan<f64>,
        wide: [Kernel<f64>; 2],
        narrow: [Kernel<f64, f32>; 2],
    },
}

/// Device state of a public [`Cuda1dPlan`].
pub(super) struct Engine {
    tables: Tables,
    pixels: usize,
    bins: usize,
    max_batch_frames: usize,
    scratch: Mutex<Scratch>,
}

impl Engine {
    fn stream(&self) -> &Arc<CudaStream> {
        match &self.tables {
            Tables::Single { plan, .. } => &plan.cuda.stream,
            Tables::Double { plan, .. } => &plan.cuda.stream,
        }
    }

    /// Integrates frames in chunks of at most `max_batch_frames`. Without
    /// `stacked`, `frames` holds non-empty frames of the plan's length;
    /// with it, `frames` holds one non-empty contiguous stack of whole frames,
    /// copied with one copy per chunk (both checked by the caller). The
    /// device sees the same chunk contents either way.
    pub(super) fn run(
        &self,
        frames: Frames<'_, '_>,
        stacked: bool,
        errors: bool,
        radial: &[f64],
    ) -> Result<Vec<Fixed1dResult>, CudaError> {
        let count = match frames {
            Frames::F32(frames) if stacked => frame_count(frames, self.pixels)?,
            Frames::F64(frames) if stacked => frame_count(frames, self.pixels)?,
            Frames::F32(frames) => frames.len(),
            Frames::F64(frames) => frames.len(),
        };
        if stacked && frames_len(&frames) != 1 {
            return Err(CudaError::InvalidInput(
                "a CUDA stack must be one slice".into(),
            ));
        }
        let pixels = self.pixels;
        let stride = 1 + usize::from(errors);
        let per_frame = self.bins * stride;
        let mut host = vec![0.0; count * per_frame];
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(PoisonError::into_inner);
            let outcome = guarded(|| {
                for first in (0..count).step_by(self.max_batch_frames) {
                    let end = (first + self.max_batch_frames).min(count);
                    let host = &mut host[first * per_frame..end * per_frame];
                    let values = first * pixels..end * pixels;
                    match frames {
                        Frames::F32(frames) if stacked => {
                            let chunk = &frames[0][values];
                            let chunk = Frames::F32(std::slice::from_ref(&chunk));
                            self.run_chunk(&mut scratch, chunk, errors, host)?;
                        }
                        Frames::F64(frames) if stacked => {
                            let chunk = &frames[0][values];
                            let chunk = Frames::F64(std::slice::from_ref(&chunk));
                            self.run_chunk(&mut scratch, chunk, errors, host)?;
                        }
                        Frames::F32(frames) => {
                            let chunk = Frames::F32(&frames[first..end]);
                            self.run_chunk(&mut scratch, chunk, errors, host)?;
                        }
                        Frames::F64(frames) => {
                            let chunk = Frames::F64(&frames[first..end]);
                            self.run_chunk(&mut scratch, chunk, errors, host)?;
                        }
                    }
                }
                Ok(())
            });
            if outcome.is_err() {
                // Borrowed host frames must not be read after returning, and
                // a failed or panicked call leaves the scratch in an unknown
                // state. `settle` synchronises even when an error recorded on
                // the shared context made the checked calls return early; that
                // error may also be the one returned here, although another
                // plan on the device recorded it (see `settle`).
                settle(self.stream());
                *scratch = Scratch::default();
            }
            outcome?;
        }
        Ok(host
            .chunks_exact(per_frame)
            .map(|frame| Fixed1dResult {
                radial: radial.to_vec(),
                intensity: frame.iter().step_by(stride).copied().collect(),
                errors: errors.then(|| frame.iter().skip(1).step_by(stride).copied().collect()),
            })
            .collect())
    }

    /// Copies one chunk to the device, integrates it, copies the interleaved
    /// results into `host` and synchronises.
    fn run_chunk(
        &self,
        scratch: &mut Scratch,
        frames: Frames<'_, '_>,
        errors: bool,
        host: &mut [f64],
    ) -> Result<(), CudaError> {
        let pixels = self.pixels;
        let stream = self.stream();
        let count = match frames {
            Frames::F32(frames) => frame_count(frames, pixels)?,
            Frames::F64(frames) => frame_count(frames, pixels)?,
        };
        let launched = u32::try_from(count)
            .map_err(|_| CudaError::InvalidInput("too many frames in one chunk".into()))?;
        let values = count * pixels;
        let Scratch {
            narrow,
            wide,
            output,
        } = scratch;
        let output = reserve(stream, output, count * self.bins * 2)?;
        let e = usize::from(errors);
        match (&self.tables, frames) {
            (Tables::Single { plan, gather, .. }, Frames::F32(frames)) => {
                let input = reserve(stream, narrow, values)?;
                copy_frames(stream, frames, input, pixels)?;
                plan.launch(stream, &gather[e], input, output, launched)?;
            }
            (Tables::Single { plan, gather, cast }, Frames::F64(frames)) => {
                let staging = reserve(stream, wide, values)?;
                copy_frames(stream, frames, staging, pixels)?;
                let input = reserve(stream, narrow, values)?;
                cast_f64(stream, cast, staging, input, values)?;
                plan.launch(stream, &gather[e], input, output, launched)?;
            }
            (
                Tables::Double {
                    plan,
                    wide: kernels,
                    ..
                },
                Frames::F64(frames),
            ) => {
                // f64 plans keep only one input buffer, as budgeted.
                *narrow = None;
                let input = reserve(stream, wide, values)?;
                copy_frames(stream, frames, input, pixels)?;
                plan.launch(stream, &kernels[e], input, output, launched)?;
            }
            (
                Tables::Double {
                    plan,
                    narrow: kernels,
                    ..
                },
                Frames::F32(frames),
            ) => {
                *wide = None;
                let input = reserve(stream, narrow, values)?;
                copy_frames(stream, frames, input, pixels)?;
                plan.launch(stream, &kernels[e], input, output, launched)?;
            }
        }
        let len = count * self.bins * (1 + e);
        if host.len() != len {
            return Err(CudaError::InvalidInput(
                "host output does not match the CUDA chunk".into(),
            ));
        }
        stream.memcpy_dtoh(&output.slice(..len), host)?;
        stream.synchronize()?;
        Ok(())
    }
}

/// Steps 6 to 8 of [`Integrator::prepare_cuda1d`]; steps 1 to 4 have passed.
pub(super) fn prepare(
    integrator: &Integrator,
    shape: (usize, usize),
    pixels: usize,
    n_points: usize,
    options: &Cuda1dOptions,
) -> Result<Cuda1dPlan, CudaError> {
    let cuda = Cuda::open(options.device())?;
    let precision = options.precision();
    let (tables, entries, radial) = with_table(integrator, shape, n_points, |table| {
        let layout = DEFAULT_LAYOUT;
        let tables = match precision {
            CudaPrecision::F32 | CudaPrecision::F32Compensated => {
                let plan = cuda.upload(table, [1.0_f32], encode_f32)?;
                let mode = if precision == CudaPrecision::F32 {
                    Accumulation::Plain
                } else {
                    Accumulation::Compensated
                };
                let gather = both(|errors| plan.kernel(mode, layout, errors))?;
                let cast = cuda
                    .module
                    .load_function("cast_f64_input")
                    .map_err(|error| {
                        CudaError::Unavailable(format!(
                            "CUDA kernel cast_f64_input is missing: {error}"
                        ))
                    })?;
                Tables::Single { plan, gather, cast }
            }
            CudaPrecision::F64 => {
                let plan = cuda.upload(table, [1.0_f64], |weights| Cow::Borrowed(weights))?;
                let wide = both(|errors| plan.kernel(layout, errors))?;
                let narrow = both(|errors| plan.kernel_f32(layout, errors))?;
                Tables::Double { plan, wide, narrow }
            }
        };
        Ok((tables, table.entries as usize, table.radial.clone()))
    })?;
    let storage = if precision == CudaPrecision::F64 {
        size_of::<f64>()
    } else {
        size_of::<f32>() + size_of::<f64>()
    };
    let frame_bytes = pixels
        .checked_mul(storage)
        .and_then(|bytes| bytes.checked_add(n_points.checked_mul(2 * size_of::<f64>())?))
        .ok_or_else(|| CudaError::InvalidInput("one CUDA frame exceeds usize bytes".into()))?;
    let max_batch_frames = match options.max_batch_bytes() {
        Some(budget) => match budget / frame_bytes {
            0 => {
                return Err(CudaError::InvalidInput(format!(
                    "max_batch_bytes too small for one frame ({budget} < {frame_bytes} bytes)"
                )));
            }
            frames => frames,
        },
        None => {
            let (free, _) = guarded(|| Ok(cuda.context.mem_get_info()?))?;
            match DEFAULT_BATCH_BYTES.min(free / 2) / frame_bytes {
                // A large frame may use more than half of the free memory.
                0 if frame_bytes <= free => 1,
                0 => {
                    return Err(CudaError::Device(format!(
                        "not enough free device memory for one frame ({free} < {frame_bytes} bytes)"
                    )));
                }
                frames => frames,
            }
        }
    };
    // Chunks are counted in u32 by the launch code.
    let max_batch_frames = max_batch_frames.min(u32::MAX as usize);
    let bins = n_points;
    Ok(Cuda1dPlan {
        shape,
        n_points,
        precision,
        device: options.device(),
        device_name: cuda.info.device_name.clone().unwrap_or_default(),
        radial,
        table_entries: entries,
        max_batch_frames,
        engine: Engine {
            tables,
            pixels,
            bins,
            max_batch_frames,
            scratch: Mutex::new(Scratch::default()),
        },
    })
}

pub(super) fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}
