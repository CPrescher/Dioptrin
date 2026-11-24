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
        LaunchConfig, PinnedHostSlice, PushKernelArg, ValidAsZeroBits, sys::CUresult,
    },
    nvrtc::{CompileError, CompileOptions, compile_ptx_with_opts},
};
use rayon::prelude::*;

use super::{
    Bslz4Dtype, Cuda1dOptions, Cuda1dPlan, CudaError, CudaInfo, CudaPrecision, sealed::Frames,
};
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
/// Minimum element count before staging into pinned host memory uses Rayon.
const PARALLEL_STAGE_THRESHOLD: usize = 65_536;
/// Target pinned staging size per pipeline slot (192 MiB: holds 4 Eiger 9M f32 frames).
const PINNED_SLOT_BYTES: usize = 192 << 20;
/// Maximum average entries per bin that prefer `Layout::SubWarp` (8 threads/bin, ideal for 2D).
const SUBWARP_MAX_ENTRIES_PER_BIN: usize = 256;
/// Maximum average entries per bin that still prefer `Layout::Warp` over `Layout::Block`.
const WARP_MAX_ENTRIES_PER_BIN: usize = 5000;

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
    Warp,
    SubWarp,
}

impl Layout {
    #[cfg(test)]
    pub(super) const ALL: [Self; 2] = [Self::Block, Self::Warp];

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::Warp => "warp",
            Self::SubWarp => "subwarp",
        }
    }

    pub(super) fn bins_per_block(self) -> u32 {
        match self {
            Self::Block => 1,
            Self::Warp => 4,
            Self::SubWarp => 16,
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

/// Builds (or reuses) the integrator's masked 2D CSR table and strict `f64`
/// denominators, and passes them to `use_table`.
pub(super) fn with_table_2d<R>(
    integrator: &Integrator,
    shape: (usize, usize),
    n_radial: usize,
    n_azimuthal: usize,
    azimuthal_range: Option<(f64, f64)>,
    use_table: impl FnOnce(&HostTable<'_>, Option<Vec<f64>>, Option<Vec<f64>>) -> Result<R, CudaError>,
) -> Result<R, CudaError> {
    let too_large = || CudaError::Unsupported("the CUDA table exceeds 32-bit indices".into());
    let pixels = u32::try_from(shape.0 * shape.1).map_err(|_| too_large())?;
    let total_bins = n_radial.checked_mul(n_azimuthal).ok_or_else(too_large)?;
    let n_bins = u32::try_from(total_bins).map_err(|_| too_large())?;
    let view = integrator.for_shape(shape.0, shape.1);
    let norm = view.normalization();
    let (key, radial_axis, azimuthal_axis) = match azimuthal_range {
        None => {
            let key = integrator
                .try_lut2d_key(&view, n_radial, n_azimuthal, None)
                .map_err(CudaError::InvalidInput)?;
            let radial = integrator.radial_range().map(|(lo, hi)| {
                crate::math::histogram::bins(n_radial, lo, (hi - lo) / n_radial as f64)
            });
            (key, radial, None)
        }
        Some(range) => {
            let grid = integrator
                .azimuthal_range_grid(&view, n_radial, n_azimuthal, range)
                .map_err(CudaError::InvalidInput)?;
            let rad = grid.radial_centers();
            let az = grid.azimuthal_centers();
            (integrator.shared_grid_key(grid), Some(rad), Some(az))
        }
    };
    let lut = view.ensure_lut2d(key);
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
    if rows.len() != total_bins + 1 || denominator.len() != total_bins {
        return Err(CudaError::Device(
            "the CUDA table does not match the requested 2D bins".into(),
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
    use_table(
        &HostTable {
            pixels,
            bins: n_bins,
            entries,
            rows,
            columns,
            weights: table.values.as_deref(),
            denominator,
            radial: Vec::new(),
        },
        radial_axis,
        azimuthal_axis,
    )
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
        // Bound one launch on display GPUs with a watchdog.
        let per_launch = FRAMES_PER_LAUNCH;
        for first in (0..frames).step_by(per_launch as usize) {
            let end = (first + per_launch).min(frames);
            let count = end - first;
            let input = input.slice(first as usize * pixels..end as usize * pixels);
            let mut output = output
                .slice_mut(first as usize * output_per_frame..end as usize * output_per_frame);
            let config = LaunchConfig {
                grid_dim: (groups, count.div_ceil(4), 1),
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
                .arg(&self.weighted)
                .arg(&count);
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

    pub(super) fn kernel_u16(
        &self,
        mode: Accumulation,
        layout: Layout,
        errors: bool,
    ) -> Result<Kernel<f32, u16>, CudaError> {
        self.load_kernel(&format!("u16_{}", mode.name()), layout, errors)
    }

    pub(super) fn kernel_u32(
        &self,
        mode: Accumulation,
        layout: Layout,
        errors: bool,
    ) -> Result<Kernel<f32, u32>, CudaError> {
        self.load_kernel(&format!("u32_{}", mode.name()), layout, errors)
    }

    pub(super) fn kernel_i32(
        &self,
        mode: Accumulation,
        layout: Layout,
        errors: bool,
    ) -> Result<Kernel<f32, i32>, CudaError> {
        self.load_kernel(&format!("i32_{}", mode.name()), layout, errors)
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

    pub(super) fn kernel_u16(
        &self,
        layout: Layout,
        errors: bool,
    ) -> Result<Kernel<f64, u16>, CudaError> {
        self.load_kernel("u16_f64", layout, errors)
    }

    pub(super) fn kernel_u32(
        &self,
        layout: Layout,
        errors: bool,
    ) -> Result<Kernel<f64, u32>, CudaError> {
        self.load_kernel("u32_f64", layout, errors)
    }

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
#[cfg(test)]
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

/// Returns `true` when `ptr` resides in CUDA page-locked (pinned) host memory,
/// allowing async hardware PCIe DMA directly from the caller's slice without an
/// extra CPU staging copy.
///
/// If no CUDA context is bound on the calling thread, `cuPointerGetAttribute`
/// returns `CUDA_ERROR_INVALID_CONTEXT`, which safely returns `false` and falls
/// back to staging.
fn is_pinned_host(ptr: *const std::ffi::c_void) -> bool {
    if ptr.is_null() {
        return false;
    }
    let mut mem_type: u32 = 0;
    // SAFETY: `cuPointerGetAttribute` queries driver metadata for `ptr` without
    // dereferencing it; non-CUDA host pointers return `CUDA_ERROR_INVALID_VALUE`.
    let status = unsafe {
        cudarc::driver::sys::cuPointerGetAttribute(
            (&mut mem_type as *mut u32).cast(),
            cudarc::driver::sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            ptr as cudarc::driver::sys::CUdeviceptr,
        )
    };
    status == CUresult::CUDA_SUCCESS
        && mem_type == cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_HOST as u32
}

/// Owned page-locked byte buffer for Python zero-copy NumPy arrays.
#[cfg(feature = "python")]
pub(super) struct PinnedBytes {
    slice: PinnedHostSlice<u8>,
}

#[cfg(feature = "python")]
impl PinnedBytes {
    pub(super) fn as_mut_ptr(&mut self) -> Result<*mut u8, CudaError> {
        Ok(self.slice.as_mut_ptr()?)
    }
}

/// Device and pinned-host scratch for one pipeline slot, grown on demand and
/// reused across calls.
#[derive(Default)]
struct SlotScratch {
    u16_dev: Option<CudaSlice<u16>>,
    u32_dev: Option<CudaSlice<u32>>,
    i32_dev: Option<CudaSlice<i32>>,
    /// `f32` device frames.
    narrow: Option<CudaSlice<f32>>,
    /// `f64` device frames (used only by `CudaPrecision::F64` with `f64` input).
    wide: Option<CudaSlice<f64>>,
    /// Interleaved device results, sized for `[I, σ]` per bin.
    output: Option<CudaSlice<f64>>,
    /// Compressed Bitshuffle+LZ4 payloads and block descriptors on device.
    comp_dev: Option<CudaSlice<u8>>,
    descs_dev: Option<CudaSlice<u32>>,
    err_dev: Option<CudaSlice<u32>>,
    pinned_u16: Option<PinnedHostSlice<u16>>,
    pinned_u32: Option<PinnedHostSlice<u32>>,
    pinned_i32: Option<PinnedHostSlice<i32>>,
    /// Pinned host staging for `f32` uploads (including CPU-cast `f64 -> f32`).
    pinned_f32: Option<PinnedHostSlice<f32>>,
    /// Pinned host staging for `f64` uploads.
    pinned_f64: Option<PinnedHostSlice<f64>>,
    /// Pinned host staging for compressed Bitshuffle+LZ4 chunks & descriptors.
    pinned_comp: Option<PinnedHostSlice<u8>>,
    pinned_descs: Option<PinnedHostSlice<u32>>,
    pinned_err: Option<PinnedHostSlice<u32>>,
    /// Pinned host staging for interleaved `[I, σ]` downloads.
    pinned_out: Option<PinnedHostSlice<f64>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ActiveInputDtype {
    U16,
    U32,
    I32,
    F32,
    F64,
}

impl SlotScratch {
    /// Drops device and pinned-host input buffers belonging to inactive dtypes
    /// so a plan reused across multiple dtypes does not retain redundant large
    /// allocations.
    fn retain_input_dtype(&mut self, active: ActiveInputDtype) {
        if active != ActiveInputDtype::U16 {
            self.u16_dev = None;
            self.pinned_u16 = None;
        }
        if active != ActiveInputDtype::U32 {
            self.u32_dev = None;
            self.pinned_u32 = None;
        }
        if active != ActiveInputDtype::I32 {
            self.i32_dev = None;
            self.pinned_i32 = None;
        }
        if active != ActiveInputDtype::F32 {
            self.narrow = None;
            self.pinned_f32 = None;
        }
        if active != ActiveInputDtype::F64 {
            self.wide = None;
            self.pinned_f64 = None;
        }
    }
}

/// Uploads `frames` of element type `X` into `dev_slot` on `stream`.
/// If the caller's memory is already CUDA-pinned, it DMAs directly with zero
/// CPU copy; otherwise multi-frame tiles stage via Rayon into `pinned_slot`.
fn upload_same_dtype<'s, X: DeviceRepr + ValidAsZeroBits + Copy + Send + Sync>(
    stream: &Arc<CudaStream>,
    frames: &[&[X]],
    pixels: usize,
    values: usize,
    dev_slot: &'s mut Option<CudaSlice<X>>,
    pinned_slot: &mut Option<PinnedHostSlice<X>>,
    stage_to_pinned: bool,
) -> Result<&'s CudaSlice<X>, CudaError> {
    let input = reserve(stream, dev_slot, values)?;
    let all_pinned = frames.iter().all(|s| is_pinned_host(s.as_ptr().cast()));
    if all_pinned {
        let mut offset = 0;
        for slice in frames {
            let mut dev_dst = input.slice_mut(offset..offset + slice.len());
            stream.memcpy_htod(*slice, &mut dev_dst)?;
            offset += slice.len();
        }
    } else if stage_to_pinned {
        let pinned = reserve_pinned(stream.context(), pinned_slot, values)?;
        stage_copy(frames, &mut pinned.as_mut_slice()?[..values], pixels);
        let mut dev_dst = input.slice_mut(..values);
        stream.memcpy_htod(&pinned.as_slice()?[..values], &mut dev_dst)?;
    } else {
        let mut dev_dst = input.slice_mut(..values);
        stream.memcpy_htod(frames[0], &mut dev_dst)?;
    }
    Ok(input)
}

/// Two-slot double-buffered scratch of a public plan.
#[derive(Default)]
struct Scratch {
    slots: [SlotScratch; 2],
}

/// Returns a device buffer of at least `len` elements, replacing a smaller one.
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

/// Returns a cacheable pinned host buffer of at least `len` elements, replacing
/// a smaller one.
fn reserve_pinned<'s, X: DeviceRepr + ValidAsZeroBits>(
    context: &Arc<CudaContext>,
    slot: &'s mut Option<PinnedHostSlice<X>>,
    len: usize,
) -> Result<&'s mut PinnedHostSlice<X>, CudaError> {
    if !slot.as_ref().is_some_and(|buffer| buffer.len() >= len) {
        *slot = None;
        // SAFETY: `X: ValidAsZeroBits` and the allocation is zero-initialised
        // before any typed slice is created; flags = 0 requests cacheable pages.
        let mut host = unsafe { context.alloc_pinned_with_flags::<X>(len, 0)? };
        unsafe { host.as_mut_ptr()?.write_bytes(0, len) };
        *slot = Some(host);
    }
    slot.as_mut()
        .ok_or_else(|| CudaError::Device("CUDA pinned staging allocation is missing".into()))
}

/// Copies `slices` (either one contiguous stack slice or per-frame slices) into
/// `dst`, using Rayon when large enough to saturate host memory bandwidth.
fn stage_copy<X: Copy + Send + Sync>(slices: &[&[X]], dst: &mut [X], pixels: usize) {
    if slices.len() == 1 {
        let src = slices[0];
        if dst.len() >= PARALLEL_STAGE_THRESHOLD {
            let chunk = dst
                .len()
                .div_ceil(rayon::current_num_threads().max(1))
                .max(PARALLEL_STAGE_THRESHOLD);
            dst.par_chunks_mut(chunk)
                .zip(src.par_chunks(chunk))
                .for_each(|(d, s)| d.copy_from_slice(s));
        } else {
            dst.copy_from_slice(src);
        }
    } else if dst.len() >= PARALLEL_STAGE_THRESHOLD {
        dst.par_chunks_exact_mut(pixels)
            .zip(slices.par_iter())
            .for_each(|(d, s)| d.copy_from_slice(s));
    } else {
        for (d, s) in dst.chunks_exact_mut(pixels).zip(slices.iter()) {
            d.copy_from_slice(s);
        }
    }
}

/// Rounds `f64` frames to `f32` (IEEE 754 round-to-nearest-even, matching
/// `__double2float_rn`) directly into pinned host staging, halving PCIe traffic.
fn stage_f64_to_f32(slices: &[&[f64]], dst: &mut [f32], pixels: usize) {
    if slices.len() == 1 {
        let src = slices[0];
        if dst.len() >= PARALLEL_STAGE_THRESHOLD {
            let chunk = dst
                .len()
                .div_ceil(rayon::current_num_threads().max(1))
                .max(PARALLEL_STAGE_THRESHOLD);
            dst.par_chunks_mut(chunk)
                .zip(src.par_chunks(chunk))
                .for_each(|(d, s)| {
                    for (out, &x) in d.iter_mut().zip(s) {
                        *out = x as f32;
                    }
                });
        } else {
            for (out, &x) in dst.iter_mut().zip(src) {
                *out = x as f32;
            }
        }
    } else if dst.len() >= PARALLEL_STAGE_THRESHOLD {
        dst.par_chunks_exact_mut(pixels)
            .zip(slices.par_iter())
            .for_each(|(d, s)| {
                for (out, &x) in d.iter_mut().zip(s.iter()) {
                    *out = x as f32;
                }
            });
    } else {
        for (d, s) in dst.chunks_exact_mut(pixels).zip(slices.iter()) {
            for (out, &x) in d.iter_mut().zip(s.iter()) {
                *out = x as f32;
            }
        }
    }
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

/// Number of slices in `frames`.
fn frames_len(frames: &Frames<'_, '_>) -> usize {
    match frames {
        Frames::U16(frames) => frames.len(),
        Frames::U32(frames) => frames.len(),
        Frames::I32(frames) => frames.len(),
        Frames::F32(frames) => frames.len(),
        Frames::F64(frames) => frames.len(),
    }
}

/// Uploaded table and the kernels a public plan launches, by storage type.
enum Tables {
    /// `f32` weights: gather kernels by input dtype and `errors`.
    Single {
        plan: Plan<f32>,
        gather: [Kernel<f32>; 2],
        u16: [Kernel<f32, u16>; 2],
        u32: [Kernel<f32, u32>; 2],
        i32: [Kernel<f32, i32>; 2],
    },
    /// `f64` weights: gather kernels by input dtype and `errors`.
    Double {
        plan: Plan<f64>,
        wide: [Kernel<f64>; 2],
        narrow: [Kernel<f64, f32>; 2],
        u16: [Kernel<f64, u16>; 2],
        u32: [Kernel<f64, u32>; 2],
        i32: [Kernel<f64, i32>; 2],
    },
}

/// Device state of a public [`Cuda1dPlan`].
pub(super) struct Engine {
    tables: Tables,
    decode_u16: CudaFunction,
    decode_u32: CudaFunction,
    streams: [Arc<CudaStream>; 2],
    pixels: usize,
    bins: usize,
    frame_bytes: usize,
    max_batch_frames: usize,
    explicit_budget: bool,
    scratch: Mutex<Scratch>,
}

impl Engine {
    fn context(&self) -> &Arc<CudaContext> {
        match &self.tables {
            Tables::Single { plan, .. } => &plan.cuda.context,
            Tables::Double { plan, .. } => &plan.cuda.context,
        }
    }

    /// Selects the frame tile size for double-buffered streaming. Uses 4-frame
    /// tiles on large batches to match the 4-frame CSR-tiled CUDA kernels while
    /// keeping pinned host staging bounded.
    fn pipeline_tile(&self, count: usize, elem_bytes: usize) -> usize {
        let cap = self.max_batch_frames.max(1);
        if count <= 1 || self.pixels < PARALLEL_STAGE_THRESHOLD {
            return count.min(cap).max(1);
        }
        let max_slot =
            (PINNED_SLOT_BYTES / self.pixels.saturating_mul(elem_bytes).max(1)).clamp(1, 16);
        let tile = if count >= 8 {
            if max_slot >= 4 { 4 } else { max_slot }
        } else if count >= 4 {
            max_slot.min(2)
        } else {
            1
        };
        tile.min(cap).max(1)
    }

    #[cfg(feature = "python")]
    pub(super) fn alloc_pinned_bytes(
        &self,
        bytes: usize,
    ) -> Result<super::CudaPinnedBytes, CudaError> {
        let alloc_len = bytes.max(1);
        let context = self.context();
        let slice = guarded(|| {
            // SAFETY: `u8: ValidAsZeroBits` and initialized before exposure;
            // flags = 0 allocates cacheable page-locked memory.
            let mut host = unsafe { context.alloc_pinned_with_flags::<u8>(alloc_len, 0)? };
            unsafe { host.as_mut_ptr()?.write_bytes(0, alloc_len) };
            Ok(host)
        })?;
        Ok(super::CudaPinnedBytes {
            buf: PinnedBytes { slice },
        })
    }

    /// Integrates frames in double-buffered pinned tiles across two streams,
    /// calling `drain(first_frame, end_frame, pinned_slice)` directly as each
    /// tile finishes (avoiding any intermediate host copy).
    fn run_with_sink(
        &self,
        frames: Frames<'_, '_>,
        stacked: bool,
        errors: bool,
        mut drain: impl FnMut(usize, usize, &[f64]),
    ) -> Result<(), CudaError> {
        let count = match frames {
            Frames::U16(frames) if stacked => frame_count(frames, self.pixels)?,
            Frames::U32(frames) if stacked => frame_count(frames, self.pixels)?,
            Frames::I32(frames) if stacked => frame_count(frames, self.pixels)?,
            Frames::F32(frames) if stacked => frame_count(frames, self.pixels)?,
            Frames::F64(frames) if stacked => frame_count(frames, self.pixels)?,
            Frames::U16(frames) => frames.len(),
            Frames::U32(frames) => frames.len(),
            Frames::I32(frames) => frames.len(),
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
        let elem_bytes = match (&self.tables, &frames) {
            (_, Frames::U16(_)) => size_of::<u16>(),
            (Tables::Double { .. }, Frames::F64(_)) => size_of::<f64>(),
            _ => size_of::<f32>(),
        };
        let tile = self.pipeline_tile(count, elem_bytes);
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(PoisonError::into_inner);
            let outcome = guarded(|| {
                if self.explicit_budget && count > tile {
                    let chunk_frames = count.min(self.max_batch_frames);
                    let chunk_bytes = chunk_frames.saturating_mul(self.frame_bytes);
                    let (_, total) = self.context().mem_get_info()?;
                    if chunk_bytes > total {
                        return Err(CudaError::Device(format!(
                            "CUDA chunk of {chunk_frames} frames ({chunk_bytes} bytes) exceeds \
                             total device memory ({total} bytes)"
                        )));
                    }
                }
                let mut pending: [Option<(usize, usize)>; 2] = [None, None];
                for (step, first) in (0..count).step_by(tile).enumerate() {
                    let end = (first + tile).min(count);
                    let slot_idx = if self.max_batch_frames == 1 {
                        0
                    } else {
                        step % 2
                    };
                    let stream = &self.streams[slot_idx];
                    if let Some((p_first, p_end)) = pending[slot_idx].take() {
                        stream.synchronize()?;
                        let len = (p_end - p_first) * per_frame;
                        let pinned = scratch.slots[slot_idx]
                            .pinned_out
                            .as_ref()
                            .ok_or_else(|| CudaError::Device("missing pinned output".into()))?;
                        drain(p_first, p_end, &pinned.as_slice()?[..len]);
                    }
                    let values = first * pixels..end * pixels;
                    let tile_frames = match frames {
                        Frames::U16(frames) if stacked => {
                            let chunk = &frames[0][values];
                            self.enqueue_tile(
                                &mut scratch.slots[slot_idx],
                                stream,
                                Frames::U16(std::slice::from_ref(&chunk)),
                                errors,
                            )?
                        }
                        Frames::U32(frames) if stacked => {
                            let chunk = &frames[0][values];
                            self.enqueue_tile(
                                &mut scratch.slots[slot_idx],
                                stream,
                                Frames::U32(std::slice::from_ref(&chunk)),
                                errors,
                            )?
                        }
                        Frames::I32(frames) if stacked => {
                            let chunk = &frames[0][values];
                            self.enqueue_tile(
                                &mut scratch.slots[slot_idx],
                                stream,
                                Frames::I32(std::slice::from_ref(&chunk)),
                                errors,
                            )?
                        }
                        Frames::F32(frames) if stacked => {
                            let chunk = &frames[0][values];
                            self.enqueue_tile(
                                &mut scratch.slots[slot_idx],
                                stream,
                                Frames::F32(std::slice::from_ref(&chunk)),
                                errors,
                            )?
                        }
                        Frames::F64(frames) if stacked => {
                            let chunk = &frames[0][values];
                            self.enqueue_tile(
                                &mut scratch.slots[slot_idx],
                                stream,
                                Frames::F64(std::slice::from_ref(&chunk)),
                                errors,
                            )?
                        }
                        Frames::U16(frames) => self.enqueue_tile(
                            &mut scratch.slots[slot_idx],
                            stream,
                            Frames::U16(&frames[first..end]),
                            errors,
                        )?,
                        Frames::U32(frames) => self.enqueue_tile(
                            &mut scratch.slots[slot_idx],
                            stream,
                            Frames::U32(&frames[first..end]),
                            errors,
                        )?,
                        Frames::I32(frames) => self.enqueue_tile(
                            &mut scratch.slots[slot_idx],
                            stream,
                            Frames::I32(&frames[first..end]),
                            errors,
                        )?,
                        Frames::F32(frames) => self.enqueue_tile(
                            &mut scratch.slots[slot_idx],
                            stream,
                            Frames::F32(&frames[first..end]),
                            errors,
                        )?,
                        Frames::F64(frames) => self.enqueue_tile(
                            &mut scratch.slots[slot_idx],
                            stream,
                            Frames::F64(&frames[first..end]),
                            errors,
                        )?,
                    };
                    debug_assert_eq!(tile_frames, end - first);
                    pending[slot_idx] = Some((first, end));
                }
                for (slot_idx, range) in pending.into_iter().enumerate() {
                    if let Some((p_first, p_end)) = range {
                        self.streams[slot_idx].synchronize()?;
                        let len = (p_end - p_first) * per_frame;
                        let pinned = scratch.slots[slot_idx]
                            .pinned_out
                            .as_ref()
                            .ok_or_else(|| CudaError::Device("missing pinned output".into()))?;
                        drain(p_first, p_end, &pinned.as_slice()?[..len]);
                    }
                }
                Ok(())
            });
            if outcome.is_err() {
                settle(&self.streams[0]);
                settle(&self.streams[1]);
                *scratch = Scratch::default();
            }
            outcome?;
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn drain_into_vecs(
        bins: usize,
        errors: bool,
        out: &mut [Option<(Vec<f64>, Option<Vec<f64>>)>],
        p_first: usize,
        p_end: usize,
        src: &[f64],
    ) {
        let stride = 1 + usize::from(errors);
        let per_frame = bins * stride;
        for (idx, frame) in src.chunks_exact(per_frame).enumerate() {
            let slot = p_first + idx;
            debug_assert!(slot < p_end);
            if errors {
                let mut intensity = Vec::with_capacity(bins);
                let mut err = Vec::with_capacity(bins);
                for &[i, e] in frame.as_chunks::<2>().0 {
                    intensity.push(i);
                    err.push(e);
                }
                out[slot] = Some((intensity, Some(err)));
            } else {
                out[slot] = Some((frame.to_vec(), None));
            }
        }
    }

    fn drain_into_slices(
        bins: usize,
        out_intensity: &mut [f64],
        out_errors: &mut Option<&mut [f64]>,
        p_first: usize,
        p_end: usize,
        src: &[f64],
    ) {
        let dst_i = &mut out_intensity[p_first * bins..p_end * bins];
        if let Some(err_buf) = out_errors.as_deref_mut() {
            let dst_e = &mut err_buf[p_first * bins..p_end * bins];
            for ((oi, oe), &[i, e]) in dst_i
                .iter_mut()
                .zip(dst_e.iter_mut())
                .zip(src.as_chunks::<2>().0)
            {
                *oi = i;
                *oe = e;
            }
        } else {
            dst_i.copy_from_slice(src);
        }
    }

    /// Integrates frames in double-buffered pinned tiles across two streams,
    /// returning `(intensity, optional_errors)` per frame with no intermediate
    /// host copy.
    #[allow(clippy::type_complexity)]
    pub(super) fn run_raw(
        &self,
        frames: Frames<'_, '_>,
        stacked: bool,
        errors: bool,
    ) -> Result<Vec<(Vec<f64>, Option<Vec<f64>>)>, CudaError> {
        let count = match frames {
            Frames::U16(frames) if stacked => frame_count(frames, self.pixels)?,
            Frames::U32(frames) if stacked => frame_count(frames, self.pixels)?,
            Frames::I32(frames) if stacked => frame_count(frames, self.pixels)?,
            Frames::F32(frames) if stacked => frame_count(frames, self.pixels)?,
            Frames::F64(frames) if stacked => frame_count(frames, self.pixels)?,
            Frames::U16(frames) => frames.len(),
            Frames::U32(frames) => frames.len(),
            Frames::I32(frames) => frames.len(),
            Frames::F32(frames) => frames.len(),
            Frames::F64(frames) => frames.len(),
        };
        let mut out: Vec<Option<(Vec<f64>, Option<Vec<f64>>)>> = vec![None; count];
        let bins = self.bins;
        self.run_with_sink(frames, stacked, errors, |p_first, p_end, src| {
            Self::drain_into_vecs(bins, errors, &mut out, p_first, p_end, src);
        })?;
        Ok(out
            .into_iter()
            .map(|x| x.expect("every frame drained"))
            .collect())
    }

    /// Integrates frames directly into caller-provided contiguous output slices
    /// (`out_intensity.len() == count * bins`, optional `out_errors.len() == count * bins`).
    pub(super) fn run_into(
        &self,
        frames: Frames<'_, '_>,
        stacked: bool,
        out_intensity: &mut [f64],
        mut out_errors: Option<&mut [f64]>,
    ) -> Result<(), CudaError> {
        let errors = out_errors.is_some();
        let bins = self.bins;
        self.run_with_sink(frames, stacked, errors, |p_first, p_end, src| {
            Self::drain_into_slices(bins, out_intensity, &mut out_errors, p_first, p_end, src);
        })
    }

    pub(super) fn run(
        &self,
        frames: Frames<'_, '_>,
        stacked: bool,
        errors: bool,
        radial: &[f64],
    ) -> Result<Vec<Fixed1dResult>, CudaError> {
        Ok(self
            .run_raw(frames, stacked, errors)?
            .into_iter()
            .map(|(intensity, errors)| Fixed1dResult {
                radial: radial.to_vec(),
                intensity,
                errors,
            })
            .collect())
    }

    /// Stages one tile (or DMAs directly if already pinned), enqueues async
    /// H2D, gather kernel and async D2H on `stream`, without synchronising.
    fn enqueue_tile(
        &self,
        slot: &mut SlotScratch,
        stream: &Arc<CudaStream>,
        frames: Frames<'_, '_>,
        errors: bool,
    ) -> Result<usize, CudaError> {
        let pixels = self.pixels;
        let context = self.context();
        let count = match frames {
            Frames::U16(frames) => frame_count(frames, pixels)?,
            Frames::U32(frames) => frame_count(frames, pixels)?,
            Frames::I32(frames) => frame_count(frames, pixels)?,
            Frames::F32(frames) => frame_count(frames, pixels)?,
            Frames::F64(frames) => frame_count(frames, pixels)?,
        };
        let launched = u32::try_from(count)
            .map_err(|_| CudaError::InvalidInput("too many frames in one chunk".into()))?;
        let values = count * pixels;
        let e = usize::from(errors);
        let out_len = count * self.bins * (1 + e);
        let active_dtype = match (&self.tables, &frames) {
            (_, Frames::U16(_)) => ActiveInputDtype::U16,
            (_, Frames::U32(_)) => ActiveInputDtype::U32,
            (_, Frames::I32(_)) => ActiveInputDtype::I32,
            (Tables::Double { .. }, Frames::F64(_)) => ActiveInputDtype::F64,
            (_, Frames::F32(_)) | (Tables::Single { .. }, Frames::F64(_)) => ActiveInputDtype::F32,
        };
        slot.retain_input_dtype(active_dtype);
        let SlotScratch {
            u16_dev,
            u32_dev,
            i32_dev,
            narrow,
            wide,
            output,
            pinned_u16,
            pinned_u32,
            pinned_i32,
            pinned_f32,
            pinned_f64,
            pinned_out,
            ..
        } = slot;
        let output = reserve(stream, output, count * self.bins * 2)?;
        let pinned_out = reserve_pinned(context, pinned_out, count * self.bins * 2)?;
        let stage_to_pinned = count > 1;
        match (&self.tables, frames) {
            (Tables::Single { plan, u16, .. }, Frames::U16(frames)) => {
                let input = upload_same_dtype(
                    stream,
                    frames,
                    pixels,
                    values,
                    u16_dev,
                    pinned_u16,
                    stage_to_pinned,
                )?;
                plan.launch(stream, &u16[e], input, output, launched)?;
            }
            (Tables::Single { plan, u32, .. }, Frames::U32(frames)) => {
                let input = upload_same_dtype(
                    stream,
                    frames,
                    pixels,
                    values,
                    u32_dev,
                    pinned_u32,
                    stage_to_pinned,
                )?;
                plan.launch(stream, &u32[e], input, output, launched)?;
            }
            (Tables::Single { plan, i32, .. }, Frames::I32(frames)) => {
                let input = upload_same_dtype(
                    stream,
                    frames,
                    pixels,
                    values,
                    i32_dev,
                    pinned_i32,
                    stage_to_pinned,
                )?;
                plan.launch(stream, &i32[e], input, output, launched)?;
            }
            (Tables::Single { plan, gather, .. }, Frames::F32(frames)) => {
                let input = upload_same_dtype(
                    stream,
                    frames,
                    pixels,
                    values,
                    narrow,
                    pinned_f32,
                    stage_to_pinned,
                )?;
                plan.launch(stream, &gather[e], input, output, launched)?;
            }
            (Tables::Single { plan, gather, .. }, Frames::F64(frames)) => {
                let pinned = reserve_pinned(context, pinned_f32, values)?;
                stage_f64_to_f32(frames, &mut pinned.as_mut_slice()?[..values], pixels);
                let input = reserve(stream, narrow, values)?;
                let mut dev_dst = input.slice_mut(..values);
                stream.memcpy_htod(&pinned.as_slice()?[..values], &mut dev_dst)?;
                plan.launch(stream, &gather[e], input, output, launched)?;
            }
            (Tables::Double { plan, u16, .. }, Frames::U16(frames)) => {
                let input = upload_same_dtype(
                    stream,
                    frames,
                    pixels,
                    values,
                    u16_dev,
                    pinned_u16,
                    stage_to_pinned,
                )?;
                plan.launch(stream, &u16[e], input, output, launched)?;
            }
            (Tables::Double { plan, u32, .. }, Frames::U32(frames)) => {
                let input = upload_same_dtype(
                    stream,
                    frames,
                    pixels,
                    values,
                    u32_dev,
                    pinned_u32,
                    stage_to_pinned,
                )?;
                plan.launch(stream, &u32[e], input, output, launched)?;
            }
            (Tables::Double { plan, i32, .. }, Frames::I32(frames)) => {
                let input = upload_same_dtype(
                    stream,
                    frames,
                    pixels,
                    values,
                    i32_dev,
                    pinned_i32,
                    stage_to_pinned,
                )?;
                plan.launch(stream, &i32[e], input, output, launched)?;
            }
            (
                Tables::Double {
                    plan,
                    wide: kernels,
                    ..
                },
                Frames::F64(frames),
            ) => {
                *narrow = None;
                let input = upload_same_dtype(
                    stream,
                    frames,
                    pixels,
                    values,
                    wide,
                    pinned_f64,
                    stage_to_pinned,
                )?;
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
                let input = upload_same_dtype(
                    stream,
                    frames,
                    pixels,
                    values,
                    narrow,
                    pinned_f32,
                    stage_to_pinned,
                )?;
                plan.launch(stream, &kernels[e], input, output, launched)?;
            }
        }
        stream.memcpy_dtoh(
            &output.slice(..out_len),
            &mut pinned_out.as_mut_slice()?[..out_len],
        )?;
        Ok(count)
    }

    fn run_bslz4_with_sink(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
        mut drain: impl FnMut(usize, usize, &[f64]),
    ) -> Result<(), CudaError> {
        let count = chunks.len();
        let stride = 1 + usize::from(errors);
        let per_frame = self.bins * stride;
        let tile = self.pipeline_tile(count, dtype.elem_bytes());
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(PoisonError::into_inner);
            let outcome = guarded(|| {
                let mut pending: [Option<(usize, usize)>; 2] = [None, None];
                for (step, first) in (0..count).step_by(tile).enumerate() {
                    let end = (first + tile).min(count);
                    let slot_idx = if self.max_batch_frames == 1 {
                        0
                    } else {
                        step % 2
                    };
                    let stream = &self.streams[slot_idx];
                    if let Some((p_first, p_end)) = pending[slot_idx].take() {
                        stream.synchronize()?;
                        let err = scratch.slots[slot_idx]
                            .pinned_err
                            .as_ref()
                            .map_or(Ok(0u32), |p| p.as_slice().map(|s| s[0]))?;
                        if err != 0 {
                            return Err(CudaError::InvalidInput(
                                "corrupt or truncated Bitshuffle/LZ4 compressed block".into(),
                            ));
                        }
                        let len = (p_end - p_first) * per_frame;
                        let pinned = scratch.slots[slot_idx]
                            .pinned_out
                            .as_ref()
                            .ok_or_else(|| CudaError::Device("missing pinned output".into()))?;
                        drain(p_first, p_end, &pinned.as_slice()?[..len]);
                    }
                    self.enqueue_bslz4_tile(
                        &mut scratch.slots[slot_idx],
                        stream,
                        &chunks[first..end],
                        dtype,
                        errors,
                    )?;
                    pending[slot_idx] = Some((first, end));
                }
                for (slot_idx, range) in pending.into_iter().enumerate() {
                    if let Some((p_first, p_end)) = range {
                        self.streams[slot_idx].synchronize()?;
                        let err = scratch.slots[slot_idx]
                            .pinned_err
                            .as_ref()
                            .map_or(Ok(0u32), |p| p.as_slice().map(|s| s[0]))?;
                        if err != 0 {
                            return Err(CudaError::InvalidInput(
                                "corrupt or truncated Bitshuffle/LZ4 compressed block".into(),
                            ));
                        }
                        let len = (p_end - p_first) * per_frame;
                        let pinned = scratch.slots[slot_idx]
                            .pinned_out
                            .as_ref()
                            .ok_or_else(|| CudaError::Device("missing pinned output".into()))?;
                        drain(p_first, p_end, &pinned.as_slice()?[..len]);
                    }
                }
                Ok(())
            });
            if outcome.is_err() {
                settle(&self.streams[0]);
                settle(&self.streams[1]);
                *scratch = Scratch::default();
            }
            outcome?;
        }
        Ok(())
    }

    /// Decompresses raw Bitshuffle+LZ4 HDF5 chunks on the GPU and integrates
    /// them in VRAM across two double-buffered streams.
    #[allow(clippy::type_complexity)]
    pub(super) fn run_bslz4_raw(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
    ) -> Result<Vec<(Vec<f64>, Option<Vec<f64>>)>, CudaError> {
        let mut out: Vec<Option<(Vec<f64>, Option<Vec<f64>>)>> = vec![None; chunks.len()];
        let bins = self.bins;
        self.run_bslz4_with_sink(chunks, dtype, errors, |p_first, p_end, src| {
            Self::drain_into_vecs(bins, errors, &mut out, p_first, p_end, src);
        })?;
        Ok(out
            .into_iter()
            .map(|x| x.expect("every frame drained"))
            .collect())
    }

    pub(super) fn run_bslz4_into(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        out_intensity: &mut [f64],
        mut out_errors: Option<&mut [f64]>,
    ) -> Result<(), CudaError> {
        let errors = out_errors.is_some();
        let bins = self.bins;
        self.run_bslz4_with_sink(chunks, dtype, errors, |p_first, p_end, src| {
            Self::drain_into_slices(bins, out_intensity, &mut out_errors, p_first, p_end, src);
        })
    }

    pub(super) fn run_bslz4(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
        radial: &[f64],
    ) -> Result<Vec<Fixed1dResult>, CudaError> {
        Ok(self
            .run_bslz4_raw(chunks, dtype, errors)?
            .into_iter()
            .map(|(intensity, errors)| Fixed1dResult {
                radial: radial.to_vec(),
                intensity,
                errors,
            })
            .collect())
    }

    fn enqueue_bslz4_tile(
        &self,
        slot: &mut SlotScratch,
        stream: &Arc<CudaStream>,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
    ) -> Result<(), CudaError> {
        let pixels = self.pixels;
        let elem_bytes = dtype.elem_bytes();
        let expected_uncomp = pixels * elem_bytes;
        let count = chunks.len();
        let launched = u32::try_from(count)
            .map_err(|_| CudaError::InvalidInput("too many frames in one chunk".into()))?;
        let total_comp: usize = chunks.iter().map(|c| c.len()).sum();
        if total_comp > u32::MAX as usize {
            return Err(CudaError::InvalidInput(
                "compressed tile exceeds 4 GiB".into(),
            ));
        }

        // Pre-scan Bitshuffle+LZ4 block headers (~10 us per Eiger 9M frame).
        let mut descs: Vec<u32> = Vec::with_capacity(count * (expected_uncomp / 8192 + 2) * 4);
        let mut max_block_bytes: u32 = 0;
        let mut base_off: u32 = 0;
        let u32_overflow =
            || CudaError::InvalidInput("Bitshuffle/LZ4 offset exceeds 32-bit range".into());
        for (frame_idx, chunk) in chunks.iter().enumerate() {
            if chunk.len() < 12 {
                return Err(CudaError::InvalidInput(
                    "Bitshuffle/LZ4 chunk shorter than 12-byte header".into(),
                ));
            }
            let uncomp_bytes = u64::from_be_bytes(chunk[0..8].try_into().unwrap()) as usize;
            if uncomp_bytes != expected_uncomp {
                return Err(CudaError::InvalidInput(format!(
                    "Bitshuffle/LZ4 chunk uncompressed size {uncomp_bytes} does not match \
                     plan frame size {expected_uncomp} bytes"
                )));
            }
            let raw_block = u32::from_be_bytes(chunk[8..12].try_into().unwrap()) as usize;
            let block_bytes = if raw_block == 0 { 8192 } else { raw_block };
            if block_bytes == 0
                || block_bytes > 49_152
                || !block_bytes.is_multiple_of(8 * elem_bytes)
            {
                return Err(CudaError::InvalidInput(format!(
                    "unsupported Bitshuffle/LZ4 block size {block_bytes} bytes"
                )));
            }
            let block_bytes_u32 = u32::try_from(block_bytes).map_err(|_| u32_overflow())?;
            max_block_bytes = max_block_bytes.max(block_bytes_u32);
            let max_full_comp = block_bytes + block_bytes / 255 + 16;
            let block_elems = block_bytes / elem_bytes;
            let n_full = pixels / block_elems;
            let rem_elems = pixels % block_elems;
            let rem_comp_elems = (rem_elems / 8) * 8;
            let tail_raw_elems = rem_elems % 8;
            let frame_dst = frame_idx * pixels;

            let mut off = 12usize;
            for b in 0..n_full {
                if off.checked_add(4).is_none_or(|end| end > chunk.len()) {
                    return Err(CudaError::InvalidInput(
                        "truncated Bitshuffle/LZ4 block header".into(),
                    ));
                }
                let comp_len = u32::from_be_bytes(chunk[off..off + 4].try_into().unwrap()) as usize;
                off += 4;
                if comp_len > max_full_comp {
                    return Err(CudaError::InvalidInput(format!(
                        "Bitshuffle/LZ4 compressed block length {comp_len} exceeds LZ4 bound {max_full_comp}"
                    )));
                }
                if off
                    .checked_add(comp_len)
                    .is_none_or(|end| end > chunk.len())
                {
                    return Err(CudaError::InvalidInput(
                        "truncated Bitshuffle/LZ4 block payload".into(),
                    ));
                }
                let src_off = base_off
                    .checked_add(u32::try_from(off).map_err(|_| u32_overflow())?)
                    .ok_or_else(u32_overflow)?;
                let comp_len_u32 = u32::try_from(comp_len).map_err(|_| u32_overflow())?;
                let dst_off =
                    u32::try_from(frame_dst + b * block_elems).map_err(|_| u32_overflow())?;
                descs.push(src_off);
                descs.push(comp_len_u32);
                descs.push(dst_off);
                descs.push(block_bytes_u32);
                off += comp_len;
            }
            if rem_comp_elems > 0 {
                if off.checked_add(4).is_none_or(|end| end > chunk.len()) {
                    return Err(CudaError::InvalidInput(
                        "truncated Bitshuffle/LZ4 partial block header".into(),
                    ));
                }
                let comp_len = u32::from_be_bytes(chunk[off..off + 4].try_into().unwrap()) as usize;
                off += 4;
                let rem_bytes = rem_comp_elems * elem_bytes;
                let max_rem_comp = rem_bytes + rem_bytes / 255 + 16;
                if comp_len > max_rem_comp {
                    return Err(CudaError::InvalidInput(format!(
                        "Bitshuffle/LZ4 partial compressed block length {comp_len} exceeds LZ4 bound {max_rem_comp}"
                    )));
                }
                if off
                    .checked_add(comp_len)
                    .is_none_or(|end| end > chunk.len())
                {
                    return Err(CudaError::InvalidInput(
                        "truncated Bitshuffle/LZ4 partial block payload".into(),
                    ));
                }
                let src_off = base_off
                    .checked_add(u32::try_from(off).map_err(|_| u32_overflow())?)
                    .ok_or_else(u32_overflow)?;
                let comp_len_u32 = u32::try_from(comp_len).map_err(|_| u32_overflow())?;
                let dst_off =
                    u32::try_from(frame_dst + n_full * block_elems).map_err(|_| u32_overflow())?;
                let rem_bytes_u32 = u32::try_from(rem_bytes).map_err(|_| u32_overflow())?;
                descs.push(src_off);
                descs.push(comp_len_u32);
                descs.push(dst_off);
                descs.push(rem_bytes_u32);
                off += comp_len;
            }
            if tail_raw_elems > 0 {
                let raw_bytes = tail_raw_elems * elem_bytes;
                if off
                    .checked_add(raw_bytes)
                    .is_none_or(|end| end > chunk.len())
                {
                    return Err(CudaError::InvalidInput(
                        "truncated Bitshuffle/LZ4 tail bytes".into(),
                    ));
                }
                let src_off = base_off
                    .checked_add(u32::try_from(off).map_err(|_| u32_overflow())?)
                    .ok_or_else(u32_overflow)?;
                let raw_bytes_u32 = u32::try_from(raw_bytes).map_err(|_| u32_overflow())?;
                let dst_off = u32::try_from(frame_dst + n_full * block_elems + rem_comp_elems)
                    .map_err(|_| u32_overflow())?;
                descs.push(src_off);
                descs.push(0x8000_0000 | raw_bytes_u32);
                descs.push(dst_off);
                descs.push(raw_bytes_u32);
            }
            base_off = base_off
                .checked_add(u32::try_from(chunk.len()).map_err(|_| u32_overflow())?)
                .ok_or_else(u32_overflow)?;
        }

        let n_blocks = u32::try_from(descs.len() / 4).map_err(|_| u32_overflow())?;
        let values = count * pixels;
        let e = usize::from(errors);
        let out_len = count * self.bins * (1 + e);
        let context = self.context();

        slot.retain_input_dtype(match dtype {
            Bslz4Dtype::U16 => ActiveInputDtype::U16,
            Bslz4Dtype::U32 => ActiveInputDtype::U32,
            Bslz4Dtype::I32 => ActiveInputDtype::I32,
        });
        let SlotScratch {
            u16_dev,
            u32_dev,
            i32_dev,
            output,
            comp_dev,
            descs_dev,
            err_dev,
            pinned_comp,
            pinned_descs,
            pinned_err,
            pinned_out,
            ..
        } = slot;

        let pinned_c = reserve_pinned(context, pinned_comp, total_comp.max(1))?;
        {
            let dst = &mut pinned_c.as_mut_slice()?[..total_comp];
            if total_comp >= PARALLEL_STAGE_THRESHOLD && chunks.len() > 1 {
                let mut rem = dst;
                let mut dst_slices: Vec<&mut [u8]> = Vec::with_capacity(chunks.len());
                for chunk in chunks {
                    let (head, tail) = rem.split_at_mut(chunk.len());
                    dst_slices.push(head);
                    rem = tail;
                }
                dst_slices
                    .into_par_iter()
                    .zip(chunks.par_iter())
                    .for_each(|(d, s)| d.copy_from_slice(s));
            } else {
                let mut off = 0;
                for chunk in chunks {
                    dst[off..off + chunk.len()].copy_from_slice(chunk);
                    off += chunk.len();
                }
            }
        }
        let pinned_d = reserve_pinned(context, pinned_descs, descs.len().max(1))?;
        pinned_d.as_mut_slice()?[..descs.len()].copy_from_slice(&descs);
        let pinned_e = reserve_pinned(context, pinned_err, 1)?;
        pinned_e.as_mut_slice()?[0] = 0;

        let comp_buf = reserve(stream, comp_dev, total_comp.max(1))?;
        let descs_buf = reserve(stream, descs_dev, descs.len().max(1))?;
        let err_buf = reserve(stream, err_dev, 1)?;
        let output = reserve(stream, output, count * self.bins * 2)?;
        let pinned_out = reserve_pinned(context, pinned_out, count * self.bins * 2)?;

        let mut comp_dst = comp_buf.slice_mut(..total_comp);
        stream.memcpy_htod(&pinned_c.as_slice()?[..total_comp], &mut comp_dst)?;
        let mut descs_dst = descs_buf.slice_mut(..descs.len());
        stream.memcpy_htod(&pinned_d.as_slice()?[..descs.len()], &mut descs_dst)?;
        let mut err_dst = err_buf.slice_mut(..1);
        stream.memcpy_htod(&pinned_e.as_slice()?[..1], &mut err_dst)?;

        let decode_cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: max_block_bytes,
        };
        let comp_view = comp_buf.slice(..total_comp);
        let descs_view = descs_buf.slice(..descs.len());
        let mut err_view = err_buf.slice_mut(..1);

        match (&self.tables, dtype) {
            (Tables::Single { plan, u16, .. }, Bslz4Dtype::U16) => {
                let input = reserve(stream, u16_dev, values)?;
                let mut out_px = input.slice_mut(..values);
                let mut launch = stream.launch_builder(&self.decode_u16);
                launch
                    .arg(&comp_view)
                    .arg(&descs_view)
                    .arg(&mut out_px)
                    .arg(&mut err_view);
                unsafe { launch.launch(decode_cfg)? };
                plan.launch(stream, &u16[e], input, output, launched)?;
            }
            (Tables::Single { plan, u32, .. }, Bslz4Dtype::U32) => {
                let input = reserve(stream, u32_dev, values)?;
                let mut out_px = input.slice_mut(..values);
                let mut launch = stream.launch_builder(&self.decode_u32);
                launch
                    .arg(&comp_view)
                    .arg(&descs_view)
                    .arg(&mut out_px)
                    .arg(&mut err_view);
                unsafe { launch.launch(decode_cfg)? };
                plan.launch(stream, &u32[e], input, output, launched)?;
            }
            (Tables::Single { plan, i32, .. }, Bslz4Dtype::I32) => {
                let input = reserve(stream, i32_dev, values)?;
                let mut out_px = input.slice_mut(..values);
                let mut launch = stream.launch_builder(&self.decode_u32);
                launch
                    .arg(&comp_view)
                    .arg(&descs_view)
                    .arg(&mut out_px)
                    .arg(&mut err_view);
                unsafe { launch.launch(decode_cfg)? };
                plan.launch(stream, &i32[e], input, output, launched)?;
            }
            (Tables::Double { plan, u16, .. }, Bslz4Dtype::U16) => {
                let input = reserve(stream, u16_dev, values)?;
                let mut out_px = input.slice_mut(..values);
                let mut launch = stream.launch_builder(&self.decode_u16);
                launch
                    .arg(&comp_view)
                    .arg(&descs_view)
                    .arg(&mut out_px)
                    .arg(&mut err_view);
                unsafe { launch.launch(decode_cfg)? };
                plan.launch(stream, &u16[e], input, output, launched)?;
            }
            (Tables::Double { plan, u32, .. }, Bslz4Dtype::U32) => {
                let input = reserve(stream, u32_dev, values)?;
                let mut out_px = input.slice_mut(..values);
                let mut launch = stream.launch_builder(&self.decode_u32);
                launch
                    .arg(&comp_view)
                    .arg(&descs_view)
                    .arg(&mut out_px)
                    .arg(&mut err_view);
                unsafe { launch.launch(decode_cfg)? };
                plan.launch(stream, &u32[e], input, output, launched)?;
            }
            (Tables::Double { plan, i32, .. }, Bslz4Dtype::I32) => {
                let input = reserve(stream, i32_dev, values)?;
                let mut out_px = input.slice_mut(..values);
                let mut launch = stream.launch_builder(&self.decode_u32);
                launch
                    .arg(&comp_view)
                    .arg(&descs_view)
                    .arg(&mut out_px)
                    .arg(&mut err_view);
                unsafe { launch.launch(decode_cfg)? };
                plan.launch(stream, &i32[e], input, output, launched)?;
            }
        }
        stream.memcpy_dtoh(&err_buf.slice(..1), &mut pinned_e.as_mut_slice()?[..1])?;
        stream.memcpy_dtoh(
            &output.slice(..out_len),
            &mut pinned_out.as_mut_slice()?[..out_len],
        )?;
        Ok(())
    }
}

fn build_engine(
    cuda: &Cuda,
    table: &HostTable<'_>,
    pixels: usize,
    bins: usize,
    options: &Cuda1dOptions,
    allow_subwarp: bool,
) -> Result<(Engine, usize, usize), CudaError> {
    let stream1 = guarded(|| Ok(cuda.context.new_stream()?))?;
    let streams = [Arc::clone(&cuda.stream), stream1];
    let precision = options.precision();
    let layout = if allow_subwarp
        && table.entries as usize <= bins.saturating_mul(SUBWARP_MAX_ENTRIES_PER_BIN)
    {
        Layout::SubWarp
    } else if table.entries as usize <= bins.saturating_mul(WARP_MAX_ENTRIES_PER_BIN) {
        Layout::Warp
    } else {
        Layout::Block
    };
    let tables = match precision {
        CudaPrecision::F32 | CudaPrecision::F32Compensated => {
            let plan = cuda.upload(table, [1.0_f32], encode_f32)?;
            let mode = if precision == CudaPrecision::F32 {
                Accumulation::Plain
            } else {
                Accumulation::Compensated
            };
            let gather = both(|errors| plan.kernel(mode, layout, errors))?;
            let u16 = both(|errors| plan.kernel_u16(mode, layout, errors))?;
            let u32 = both(|errors| plan.kernel_u32(mode, layout, errors))?;
            let i32 = both(|errors| plan.kernel_i32(mode, layout, errors))?;
            Tables::Single {
                plan,
                gather,
                u16,
                u32,
                i32,
            }
        }
        CudaPrecision::F64 => {
            let plan = cuda.upload(table, [1.0_f64], |weights| Cow::Borrowed(weights))?;
            let wide = both(|errors| plan.kernel(layout, errors))?;
            let narrow = both(|errors| plan.kernel_f32(layout, errors))?;
            let u16 = both(|errors| plan.kernel_u16(layout, errors))?;
            let u32 = both(|errors| plan.kernel_u32(layout, errors))?;
            let i32 = both(|errors| plan.kernel_i32(layout, errors))?;
            Tables::Double {
                plan,
                wide,
                narrow,
                u16,
                u32,
                i32,
            }
        }
    };
    let entries = table.entries as usize;
    let storage = if precision == CudaPrecision::F64 {
        size_of::<f64>()
    } else {
        size_of::<f32>() + size_of::<f64>()
    };
    let frame_bytes = pixels
        .checked_mul(storage)
        .and_then(|bytes| bytes.checked_add(bins.checked_mul(2 * size_of::<f64>())?))
        .ok_or_else(|| CudaError::InvalidInput("one CUDA frame exceeds usize bytes".into()))?;
    let explicit_budget = options.max_batch_bytes().is_some();
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
    let decode_u16 = cuda
        .module
        .load_function("bshuf_lz4_decode_u16")
        .map_err(|error| {
            CudaError::Unavailable(format!(
                "CUDA kernel bshuf_lz4_decode_u16 is missing: {error}"
            ))
        })?;
    let decode_u32 = cuda
        .module
        .load_function("bshuf_lz4_decode_u32")
        .map_err(|error| {
            CudaError::Unavailable(format!(
                "CUDA kernel bshuf_lz4_decode_u32 is missing: {error}"
            ))
        })?;
    let engine = Engine {
        tables,
        decode_u16,
        decode_u32,
        streams,
        pixels,
        bins,
        frame_bytes,
        max_batch_frames,
        explicit_budget,
        scratch: Mutex::new(Scratch::default()),
    };
    Ok((engine, entries, max_batch_frames))
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
    let (engine, entries, max_batch_frames, radial) =
        with_table(integrator, shape, n_points, |table| {
            let (engine, entries, max_batch_frames) =
                build_engine(&cuda, table, pixels, n_points, options, false)?;
            Ok((engine, entries, max_batch_frames, table.radial.clone()))
        })?;
    Ok(Cuda1dPlan {
        shape,
        n_points,
        precision: options.precision(),
        device: options.device(),
        device_name: cuda.info.device_name.clone().unwrap_or_default(),
        radial,
        table_entries: entries,
        max_batch_frames,
        engine,
    })
}

/// Steps 6 to 8 of [`Integrator::prepare_cuda2d`]; steps 1 to 4 have passed.
pub(super) fn prepare_2d(
    integrator: &Integrator,
    shape: (usize, usize),
    pixels: usize,
    n_radial: usize,
    n_azimuthal: usize,
    azimuthal_range: Option<(f64, f64)>,
    options: &Cuda1dOptions,
) -> Result<super::Cuda2dPlan, CudaError> {
    let cuda = Cuda::open(options.device())?;
    let total_bins = n_radial
        .checked_mul(n_azimuthal)
        .ok_or_else(|| CudaError::InvalidInput("2D bin count overflows usize".into()))?;
    let (engine, entries, max_batch_frames, radial, azimuthal) = with_table_2d(
        integrator,
        shape,
        n_radial,
        n_azimuthal,
        azimuthal_range,
        |table, radial, azimuthal| {
            let (engine, entries, max_batch_frames) =
                build_engine(&cuda, table, pixels, total_bins, options, true)?;
            Ok((engine, entries, max_batch_frames, radial, azimuthal))
        },
    )?;
    Ok(super::Cuda2dPlan {
        shape,
        n_radial,
        n_azimuthal,
        precision: options.precision(),
        device: options.device(),
        device_name: cuda.info.device_name.clone().unwrap_or_default(),
        radial,
        azimuthal,
        table_entries: entries,
        max_batch_frames,
        engine,
    })
}

pub(super) fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}
