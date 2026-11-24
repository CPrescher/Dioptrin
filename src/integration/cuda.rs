//! Experimental CUDA 1D integration of fixed-mask lookup tables.
//!
//! **Experimental:** everything in this module is excluded from semver
//! guarantees until it is stabilised.
//!
//! The public items are always compiled, so the same code builds with and
//! without the `cuda` cargo feature. Without the feature,
//! [`Integrator::prepare_cuda1d`] still checks the configuration and then
//! returns [`CudaError::NotCompiled`], and [`cuda_info`] reports
//! `compiled: false`.
//!
//! With the feature, the crate loads the CUDA driver (`libcuda`) and NVRTC
//! (`libnvrtc`) at run time; neither the CUDA toolkit nor a GPU is needed to
//! build. The kernels are compiled by NVRTC for the device's compute capability
//! on first use of a device, once per process and device.
//!
//! A [`Cuda1dPlan`] is a snapshot of an [`Integrator`]'s fixed-mask CSR lookup
//! table, corrections (solid angle, polarisation), radial unit and radial
//! range, uploaded to one device. It integrates `f32` or `f64` frames and
//! returns the same [`Fixed1dResult`] as the CPU plans:
//!
//! - Bin denominators are summed on the CPU in strict `f64` when the plan is
//!   prepared; the device divides in `f64` and returns 0 for a zero
//!   denominator, as the CPU does. The error is `sqrt(max(var, 0)) / |den|`
//!   with `var = Σ w² · max(signal, 0)`.
//! - [`CudaPrecision`] selects the storage and accumulation precision.
//! - The device gathers the rows of the masked table in its own fixed order,
//!   so its results agree with CPU [`Integrator::integrate1d`] within the
//!   tolerance of the chosen precision, not bit for bit. Results are
//!   deterministic, and the single-frame and multi-frame calls give
//!   bit-identical results for the same frame.
//! - [`Integrator::with_fast_math`] and [`Integrator::with_sparse_layout`] do
//!   not affect the device.
//!
//! CUDA is not fork-safe once initialised: do not fork a process (for example
//! with Python `multiprocessing` and the `fork` start method) after using this
//! module in it.
// The parent `integration` module allows dead code; re-enable the lint here so
// items that only the tests and benchmarks use stay behind `cfg(test)`.
#![warn(dead_code)]
use std::{fmt, str::FromStr};

use super::{Fixed1dResult, IntegrationMethod, Integrator, MaskMode};

#[cfg(feature = "cuda")]
mod device;
// The prototype benchmarks reach the internal device types through `super::*`.
#[cfg(all(feature = "cuda", test))]
use device::*;

#[cfg(all(feature = "cuda", test))]
mod bench;
#[cfg(test)]
mod tests;

/// Storage and accumulation precision of a [`Cuda1dPlan`].
///
/// **Experimental:** excluded from semver guarantees until stabilised.
///
/// In every mode the bin denominators are computed on the CPU in `f64`, and
/// the division, square root and output are `f64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CudaPrecision {
    /// `f32` pixels and weights, `f32` fused multiply-add accumulation.
    F32,
    /// `f32` pixels and weights, double-word (hi, lo) compensated `f32`
    /// accumulation. The default.
    #[default]
    F32Compensated,
    /// `f64` pixels and weights, `f64` accumulation.
    F64,
}

impl CudaPrecision {
    /// Name of the precision: `"f32"`, `"f32_compensated"` or `"f64"`.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F32Compensated => "f32_compensated",
            Self::F64 => "f64",
        }
    }
}

impl fmt::Display for CudaPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for CudaPrecision {
    type Err = CudaError;

    /// Parses `"f32"`, `"f32_compensated"` or `"f64"`; anything else is
    /// [`CudaError::InvalidInput`].
    fn from_str(name: &str) -> Result<Self, CudaError> {
        match name {
            "f32" => Ok(Self::F32),
            "f32_compensated" => Ok(Self::F32Compensated),
            "f64" => Ok(Self::F64),
            _ => Err(CudaError::InvalidInput(format!(
                "unknown CUDA precision {name:?}; expected \"f32\", \"f32_compensated\" or \"f64\""
            ))),
        }
    }
}

/// Options for [`Integrator::prepare_cuda1d`].
///
/// **Experimental:** excluded from semver guarantees until stabilised.
///
/// The defaults are [`CudaPrecision::F32Compensated`], device 0 and an
/// automatic chunk budget (see [`Cuda1dOptions::with_max_batch_bytes`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cuda1dOptions {
    precision: CudaPrecision,
    device: usize,
    max_batch_bytes: Option<usize>,
}

impl Cuda1dOptions {
    /// Default options: compensated `f32`, device 0, automatic chunk budget.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects the storage and accumulation precision.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn with_precision(mut self, precision: CudaPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// Selects the CUDA device by ordinal, counted among the devices visible
    /// to the driver (`CUDA_VISIBLE_DEVICES` is honoured). An ordinal out of
    /// range makes [`Integrator::prepare_cuda1d`] return
    /// [`CudaError::Unavailable`].
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn with_device(mut self, ordinal: usize) -> Self {
        self.device = ordinal;
        self
    }

    /// Caps the device scratch memory of one chunk of frames.
    ///
    /// Multi-frame calls copy, integrate and download frames in chunks of at
    /// most [`Cuda1dPlan::max_batch_frames`] frames. One frame costs
    /// `pixels × size_of(storage)` bytes, plus `pixels × 8` for the `f64`
    /// staging buffer unless the precision is [`CudaPrecision::F64`], plus
    /// `n_points × 16` for the results. Without this option the budget is
    /// `min(1 GiB, free device memory after the table upload / 2)`.
    ///
    /// Zero, or a budget smaller than one frame, makes
    /// [`Integrator::prepare_cuda1d`] return [`CudaError::InvalidInput`].
    /// An explicit budget is not checked against the free device memory: if
    /// a chunk does not fit, the integration call that needs it returns
    /// [`CudaError::Device`] (out of memory) and the plan stays usable with
    /// smaller calls.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn with_max_batch_bytes(mut self, bytes: usize) -> Self {
        self.max_batch_bytes = Some(bytes);
        self
    }

    /// The selected precision.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn precision(&self) -> CudaPrecision {
        self.precision
    }

    /// The selected device ordinal.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn device(&self) -> usize {
        self.device
    }

    /// The explicit chunk budget in bytes, if one was set.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn max_batch_bytes(&self) -> Option<usize> {
        self.max_batch_bytes
    }
}

/// Error of the CUDA integration API.
///
/// **Experimental:** excluded from semver guarantees until stabilised.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CudaError {
    /// The crate was built without the `cuda` cargo feature.
    NotCompiled,
    /// CUDA cannot be used: no driver library, no device or no such ordinal,
    /// NVRTC missing, or the kernels failed to compile or load for the device.
    Unavailable(String),
    /// The integrator configuration is not supported on the device (dynamic
    /// mask mode, histogram method, or a table too large for
    /// 32-bit indices).
    Unsupported(String),
    /// Invalid shapes, sizes, frame lengths or memory budgets.
    InvalidInput(String),
    /// A CUDA driver error at run time, such as running out of device memory
    /// or a failed kernel launch.
    ///
    /// The driver also reports some failures later, for example while freeing
    /// the memory of a dropped plan; such an error is returned by the next
    /// call on the same device, which may belong to another plan.
    Device(String),
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCompiled => f.write_str(
                "dioptrin was built without CUDA support; enable the `cuda` cargo feature",
            ),
            Self::Unavailable(reason) => write!(f, "CUDA unavailable: {reason}"),
            Self::Unsupported(reason) => write!(f, "unsupported CUDA configuration: {reason}"),
            Self::InvalidInput(reason) => write!(f, "invalid CUDA input: {reason}"),
            Self::Device(reason) => write!(f, "CUDA device error: {reason}"),
        }
    }
}

impl std::error::Error for CudaError {}

/// Description of the CUDA runtime and one device, from [`cuda_info`].
///
/// **Experimental:** excluded from semver guarantees until stabilised.
///
/// Fields that could not be determined are `None`; when initialisation fails
/// part-way, the fields read before the failure are still filled in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaInfo {
    /// Whether the crate was built with the `cuda` feature.
    pub compiled: bool,
    /// Whether the device is initialised and the kernels are loaded.
    pub available: bool,
    /// Why the device is unavailable; `Some` whenever `available` is false.
    pub reason: Option<String>,
    /// Device ordinal this description refers to.
    pub device: usize,
    /// Device name reported by the driver.
    pub device_name: Option<String>,
    /// Compute capability `(major, minor)`.
    pub compute_capability: Option<(i32, i32)>,
    /// Total device memory in bytes.
    pub total_memory_bytes: Option<usize>,
    /// Driver version from `cuDriverGetVersion`, e.g. `13040` for CUDA 13.4.
    pub driver_version: Option<i32>,
    /// NVRTC version `(major, minor)` of the library that was loaded.
    pub nvrtc_version: Option<(i32, i32)>,
    /// CUDA API version of the compiled bindings, `(13, 4)` for cudarc's
    /// `cuda-13040` feature; `None` without the `cuda` feature.
    pub bindings_version: Option<(u32, u32)>,
}

impl CudaInfo {
    fn unavailable(device: usize, compiled: bool, reason: String) -> Self {
        Self {
            compiled,
            available: false,
            reason: Some(reason),
            device,
            device_name: None,
            compute_capability: None,
            total_memory_bytes: None,
            driver_version: None,
            nvrtc_version: None,
            bindings_version: None,
        }
    }
}

/// Describes CUDA support and device `device`. Never panics.
///
/// **Experimental:** excluded from semver guarantees until stabilised.
///
/// With the `cuda` feature this performs the full device initialisation
/// (driver, context, NVRTC compile of the kernels and module load). A
/// successful initialisation is cached for the process per device ordinal, and
/// later plans on that device reuse it; a failed one is retried on the next
/// call. Without the feature it returns `compiled: false` and
/// `available: false`.
pub fn cuda_info(device: usize) -> CudaInfo {
    #[cfg(feature = "cuda")]
    {
        match device::device(device) {
            Ok(device) => device.info.clone(),
            Err(info) => info,
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        CudaInfo::unavailable(
            device,
            false,
            "dioptrin was built without the `cuda` cargo feature".into(),
        )
    }
}

mod sealed {
    /// Borrowed frames of one supported pixel type.
    pub enum Frames<'a, 'b> {
        U16(&'a [&'b [u16]]),
        U32(&'a [&'b [u32]]),
        I32(&'a [&'b [i32]]),
        F32(&'a [&'b [f32]]),
        F64(&'a [&'b [f64]]),
    }

    pub trait Sealed {
        fn frames<'a, 'b>(frames: &'a [&'b [Self]]) -> Frames<'a, 'b>
        where
            Self: Sized;
    }

    impl Sealed for u16 {
        fn frames<'a, 'b>(frames: &'a [&'b [u16]]) -> Frames<'a, 'b> {
            Frames::U16(frames)
        }
    }

    impl Sealed for u32 {
        fn frames<'a, 'b>(frames: &'a [&'b [u32]]) -> Frames<'a, 'b> {
            Frames::U32(frames)
        }
    }

    impl Sealed for i32 {
        fn frames<'a, 'b>(frames: &'a [&'b [i32]]) -> Frames<'a, 'b> {
            Frames::I32(frames)
        }
    }

    impl Sealed for f32 {
        fn frames<'a, 'b>(frames: &'a [&'b [f32]]) -> Frames<'a, 'b> {
            Frames::F32(frames)
        }
    }

    impl Sealed for f64 {
        fn frames<'a, 'b>(frames: &'a [&'b [f64]]) -> Frames<'a, 'b> {
            Frames::F64(frames)
        }
    }
}

/// Pixel element type of raw Bitshuffle+LZ4 HDF5 chunks passed to
/// [`Cuda1dPlan::integrate_bslz4`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bslz4Dtype {
    U16,
    U32,
    I32,
}

#[cfg(feature = "cuda")]
impl Bslz4Dtype {
    pub(crate) fn elem_bytes(self) -> usize {
        match self {
            Self::U16 => size_of::<u16>(),
            Self::U32 | Self::I32 => size_of::<u32>(),
        }
    }
}

/// Pixel types a [`Cuda1dPlan`] accepts: `u16`, `u32`, `i32`, `f32` and `f64`. Sealed.
///
/// **Experimental:** excluded from semver guarantees until stabilised.
///
/// Integer frames (`u16`, `u32`, `i32`) are uploaded natively and converted
/// in-register on the device; `f64` frames are rounded to `f32` (matching
/// `__double2float_rn`) unless the precision is [`CudaPrecision::F64`]; `f32`
/// frames are widened exactly for [`CudaPrecision::F64`].
pub trait CudaPixel: sealed::Sealed + Copy + Send + Sync + 'static {}
impl CudaPixel for u16 {}
impl CudaPixel for u32 {}
impl CudaPixel for i32 {}
impl CudaPixel for f32 {}
impl CudaPixel for f64 {}

#[cfg(feature = "cuda")]
type Engine = device::Engine;
#[cfg(not(feature = "cuda"))]
type Engine = std::convert::Infallible;

/// A fixed-mask 1D integration table uploaded to one CUDA device.
///
/// **Experimental:** excluded from semver guarantees until stabilised.
///
/// Created by [`Integrator::prepare_cuda1d`]. The plan is a snapshot: later
/// changes to the integrator (mask, corrections, method, unit, radial range)
/// do not affect it. It owns its device memory and its own CUDA stream, and
/// frees them when dropped.
///
/// The plan is `Send + Sync`. Calls on one plan are serialised by an internal
/// lock that guards the reusable device scratch buffers; separate plans may
/// run concurrently. Every call copies the frames to the device, integrates
/// them, copies the results back and synchronises before it returns.
pub struct Cuda1dPlan {
    shape: (usize, usize),
    n_points: usize,
    precision: CudaPrecision,
    device: usize,
    device_name: String,
    radial: Vec<f64>,
    table_entries: usize,
    max_batch_frames: usize,
    engine: Engine,
}

impl fmt::Debug for Cuda1dPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cuda1dPlan")
            .field("shape", &self.shape)
            .field("n_points", &self.n_points)
            .field("precision", &self.precision)
            .field("device", &self.device)
            .field("device_name", &self.device_name)
            .field("table_entries", &self.table_entries)
            .field("max_batch_frames", &self.max_batch_frames)
            .finish_non_exhaustive()
    }
}

impl Cuda1dPlan {
    /// Image shape `(dim1, dim2)` every frame must have.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Number of radial bins.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Storage and accumulation precision.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn precision(&self) -> CudaPrecision {
        self.precision
    }

    /// Ordinal of the device holding the plan.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn device(&self) -> usize {
        self.device
    }

    /// Name of the device holding the plan.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Radial bin centres; identical to the radial axis of CPU
    /// [`Integrator::integrate1d`] for the same integrator and `n_points`.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn radial(&self) -> &[f64] {
        &self.radial
    }

    /// Number of (unmasked) entries of the uploaded table.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn table_entries(&self) -> usize {
        self.table_entries
    }

    /// Maximum number of frames copied to the device in one chunk, sized for
    /// the worst case of `f64` input (see
    /// [`Cuda1dOptions::with_max_batch_bytes`]).
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn max_batch_frames(&self) -> usize {
        self.max_batch_frames
    }

    /// Integrates one frame of `dim1 × dim2` pixels in row-major order.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    ///
    /// `errors` requests the propagated standard deviation. A frame of the
    /// wrong length is [`CudaError::InvalidInput`]; driver failures are
    /// [`CudaError::Device`].
    pub fn integrate_frame<P: CudaPixel>(
        &self,
        frame: &[P],
        errors: bool,
    ) -> Result<Fixed1dResult, CudaError> {
        let mut results = self.integrate_frames(&[frame], errors)?;
        results
            .pop()
            .ok_or_else(|| CudaError::Device("CUDA integration returned no result".into()))
    }

    /// Integrates several frames, in chunks of at most
    /// [`max_batch_frames`](Self::max_batch_frames) frames.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    ///
    /// Every frame length is checked before anything is copied. The results
    /// are bit-identical to [`integrate_frame`](Self::integrate_frame) on each
    /// frame. An empty slice returns an empty vector.
    pub fn integrate_frames<P: CudaPixel>(
        &self,
        frames: &[&[P]],
        errors: bool,
    ) -> Result<Vec<Fixed1dResult>, CudaError> {
        let pixels = self.shape.0 * self.shape.1;
        if let Some((index, frame)) = frames
            .iter()
            .enumerate()
            .find(|(_, frame)| frame.len() != pixels)
        {
            return Err(CudaError::InvalidInput(format!(
                "frame {index} has {} pixels, but the CUDA plan expects {pixels} ({} x {})",
                frame.len(),
                self.shape.0,
                self.shape.1
            )));
        }
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        self.run(<P as sealed::Sealed>::frames(frames), false, errors)
    }

    /// Integrates `k` frames stored contiguously (`stack.len() == k × pixels`).
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    ///
    /// Equivalent to [`integrate_frames`](Self::integrate_frames) on the
    /// frames of the stack, with bit-identical results, but each chunk is
    /// copied to the device in one piece. A length that is not a multiple of
    /// the frame size is [`CudaError::InvalidInput`]; an empty stack returns
    /// an empty vector.
    pub fn integrate_stack<P: CudaPixel>(
        &self,
        stack: &[P],
        errors: bool,
    ) -> Result<Vec<Fixed1dResult>, CudaError> {
        let pixels = self.shape.0 * self.shape.1;
        if !stack.len().is_multiple_of(pixels) {
            return Err(CudaError::InvalidInput(format!(
                "stack of {} values is not a whole number of {pixels}-pixel frames",
                stack.len()
            )));
        }
        if stack.is_empty() {
            return Ok(Vec::new());
        }
        self.run(<P as sealed::Sealed>::frames(&[stack]), true, errors)
    }

    /// Decompresses raw HDF5 Bitshuffle+LZ4 chunks on the GPU and integrates
    /// them in VRAM without ever transferring uncompressed frames over PCIe.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn integrate_bslz4(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
    ) -> Result<Vec<Fixed1dResult>, CudaError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        self.run_bslz4(chunks, dtype, errors)
    }

    #[cfg(feature = "cuda")]
    fn run_bslz4(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
    ) -> Result<Vec<Fixed1dResult>, CudaError> {
        self.engine.run_bslz4(chunks, dtype, errors, &self.radial)
    }

    #[cfg(not(feature = "cuda"))]
    fn run_bslz4(
        &self,
        _chunks: &[&[u8]],
        _dtype: Bslz4Dtype,
        _errors: bool,
    ) -> Result<Vec<Fixed1dResult>, CudaError> {
        match self.engine {}
    }

    /// Runs `frames` (see `Engine::run` for `stacked`).
    #[cfg(feature = "cuda")]
    fn run(
        &self,
        frames: sealed::Frames<'_, '_>,
        stacked: bool,
        errors: bool,
    ) -> Result<Vec<Fixed1dResult>, CudaError> {
        self.engine.run(frames, stacked, errors, &self.radial)
    }

    #[cfg(not(feature = "cuda"))]
    fn run(
        &self,
        _frames: sealed::Frames<'_, '_>,
        _stacked: bool,
        _errors: bool,
    ) -> Result<Vec<Fixed1dResult>, CudaError> {
        match self.engine {}
    }

    /// Allocates `bytes` of page-locked (pinned) host memory on this plan's
    /// CUDA context, for zero-copy DMA uploads.
    #[cfg(all(feature = "cuda", feature = "python"))]
    pub(crate) fn alloc_pinned_bytes(&self, bytes: usize) -> Result<CudaPinnedBytes, CudaError> {
        self.engine.alloc_pinned_bytes(bytes)
    }

    #[cfg(all(not(feature = "cuda"), feature = "python"))]
    pub(crate) fn alloc_pinned_bytes(&self, _bytes: usize) -> Result<CudaPinnedBytes, CudaError> {
        match self.engine {}
    }
}

/// Result of a fixed-mask CUDA 2D integration.
///
/// **Experimental:** excluded from semver guarantees until stabilised.
#[derive(Clone, Debug)]
pub struct Fixed2dResult {
    pub radial: Option<Vec<f64>>,
    pub azimuthal: Option<Vec<f64>>,
    pub intensity: Vec<f64>,
    pub errors: Option<Vec<f64>>,
    pub n_radial: usize,
    pub n_azimuthal: usize,
}

/// Options for [`Integrator::prepare_cuda2d`].
///
/// **Experimental:** excluded from semver guarantees until stabilised.
pub type Cuda2dOptions = Cuda1dOptions;

/// A fixed-mask 2D lookup table uploaded to one CUDA device.
///
/// **Experimental:** excluded from semver guarantees until stabilised.
///
/// Created by [`Integrator::prepare_cuda2d`]. Mirrors [`Cuda1dPlan`] for 2D
/// azimuthal integration (`n_radial × n_azimuthal` cakes).
pub struct Cuda2dPlan {
    shape: (usize, usize),
    n_radial: usize,
    n_azimuthal: usize,
    precision: CudaPrecision,
    device: usize,
    device_name: String,
    radial: Option<Vec<f64>>,
    azimuthal: Option<Vec<f64>>,
    table_entries: usize,
    max_batch_frames: usize,
    engine: Engine,
}

impl fmt::Debug for Cuda2dPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cuda2dPlan")
            .field("shape", &self.shape)
            .field("n_radial", &self.n_radial)
            .field("n_azimuthal", &self.n_azimuthal)
            .field("precision", &self.precision)
            .field("device", &self.device)
            .field("device_name", &self.device_name)
            .field("table_entries", &self.table_entries)
            .field("max_batch_frames", &self.max_batch_frames)
            .finish_non_exhaustive()
    }
}

impl Cuda2dPlan {
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
    pub fn n_radial(&self) -> usize {
        self.n_radial
    }
    pub fn n_azimuthal(&self) -> usize {
        self.n_azimuthal
    }
    pub fn precision(&self) -> CudaPrecision {
        self.precision
    }
    pub fn device(&self) -> usize {
        self.device
    }
    pub fn device_name(&self) -> &str {
        &self.device_name
    }
    pub fn radial(&self) -> Option<&[f64]> {
        self.radial.as_deref()
    }
    pub fn azimuthal(&self) -> Option<&[f64]> {
        self.azimuthal.as_deref()
    }
    pub fn table_entries(&self) -> usize {
        self.table_entries
    }
    pub fn max_batch_frames(&self) -> usize {
        self.max_batch_frames
    }

    pub fn integrate_frame<P: CudaPixel>(
        &self,
        frame: &[P],
        errors: bool,
    ) -> Result<Fixed2dResult, CudaError> {
        let mut results = self.integrate_frames(&[frame], errors)?;
        results
            .pop()
            .ok_or_else(|| CudaError::Device("CUDA integration returned no result".into()))
    }

    pub fn integrate_frames<P: CudaPixel>(
        &self,
        frames: &[&[P]],
        errors: bool,
    ) -> Result<Vec<Fixed2dResult>, CudaError> {
        let pixels = self.shape.0 * self.shape.1;
        if let Some((index, frame)) = frames
            .iter()
            .enumerate()
            .find(|(_, frame)| frame.len() != pixels)
        {
            return Err(CudaError::InvalidInput(format!(
                "frame {index} has {} pixels, but the CUDA plan expects {pixels} ({} x {})",
                frame.len(),
                self.shape.0,
                self.shape.1
            )));
        }
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        self.run(<P as sealed::Sealed>::frames(frames), false, errors)
    }

    pub fn integrate_stack<P: CudaPixel>(
        &self,
        stack: &[P],
        errors: bool,
    ) -> Result<Vec<Fixed2dResult>, CudaError> {
        let pixels = self.shape.0 * self.shape.1;
        if !stack.len().is_multiple_of(pixels) {
            return Err(CudaError::InvalidInput(format!(
                "stack of {} values is not a whole number of {pixels}-pixel frames",
                stack.len()
            )));
        }
        if stack.is_empty() {
            return Ok(Vec::new());
        }
        self.run(<P as sealed::Sealed>::frames(&[stack]), true, errors)
    }

    pub fn integrate_bslz4(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
    ) -> Result<Vec<Fixed2dResult>, CudaError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        self.run_bslz4(chunks, dtype, errors)
    }

    #[cfg(feature = "cuda")]
    fn wrap_results(&self, raw: Vec<(Vec<f64>, Option<Vec<f64>>)>) -> Vec<Fixed2dResult> {
        raw.into_iter()
            .map(|(intensity, errors)| Fixed2dResult {
                radial: self.radial.clone(),
                azimuthal: self.azimuthal.clone(),
                intensity,
                errors,
                n_radial: self.n_radial,
                n_azimuthal: self.n_azimuthal,
            })
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn run(
        &self,
        frames: sealed::Frames<'_, '_>,
        stacked: bool,
        errors: bool,
    ) -> Result<Vec<Fixed2dResult>, CudaError> {
        Ok(self.wrap_results(self.engine.run_raw(frames, stacked, errors)?))
    }

    #[cfg(not(feature = "cuda"))]
    fn run(
        &self,
        _frames: sealed::Frames<'_, '_>,
        _stacked: bool,
        _errors: bool,
    ) -> Result<Vec<Fixed2dResult>, CudaError> {
        match self.engine {}
    }

    #[cfg(feature = "cuda")]
    fn run_bslz4(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
    ) -> Result<Vec<Fixed2dResult>, CudaError> {
        Ok(self.wrap_results(self.engine.run_bslz4_raw(chunks, dtype, errors)?))
    }

    #[cfg(not(feature = "cuda"))]
    fn run_bslz4(
        &self,
        _chunks: &[&[u8]],
        _dtype: Bslz4Dtype,
        _errors: bool,
    ) -> Result<Vec<Fixed2dResult>, CudaError> {
        match self.engine {}
    }

    #[cfg(all(feature = "cuda", feature = "python"))]
    pub(crate) fn alloc_pinned_bytes(&self, bytes: usize) -> Result<CudaPinnedBytes, CudaError> {
        self.engine.alloc_pinned_bytes(bytes)
    }

    #[cfg(all(not(feature = "cuda"), feature = "python"))]
    pub(crate) fn alloc_pinned_bytes(&self, _bytes: usize) -> Result<CudaPinnedBytes, CudaError> {
        match self.engine {}
    }
}

/// Owned page-locked host byte buffer allocated on a CUDA context.
#[cfg(all(feature = "cuda", feature = "python"))]
pub(crate) struct CudaPinnedBytes {
    buf: device::PinnedBytes,
}

#[cfg(all(feature = "cuda", feature = "python"))]
impl CudaPinnedBytes {
    pub(crate) fn as_mut_ptr(&mut self) -> Result<*mut u8, CudaError> {
        self.buf.as_mut_ptr()
    }
}

#[cfg(all(not(feature = "cuda"), feature = "python"))]
pub(crate) struct CudaPinnedBytes(std::convert::Infallible);

#[cfg(all(not(feature = "cuda"), feature = "python"))]
impl CudaPinnedBytes {
    pub(crate) fn as_mut_ptr(&mut self) -> Result<*mut u8, CudaError> {
        match self.0 {}
    }
}

/// Checks steps 1 to 4 of [`Integrator::prepare_cuda1d`]; returns the number
/// of pixels per frame.
fn check_configuration(
    integrator: &Integrator,
    shape: (usize, usize),
    n_points: usize,
    max_batch_bytes: Option<usize>,
) -> Result<usize, CudaError> {
    if n_points == 0 || n_points > i32::MAX as usize {
        return Err(CudaError::InvalidInput(format!(
            "n_points must be between 1 and {}, got {n_points}",
            i32::MAX
        )));
    }
    if shape.0 == 0 || shape.1 == 0 {
        return Err(CudaError::InvalidInput(format!(
            "image shape ({}, {}) has a zero side",
            shape.0, shape.1
        )));
    }
    let pixels = shape
        .0
        .checked_mul(shape.1)
        .filter(|&pixels| pixels <= u32::MAX as usize)
        .ok_or_else(|| {
            CudaError::InvalidInput(format!(
                "image shape ({}, {}) exceeds {} pixels",
                shape.0,
                shape.1,
                u32::MAX
            ))
        })?;
    if max_batch_bytes == Some(0) {
        return Err(CudaError::InvalidInput(
            "max_batch_bytes must be positive".into(),
        ));
    }
    if integrator.mask_mode != MaskMode::Fixed {
        return Err(CudaError::Unsupported(
            "CUDA 1D requires mask_mode fixed".into(),
        ));
    }
    if !matches!(
        integrator.method,
        IntegrationMethod::Counting
            | IntegrationMethod::PixelSplit
            | IntegrationMethod::PyfaiFull
            | IntegrationMethod::Supersampled { .. }
    ) {
        return Err(CudaError::Unsupported(format!(
            "CUDA 1D supports the counting, pixel_split, pyfai_full and supersampled methods, not {:?}",
            integrator.method
        )));
    }
    integrator
        .check_frame_shape(shape.0, shape.1)
        .map_err(CudaError::InvalidInput)?;
    Ok(pixels)
}

impl Integrator {
    /// Uploads this integrator's fixed-mask 1D table for `shape` and
    /// `n_points` bins to a CUDA device.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    ///
    /// The plan is a snapshot of the current mask, corrections, method, unit
    /// and radial range; see the [module documentation](crate::integration::cuda) for how
    /// its results relate to the CPU. `fast_math` and `sparse_layout` are
    /// ignored. The checks run in this order, so configuration errors are
    /// reported without touching the driver:
    ///
    /// 1. `n_points` is 0 or above `i32::MAX`, a side of `shape` is 0, the
    ///    frame has more than `u32::MAX` pixels, or `max_batch_bytes` is 0:
    ///    [`CudaError::InvalidInput`].
    /// 2. The mask mode is not [`MaskMode::Fixed`]: [`CudaError::Unsupported`].
    /// 3. The method is not counting, pixel_split, pyfai_full or supersampled:
    ///    [`CudaError::Unsupported`].
    /// 4. `shape` does not match the mask or the attached detector:
    ///    [`CudaError::InvalidInput`].
    /// 5. The crate was built without the `cuda` feature:
    ///    [`CudaError::NotCompiled`].
    /// 6. The device cannot be initialised: [`CudaError::Unavailable`]
    ///    (cached per device after the first success).
    /// 7. The CPU table is built (more than `u32::MAX - 128` entries is
    ///    [`CudaError::Unsupported`]) and uploaded (driver failures, such as
    ///    running out of memory, are [`CudaError::Device`]).
    /// 8. The chunk size is computed; a `max_batch_bytes` too small for one
    ///    frame is [`CudaError::InvalidInput`], too little free device memory
    ///    for one frame is [`CudaError::Device`].
    pub fn prepare_cuda1d(
        &self,
        shape: (usize, usize),
        n_points: usize,
        options: &Cuda1dOptions,
    ) -> Result<Cuda1dPlan, CudaError> {
        let pixels = check_configuration(self, shape, n_points, options.max_batch_bytes)?;
        #[cfg(feature = "cuda")]
        {
            device::prepare(self, shape, pixels, n_points, options)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = pixels;
            Err(CudaError::NotCompiled)
        }
    }

    /// Uploads this integrator's fixed-mask 2D table for `shape`, `n_radial`
    /// and `n_azimuthal` (with optional `azimuthal_range`) to a CUDA device.
    ///
    /// **Experimental:** excluded from semver guarantees until stabilised.
    pub fn prepare_cuda2d(
        &self,
        shape: (usize, usize),
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_range: Option<(f64, f64)>,
        options: &Cuda2dOptions,
    ) -> Result<Cuda2dPlan, CudaError> {
        let total_bins = n_radial
            .checked_mul(n_azimuthal)
            .filter(|&b| b > 0 && b <= i32::MAX as usize)
            .ok_or_else(|| {
                CudaError::InvalidInput(format!(
                    "2D bin count ({n_radial} x {n_azimuthal}) must be in 1..={}",
                    i32::MAX
                ))
            })?;
        let pixels = check_configuration(self, shape, total_bins, options.max_batch_bytes)?;
        if let Some((min, _)) = self.radial_range()
            && (!min.is_finite() || min < 0.0)
        {
            return Err(CudaError::InvalidInput(
                "2D integration requires radial_range min >= 0".into(),
            ));
        }
        if let Some(range) = azimuthal_range {
            super::grid::IntegrationGrid2d::new(n_radial, n_azimuthal, (0.0, 1.0), range)
                .map_err(CudaError::InvalidInput)?;
        }
        #[cfg(feature = "cuda")]
        {
            device::prepare_2d(
                self,
                shape,
                pixels,
                n_radial,
                n_azimuthal,
                azimuthal_range,
                options,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (pixels, azimuthal_range);
            Err(CudaError::NotCompiled)
        }
    }
}
