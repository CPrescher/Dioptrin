//! Experimental Metal 1D and 2D integration of fixed-mask lookup tables on macOS.
//!
//! **Experimental:** everything in this module is excluded from semver
//! guarantees until it is stabilised.
//!
//! The public items are always compiled, so the same code builds with and
//! without the `metal` cargo feature (and on non-macOS platforms). Without the
//! feature, [`Integrator::prepare_metal1d`] and [`Integrator::prepare_metal2d`]
//! still check the configuration and then return [`MetalError::NotCompiled`],
//! and [`metal_info`] reports `compiled: false`.
//!
//! With the feature on macOS, the crate links against `Metal.framework` and
//! compiles the Metal Shading Language (MSL) kernels at run time on first use
//! (with `MTLMathMode::Safe` to preserve error-free compensated summation).
//!
//! A [`Metal1dPlan`] or [`Metal2dPlan`] is a snapshot of an [`Integrator`]'s
//! fixed-mask CSR lookup table, corrections (solid angle, polarisation), radial
//! unit and radial/azimuthal range. It integrates `f64`, `f32`, `u16`, `u32` or
//! `i32` frames (as well as Bitshuffle+LZ4 compressed chunks decoded on the GPU)
//! and returns the same [`Fixed1dResult`] / [`Fixed2dResult`] as the CPU and
//! CUDA plans:
//!
//! - Bin denominators are summed on the CPU in strict `f64` when the plan is
//!   prepared; the final division and Poisson error propagation are computed in
//!   `f64`, returning `0.0` for bins with zero denominator.
//! - [`MetalPrecision::F32Compensated`] (the default) uses double-word `(hi, lo)`
//!   compensated `f32` accumulation (Knuth TwoSum + FMA exact products).
//! - Apple Silicon GPUs do not support hardware `f64` in MSL; arbitrary `f64`
//!   inputs are rounded to `f32` during staging.
//! - Page-aligned native inputs (`f32`, `u16`, `u32`, `i32`) and buffers
//!   allocated through [`Metal1dPlan::alloc_shared_bytes`] /
//!   [`Metal2dPlan::alloc_shared_bytes`] are bound zero-copy on Unified Memory.
#![warn(dead_code)]

use std::{fmt, str::FromStr};

use super::{Bslz4Dtype, Fixed1dResult, Fixed2dResult, IntegrationMethod, Integrator, MaskMode};

#[cfg(all(feature = "metal", target_os = "macos", test))]
mod bench;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod device;
#[cfg(test)]
mod tests;

/// Storage and accumulation precision of a [`Metal1dPlan`] or [`Metal2dPlan`].
///
/// **Experimental:** excluded from semver guarantees until stabilised.
///
/// In every mode the bin denominators are computed on the CPU in `f64`, and
/// the division, square root and output are `f64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MetalPrecision {
    /// `f32` pixels and weights, `f32` fused multiply-add accumulation.
    F32,
    /// `f32` pixels and weights, double-word `(hi, lo)` compensated `f32`
    /// accumulation. The default.
    #[default]
    F32Compensated,
}

impl MetalPrecision {
    /// Name of the precision: `"f32"` or `"f32_compensated"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F32Compensated => "f32_compensated",
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn acc_mode(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F32Compensated => 1,
        }
    }
}

impl fmt::Display for MetalPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MetalPrecision {
    type Err = MetalError;

    fn from_str(name: &str) -> Result<Self, MetalError> {
        match name {
            "f32" => Ok(Self::F32),
            "f32_compensated" => Ok(Self::F32Compensated),
            _ => Err(MetalError::InvalidInput(format!(
                "unknown Metal precision {name:?}; expected \"f32\" or \"f32_compensated\""
            ))),
        }
    }
}

/// Options for [`Integrator::prepare_metal1d`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Metal1dOptions {
    precision: MetalPrecision,
    device: usize,
    max_batch_bytes: Option<usize>,
}

impl Metal1dOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_precision(mut self, precision: MetalPrecision) -> Self {
        self.precision = precision;
        self
    }

    pub fn with_device(mut self, ordinal: usize) -> Self {
        self.device = ordinal;
        self
    }

    pub fn with_max_batch_bytes(mut self, bytes: usize) -> Self {
        self.max_batch_bytes = Some(bytes);
        self
    }

    pub fn precision(&self) -> MetalPrecision {
        self.precision
    }

    pub fn device(&self) -> usize {
        self.device
    }

    pub fn max_batch_bytes(&self) -> Option<usize> {
        self.max_batch_bytes
    }
}

/// Options for [`Integrator::prepare_metal2d`].
pub type Metal2dOptions = Metal1dOptions;

/// Error of the Metal integration API.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetalError {
    /// The crate was built without the `metal` cargo feature.
    NotCompiled,
    /// Metal cannot be used (non-macOS platform, no Metal GPU, invalid ordinal,
    /// or shader compilation failure).
    Unavailable(String),
    /// The integrator configuration is not supported on Metal.
    Unsupported(String),
    /// Invalid shapes, sizes, frame lengths or memory budgets.
    InvalidInput(String),
    /// A Metal runtime error.
    Device(String),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCompiled => f.write_str(
                "dioptrin was built without Metal support; enable the `metal` cargo feature on macOS",
            ),
            Self::Unavailable(reason) => write!(f, "Metal unavailable: {reason}"),
            Self::Unsupported(reason) => write!(f, "unsupported Metal configuration: {reason}"),
            Self::InvalidInput(reason) => write!(f, "invalid Metal input: {reason}"),
            Self::Device(reason) => write!(f, "Metal device error: {reason}"),
        }
    }
}

impl std::error::Error for MetalError {}

/// Description of the Metal runtime and one device, from [`metal_info`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalInfo {
    pub compiled: bool,
    pub available: bool,
    pub reason: Option<String>,
    pub device: usize,
    pub device_name: Option<String>,
    pub has_unified_memory: Option<bool>,
    pub recommended_max_working_set_size: Option<usize>,
    pub max_threadgroup_memory_length: Option<usize>,
}

impl MetalInfo {
    fn unavailable(device: usize, compiled: bool, reason: String) -> Self {
        Self {
            compiled,
            available: false,
            reason: Some(reason),
            device,
            device_name: None,
            has_unified_memory: None,
            recommended_max_working_set_size: None,
            max_threadgroup_memory_length: None,
        }
    }
}

/// Describes Metal support and device `device`. Never panics.
pub fn metal_info(device: usize) -> MetalInfo {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        match device::device(device) {
            Ok(dev) => dev.info.clone(),
            Err(info) => info,
        }
    }
    #[cfg(all(feature = "metal", not(target_os = "macos")))]
    {
        MetalInfo::unavailable(device, true, "Metal is only available on macOS".into())
    }
    #[cfg(not(feature = "metal"))]
    {
        MetalInfo::unavailable(device, false, MetalError::NotCompiled.to_string())
    }
}

mod sealed {
    use super::MetalError;

    #[derive(Clone, Copy)]
    pub enum Frames<'a, 'b> {
        U16(&'a [&'b [u16]]),
        U32(&'a [&'b [u32]]),
        I32(&'a [&'b [i32]]),
        F32(&'a [&'b [f32]]),
        F64(&'a [&'b [f64]]),
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    impl Frames<'_, '_> {
        pub(crate) fn count(self, pixels: usize, stacked: bool) -> Result<usize, MetalError> {
            fn check<T>(
                slices: &[&[T]],
                pixels: usize,
                stacked: bool,
            ) -> Result<usize, MetalError> {
                if stacked {
                    let len = slices[0].len();
                    if !len.is_multiple_of(pixels) {
                        return Err(MetalError::InvalidInput(format!(
                            "stacked frame length {len} is not a multiple of plan pixel count {pixels}"
                        )));
                    }
                    Ok(len / pixels)
                } else {
                    for (idx, frame) in slices.iter().enumerate() {
                        if frame.len() != pixels {
                            return Err(MetalError::InvalidInput(format!(
                                "frame {idx} has {} pixels, expected {pixels}",
                                frame.len()
                            )));
                        }
                    }
                    Ok(slices.len())
                }
            }
            match self {
                Self::U16(s) => check(s, pixels, stacked),
                Self::U32(s) => check(s, pixels, stacked),
                Self::I32(s) => check(s, pixels, stacked),
                Self::F32(s) => check(s, pixels, stacked),
                Self::F64(s) => check(s, pixels, stacked),
            }
        }

        pub(crate) fn contiguous_bytes(
            self,
            offset_frames: usize,
            count_frames: usize,
            pixels: usize,
        ) -> Option<(*const u8, usize)> {
            fn slice_bytes<T>(
                slice: &[T],
                offset_frames: usize,
                count_frames: usize,
                pixels: usize,
            ) -> (*const u8, usize) {
                let start = offset_frames * pixels;
                let len = count_frames * pixels;
                (
                    slice[start..start + len].as_ptr().cast::<u8>(),
                    len * size_of::<T>(),
                )
            }
            match self {
                Self::U16(s) => Some(slice_bytes(s[0], offset_frames, count_frames, pixels)),
                Self::U32(s) => Some(slice_bytes(s[0], offset_frames, count_frames, pixels)),
                Self::I32(s) => Some(slice_bytes(s[0], offset_frames, count_frames, pixels)),
                Self::F32(s) => Some(slice_bytes(s[0], offset_frames, count_frames, pixels)),
                Self::F64(_) => None,
            }
        }
    }

    pub trait Sealed: Sized {
        fn frames<'a, 'b>(frames: &'a [&'b [Self]]) -> Frames<'a, 'b>;
    }

    impl Sealed for u16 {
        fn frames<'a, 'b>(frames: &'a [&'b [Self]]) -> Frames<'a, 'b> {
            Frames::U16(frames)
        }
    }
    impl Sealed for u32 {
        fn frames<'a, 'b>(frames: &'a [&'b [Self]]) -> Frames<'a, 'b> {
            Frames::U32(frames)
        }
    }
    impl Sealed for i32 {
        fn frames<'a, 'b>(frames: &'a [&'b [Self]]) -> Frames<'a, 'b> {
            Frames::I32(frames)
        }
    }
    impl Sealed for f32 {
        fn frames<'a, 'b>(frames: &'a [&'b [Self]]) -> Frames<'a, 'b> {
            Frames::F32(frames)
        }
    }
    impl Sealed for f64 {
        fn frames<'a, 'b>(frames: &'a [&'b [Self]]) -> Frames<'a, 'b> {
            Frames::F64(frames)
        }
    }

    #[allow(dead_code)]
    fn _keep_error_import(_: MetalError) {}
}

/// Pixel types a [`Metal1dPlan`] or [`Metal2dPlan`] accepts: `u16`, `u32`, `i32`, `f32`, `f64`.
pub trait MetalPixel: sealed::Sealed + Copy + Send + Sync + 'static {}
impl MetalPixel for u16 {}
impl MetalPixel for u32 {}
impl MetalPixel for i32 {}
impl MetalPixel for f32 {}
impl MetalPixel for f64 {}

/// Owned Metal shared memory buffer for zero-copy input staging.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub struct MetalSharedBytes {
    buf: device::SharedBytes,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl MetalSharedBytes {
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buf.as_mut_ptr()
    }

    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        self.buf.as_slice_mut()
    }

    pub fn as_slice(&self) -> &[u8] {
        self.buf.as_slice()
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub struct MetalSharedBytes(std::convert::Infallible);

#[cfg(not(all(feature = "metal", target_os = "macos")))]
impl MetalSharedBytes {
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self.0 {}
    }

    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        match self.0 {}
    }

    pub fn as_slice(&self) -> &[u8] {
        match self.0 {}
    }
}

/// Prepared fixed-mask 1D integration plan on a Metal device.
pub struct Metal1dPlan {
    shape: (usize, usize),
    n_points: usize,
    precision: MetalPrecision,
    device: usize,
    device_name: String,
    radial: Vec<f64>,
    max_batch_frames: usize,
    table_entries: usize,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    engine: device::Engine,
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    engine: std::convert::Infallible,
}

impl fmt::Debug for Metal1dPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Metal1dPlan")
            .field("shape", &self.shape)
            .field("n_points", &self.n_points)
            .field("precision", &self.precision)
            .field("device", &self.device)
            .field("device_name", &self.device_name)
            .field("max_batch_frames", &self.max_batch_frames)
            .field("table_entries", &self.table_entries)
            .finish()
    }
}

impl Metal1dPlan {
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn n_points(&self) -> usize {
        self.n_points
    }

    pub fn precision(&self) -> MetalPrecision {
        self.precision
    }

    pub fn device(&self) -> usize {
        self.device
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn radial(&self) -> &[f64] {
        &self.radial
    }

    pub fn max_batch_frames(&self) -> usize {
        self.max_batch_frames
    }

    pub fn table_entries(&self) -> usize {
        self.table_entries
    }

    pub fn integrate_frame<P: MetalPixel>(
        &self,
        frame: &[P],
        errors: bool,
    ) -> Result<Fixed1dResult, MetalError> {
        let mut results = self.run(<P as sealed::Sealed>::frames(&[frame]), false, errors)?;
        Ok(results.pop().expect("one input frame gives one result"))
    }

    pub fn integrate_frames<P: MetalPixel>(
        &self,
        frames: &[&[P]],
        errors: bool,
    ) -> Result<Vec<Fixed1dResult>, MetalError> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        self.run(<P as sealed::Sealed>::frames(frames), false, errors)
    }

    pub fn integrate_stack<P: MetalPixel>(
        &self,
        stack: &[P],
        errors: bool,
    ) -> Result<Vec<Fixed1dResult>, MetalError> {
        if stack.is_empty() {
            return Ok(Vec::new());
        }
        self.run(<P as sealed::Sealed>::frames(&[stack]), true, errors)
    }

    pub fn integrate_stack_into<P: MetalPixel>(
        &self,
        stack: &[P],
        out_intensity: &mut [f64],
        out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        let pixels = self.shape.0 * self.shape.1;
        if !stack.len().is_multiple_of(pixels) {
            return Err(MetalError::InvalidInput(format!(
                "stack length {} is not a multiple of frame pixels {pixels}",
                stack.len()
            )));
        }
        let n_frames = stack.len() / pixels;
        let expected = n_frames * self.n_points;
        if out_intensity.len() != expected
            || out_errors.as_ref().is_some_and(|e| e.len() != expected)
        {
            return Err(MetalError::InvalidInput(
                "output buffer length does not match frames * n_points".into(),
            ));
        }
        if stack.is_empty() {
            return Ok(());
        }
        self.run_into(
            <P as sealed::Sealed>::frames(&[stack]),
            true,
            out_intensity,
            out_errors,
        )
    }

    pub fn integrate_bslz4(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
    ) -> Result<Vec<Fixed1dResult>, MetalError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        self.run_bslz4(chunks, dtype, errors)
    }

    pub fn integrate_bslz4_into(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        out_intensity: &mut [f64],
        out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        let expected = chunks.len() * self.n_points;
        if out_intensity.len() != expected
            || out_errors.as_ref().is_some_and(|e| e.len() != expected)
        {
            return Err(MetalError::InvalidInput(
                "output buffer length does not match chunks * n_points".into(),
            ));
        }
        if chunks.is_empty() {
            return Ok(());
        }
        self.run_bslz4_into(chunks, dtype, out_intensity, out_errors)
    }

    pub fn alloc_shared_bytes(&self, bytes: usize) -> Result<MetalSharedBytes, MetalError> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            Ok(MetalSharedBytes {
                buf: self.engine.alloc_shared_bytes(bytes)?,
            })
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            let _ = bytes;
            match self.engine {}
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn run(
        &self,
        frames: sealed::Frames<'_, '_>,
        stacked: bool,
        errors: bool,
    ) -> Result<Vec<Fixed1dResult>, MetalError> {
        self.engine.run_1d(frames, stacked, errors, &self.radial)
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn run(
        &self,
        _frames: sealed::Frames<'_, '_>,
        _stacked: bool,
        _errors: bool,
    ) -> Result<Vec<Fixed1dResult>, MetalError> {
        match self.engine {}
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn run_into(
        &self,
        frames: sealed::Frames<'_, '_>,
        stacked: bool,
        out_intensity: &mut [f64],
        out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        self.engine
            .run_into(frames, stacked, out_intensity, out_errors)
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn run_into(
        &self,
        _frames: sealed::Frames<'_, '_>,
        _stacked: bool,
        _out_intensity: &mut [f64],
        _out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        match self.engine {}
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn run_bslz4(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
    ) -> Result<Vec<Fixed1dResult>, MetalError> {
        self.engine
            .run_bslz4_1d(chunks, dtype, errors, &self.radial)
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn run_bslz4(
        &self,
        _chunks: &[&[u8]],
        _dtype: Bslz4Dtype,
        _errors: bool,
    ) -> Result<Vec<Fixed1dResult>, MetalError> {
        match self.engine {}
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn run_bslz4_into(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        out_intensity: &mut [f64],
        out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        self.engine
            .run_bslz4_into(chunks, dtype, out_intensity, out_errors)
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn run_bslz4_into(
        &self,
        _chunks: &[&[u8]],
        _dtype: Bslz4Dtype,
        _out_intensity: &mut [f64],
        _out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        match self.engine {}
    }
}

/// Prepared fixed-mask 2D integration plan on a Metal device.
pub struct Metal2dPlan {
    shape: (usize, usize),
    n_radial: usize,
    n_azimuthal: usize,
    precision: MetalPrecision,
    device: usize,
    device_name: String,
    radial: Option<Vec<f64>>,
    azimuthal: Option<Vec<f64>>,
    max_batch_frames: usize,
    table_entries: usize,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    engine: device::Engine,
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    engine: std::convert::Infallible,
}

impl fmt::Debug for Metal2dPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Metal2dPlan")
            .field("shape", &self.shape)
            .field("n_radial", &self.n_radial)
            .field("n_azimuthal", &self.n_azimuthal)
            .field("precision", &self.precision)
            .field("device", &self.device)
            .field("device_name", &self.device_name)
            .field("max_batch_frames", &self.max_batch_frames)
            .field("table_entries", &self.table_entries)
            .finish()
    }
}

impl Metal2dPlan {
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn n_radial(&self) -> usize {
        self.n_radial
    }

    pub fn n_azimuthal(&self) -> usize {
        self.n_azimuthal
    }

    pub fn precision(&self) -> MetalPrecision {
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

    pub fn max_batch_frames(&self) -> usize {
        self.max_batch_frames
    }

    pub fn table_entries(&self) -> usize {
        self.table_entries
    }

    pub fn integrate_frame<P: MetalPixel>(
        &self,
        frame: &[P],
        errors: bool,
    ) -> Result<Fixed2dResult, MetalError> {
        let mut results = self.run(<P as sealed::Sealed>::frames(&[frame]), false, errors)?;
        Ok(results.pop().expect("one input frame gives one result"))
    }

    pub fn integrate_frames<P: MetalPixel>(
        &self,
        frames: &[&[P]],
        errors: bool,
    ) -> Result<Vec<Fixed2dResult>, MetalError> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        self.run(<P as sealed::Sealed>::frames(frames), false, errors)
    }

    pub fn integrate_stack<P: MetalPixel>(
        &self,
        stack: &[P],
        errors: bool,
    ) -> Result<Vec<Fixed2dResult>, MetalError> {
        if stack.is_empty() {
            return Ok(Vec::new());
        }
        self.run(<P as sealed::Sealed>::frames(&[stack]), true, errors)
    }

    pub fn integrate_stack_into<P: MetalPixel>(
        &self,
        stack: &[P],
        out_intensity: &mut [f64],
        out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        let pixels = self.shape.0 * self.shape.1;
        if !stack.len().is_multiple_of(pixels) {
            return Err(MetalError::InvalidInput(format!(
                "stack length {} is not a multiple of frame pixels {pixels}",
                stack.len()
            )));
        }
        let n_frames = stack.len() / pixels;
        let expected = n_frames * self.n_radial * self.n_azimuthal;
        if out_intensity.len() != expected
            || out_errors.as_ref().is_some_and(|e| e.len() != expected)
        {
            return Err(MetalError::InvalidInput(
                "output buffer length does not match frames * n_radial * n_azimuthal".into(),
            ));
        }
        if stack.is_empty() {
            return Ok(());
        }
        self.run_into(
            <P as sealed::Sealed>::frames(&[stack]),
            true,
            out_intensity,
            out_errors,
        )
    }

    pub fn integrate_bslz4(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
    ) -> Result<Vec<Fixed2dResult>, MetalError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        self.run_bslz4(chunks, dtype, errors)
    }

    pub fn integrate_bslz4_into(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        out_intensity: &mut [f64],
        out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        let expected = chunks.len() * self.n_radial * self.n_azimuthal;
        if out_intensity.len() != expected
            || out_errors.as_ref().is_some_and(|e| e.len() != expected)
        {
            return Err(MetalError::InvalidInput(
                "output buffer length does not match chunks * n_radial * n_azimuthal".into(),
            ));
        }
        if chunks.is_empty() {
            return Ok(());
        }
        self.run_bslz4_into(chunks, dtype, out_intensity, out_errors)
    }

    pub fn alloc_shared_bytes(&self, bytes: usize) -> Result<MetalSharedBytes, MetalError> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            Ok(MetalSharedBytes {
                buf: self.engine.alloc_shared_bytes(bytes)?,
            })
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            let _ = bytes;
            match self.engine {}
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
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

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn run(
        &self,
        frames: sealed::Frames<'_, '_>,
        stacked: bool,
        errors: bool,
    ) -> Result<Vec<Fixed2dResult>, MetalError> {
        Ok(self.wrap_results(self.engine.run_raw(frames, stacked, errors)?))
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn run(
        &self,
        _frames: sealed::Frames<'_, '_>,
        _stacked: bool,
        _errors: bool,
    ) -> Result<Vec<Fixed2dResult>, MetalError> {
        match self.engine {}
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn run_into(
        &self,
        frames: sealed::Frames<'_, '_>,
        stacked: bool,
        out_intensity: &mut [f64],
        out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        self.engine
            .run_into(frames, stacked, out_intensity, out_errors)
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn run_into(
        &self,
        _frames: sealed::Frames<'_, '_>,
        _stacked: bool,
        _out_intensity: &mut [f64],
        _out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        match self.engine {}
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn run_bslz4(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
    ) -> Result<Vec<Fixed2dResult>, MetalError> {
        Ok(self.wrap_results(self.engine.run_bslz4_raw(chunks, dtype, errors)?))
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn run_bslz4(
        &self,
        _chunks: &[&[u8]],
        _dtype: Bslz4Dtype,
        _errors: bool,
    ) -> Result<Vec<Fixed2dResult>, MetalError> {
        match self.engine {}
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn run_bslz4_into(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        out_intensity: &mut [f64],
        out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        self.engine
            .run_bslz4_into(chunks, dtype, out_intensity, out_errors)
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn run_bslz4_into(
        &self,
        _chunks: &[&[u8]],
        _dtype: Bslz4Dtype,
        _out_intensity: &mut [f64],
        _out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        match self.engine {}
    }
}

fn check_configuration(
    integrator: &Integrator,
    shape: (usize, usize),
    n_points: usize,
    max_batch_bytes: Option<usize>,
) -> Result<usize, MetalError> {
    if n_points == 0 || n_points > i32::MAX as usize {
        return Err(MetalError::InvalidInput(format!(
            "n_points must be between 1 and {}, got {n_points}",
            i32::MAX
        )));
    }
    if shape.0 == 0 || shape.1 == 0 {
        return Err(MetalError::InvalidInput(format!(
            "image shape ({}, {}) has a zero side",
            shape.0, shape.1
        )));
    }
    let pixels = shape
        .0
        .checked_mul(shape.1)
        .filter(|&pixels| pixels <= u32::MAX as usize)
        .ok_or_else(|| {
            MetalError::InvalidInput(format!(
                "image shape ({}, {}) exceeds {} pixels",
                shape.0,
                shape.1,
                u32::MAX
            ))
        })?;
    if max_batch_bytes == Some(0) {
        return Err(MetalError::InvalidInput(
            "max_batch_bytes must be positive".into(),
        ));
    }
    if integrator.mask_mode != MaskMode::Fixed {
        return Err(MetalError::Unsupported(
            "Metal requires mask_mode fixed".into(),
        ));
    }
    if !matches!(
        integrator.method,
        IntegrationMethod::Counting
            | IntegrationMethod::PixelSplit
            | IntegrationMethod::PyfaiFull
            | IntegrationMethod::Supersampled { .. }
    ) {
        return Err(MetalError::Unsupported(format!(
            "Metal supports the counting, pixel_split, pyfai_full and supersampled methods, not {:?}",
            integrator.method
        )));
    }
    integrator
        .check_frame_shape(shape.0, shape.1)
        .map_err(MetalError::InvalidInput)?;
    Ok(pixels)
}

impl Integrator {
    /// Uploads this integrator's fixed-mask 1D table for `shape` and
    /// `n_points` bins to a Metal GPU device.
    pub fn prepare_metal1d(
        &self,
        shape: (usize, usize),
        n_points: usize,
        options: &Metal1dOptions,
    ) -> Result<Metal1dPlan, MetalError> {
        let pixels = check_configuration(self, shape, n_points, options.max_batch_bytes)?;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            device::prepare(self, shape, pixels, n_points, options)
        }
        #[cfg(all(feature = "metal", not(target_os = "macos")))]
        {
            let _ = pixels;
            Err(MetalError::Unavailable(
                "Metal is only available on macOS".into(),
            ))
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = pixels;
            Err(MetalError::NotCompiled)
        }
    }

    /// Uploads this integrator's fixed-mask 2D table for `shape`, `n_radial`
    /// and `n_azimuthal` (with optional `azimuthal_range`) to a Metal GPU device.
    pub fn prepare_metal2d(
        &self,
        shape: (usize, usize),
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal_range: Option<(f64, f64)>,
        options: &Metal2dOptions,
    ) -> Result<Metal2dPlan, MetalError> {
        let total_bins = n_radial
            .checked_mul(n_azimuthal)
            .filter(|&b| b > 0 && b <= i32::MAX as usize)
            .ok_or_else(|| {
                MetalError::InvalidInput(format!(
                    "2D bin count ({n_radial} x {n_azimuthal}) must be in 1..={}",
                    i32::MAX
                ))
            })?;
        let pixels = check_configuration(self, shape, total_bins, options.max_batch_bytes)?;
        if let Some((min, _)) = self.radial_range()
            && (!min.is_finite() || min < 0.0)
        {
            return Err(MetalError::InvalidInput(
                "2D integration requires radial_range min >= 0".into(),
            ));
        }
        if let Some(range) = azimuthal_range {
            super::grid::IntegrationGrid2d::new(n_radial, n_azimuthal, (0.0, 1.0), range)
                .map_err(MetalError::InvalidInput)?;
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
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
        #[cfg(all(feature = "metal", not(target_os = "macos")))]
        {
            let _ = (pixels, azimuthal_range);
            Err(MetalError::Unavailable(
                "Metal is only available on macOS".into(),
            ))
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (pixels, azimuthal_range);
            Err(MetalError::NotCompiled)
        }
    }
}
