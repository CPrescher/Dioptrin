//! Device side of the Metal 1D/2D integration (feature `metal` on macOS).
#![allow(clippy::too_many_arguments)]

use std::{
    borrow::Cow,
    collections::HashMap,
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Mutex, OnceLock, PoisonError},
    time::Instant,
};

use objc2::rc::Retained;
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::sel;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
    MTLMathMode, MTLResourceOptions, MTLSize,
};
use rayon::prelude::*;

use super::{
    Metal1dOptions, Metal1dPlan, Metal2dOptions, Metal2dPlan, MetalError, MetalInfo,
    MetalPrecision, sealed::Frames,
};
use crate::integration::{Bslz4Dtype, Fixed1dResult, InputOrder, Integrator};
use crate::lut::prepared::Columns;

/// Default chunk budget cap (1 GiB).
const DEFAULT_BATCH_BYTES: usize = 1 << 30;
/// Frames per kernel dispatch.
const FRAMES_PER_LAUNCH: u32 = 256;
/// Minimum element count before staging into shared memory uses Rayon.
const PARALLEL_STAGE_THRESHOLD: usize = 65_536;
/// Target tile staging size per pipeline slot (192 MiB: holds 4 Eiger 9M f32 frames).
const SLOT_BYTES: usize = 192 << 20;
/// Maximum average entries per bin that prefer `Layout::SubWarp` (8 threads/bin, ideal for 2D).
const SUBWARP_MAX_ENTRIES_PER_BIN: usize = 256;
/// Maximum average entries per bin that still prefer `Layout::Warp` over `Layout::Block`.
const WARP_MAX_ENTRIES_PER_BIN: usize = 5000;
/// macOS VM page size (16 KiB on Apple Silicon ARM64, 4 KiB on x86_64).
const PAGE_SIZE: usize = if cfg!(target_arch = "aarch64") {
    16_384
} else {
    4_096
};

/// Verifies via Mach `mach_vm_region` that every page in `[start, start + len)`
/// is mapped and writable (`VM_PROT_READ | VM_PROT_WRITE`).
///
/// Metal's `newBufferWithBytesNoCopy` maps host pages RW into the GPU and raises
/// `SIGBUS` on read-only pages (`.rodata`, `mmap_mode='r'`, `bytes`). Checking
/// region protections here ensures read-only inputs fall back cleanly to staging.
fn is_rw_mapped(start: usize, len: usize) -> bool {
    const VM_REGION_BASIC_INFO_64: i32 = 9;
    const VM_REGION_BASIC_INFO_COUNT_64: u32 = 9;
    const VM_PROT_RW: i32 = 1 | 2; // VM_PROT_READ | VM_PROT_WRITE

    unsafe extern "C" {
        static mach_task_self_: u32;
        fn mach_vm_region(
            target_task: u32,
            address: *mut u64,
            size: *mut u64,
            flavor: i32,
            info: *mut i32,
            info_cnt: *mut u32,
            object_name: *mut u32,
        ) -> i32;
    }

    let Some(end) = start.checked_add(len) else {
        return false;
    };
    let mut cur = start as u64;
    let end_u64 = end as u64;
    while cur < end_u64 {
        let mut region_addr = cur;
        let mut region_size: u64 = 0;
        let mut info = [0i32; 16];
        let mut count = VM_REGION_BASIC_INFO_COUNT_64;
        let mut object_name: u32 = 0;
        // SAFETY: all out-pointers are valid stack variables; `mach_task_self_` is
        // the current task's Mach port from libSystem.
        let kr = unsafe {
            mach_vm_region(
                mach_task_self_,
                &mut region_addr,
                &mut region_size,
                VM_REGION_BASIC_INFO_64,
                info.as_mut_ptr(),
                &mut count,
                &mut object_name,
            )
        };
        if kr != 0 || region_size == 0 || region_addr > cur {
            return false;
        }
        let protection = info[0];
        if (protection & VM_PROT_RW) != VM_PROT_RW {
            return false;
        }
        cur = match region_addr.checked_add(region_size) {
            Some(next) if next > cur => next,
            _ => return false,
        };
    }
    true
}

/// Marker trait restricting `MetalHandle`'s `Send + Sync` impl to Metal protocols
/// that Apple documents as thread-safe (excluding command buffers and encoders).
trait ThreadSafeMetalProtocol {}
impl ThreadSafeMetalProtocol for dyn MTLDevice {}
impl ThreadSafeMetalProtocol for dyn MTLCommandQueue {}
impl ThreadSafeMetalProtocol for dyn MTLLibrary {}
impl ThreadSafeMetalProtocol for dyn MTLComputePipelineState {}
impl ThreadSafeMetalProtocol for dyn MTLBuffer {}

/// Wrapper asserting `Send + Sync` for thread-safe Metal protocol objects.
struct MetalHandle<T: ?Sized>(Retained<ProtocolObject<T>>);

// SAFETY: `ThreadSafeMetalProtocol` is only implemented for Metal devices,
// command queues, libraries, compute pipelines, and buffers, which are
// thread-safe per Apple's Metal specification. Mutable access to scratch
// buffers is synchronised by the plan's mutex.
unsafe impl<T: ?Sized + ThreadSafeMetalProtocol> Send for MetalHandle<T> {}
unsafe impl<T: ?Sized + ThreadSafeMetalProtocol> Sync for MetalHandle<T> {}

impl<T: ?Sized> std::ops::Deref for MetalHandle<T> {
    type Target = ProtocolObject<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum PixelKind {
    F32,
    U16,
    U32,
    I32,
}

impl PixelKind {
    fn kernel_name(self) -> &'static str {
        match self {
            Self::F32 => "integrate_f32",
            Self::U16 => "integrate_u16",
            Self::U32 => "integrate_u32",
            Self::I32 => "integrate_i32",
        }
    }

    fn elem_bytes(self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
            Self::U16 => size_of::<u16>(),
            Self::U32 => size_of::<u32>(),
            Self::I32 => size_of::<i32>(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Layout {
    Block,
    Warp,
    SubWarp,
}

impl Layout {
    pub(super) fn bins_per_tg(self) -> u32 {
        match self {
            Self::Block => 1,
            Self::Warp => 4,
            Self::SubWarp => 16,
        }
    }

    #[cfg(test)]
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::Warp => "warp",
            Self::SubWarp => "subwarp",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PipelineKey {
    pixel: PixelKind,
    acc_mode: u32,
    bins_per_tg: u32,
    frames_per_tg: u32,
    weighted: bool,
    with_variance: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Bslz4KernelKind {
    U16,
    U32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct KernelParams {
    pixels: u32,
    bins: u32,
    num_frames: u32,
}

/// Process-wide state of one initialised Metal device.
pub(super) struct Device {
    device: MetalHandle<dyn MTLDevice>,
    queue: MetalHandle<dyn MTLCommandQueue>,
    library: MetalHandle<dyn MTLLibrary>,
    pipelines: Mutex<HashMap<PipelineKey, Arc<MetalHandle<dyn MTLComputePipelineState>>>>,
    bslz4_pipelines: Mutex<HashMap<Bslz4KernelKind, Arc<MetalHandle<dyn MTLComputePipelineState>>>>,
    /// Registry of buffers allocated via `alloc_shared_bytes` so zero-copy
    /// binding works even when the caller's logical byte length is not a
    /// multiple of `PAGE_SIZE` (the underlying Metal buffer is page-padded).
    shared_allocs: Mutex<HashMap<usize, Arc<MetalHandle<dyn MTLBuffer>>>>,
    pub(super) info: MetalInfo,
    #[allow(dead_code)]
    pub(super) compile_ms: f64,
}

impl Device {
    fn pipeline(
        &self,
        key: PipelineKey,
    ) -> Result<Arc<MetalHandle<dyn MTLComputePipelineState>>, MetalError> {
        let mut cache = self
            .pipelines
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(p) = cache.get(&key) {
            return Ok(Arc::clone(p));
        }

        let constants = MTLFunctionConstantValues::new();
        let with_var = key.with_variance;
        let weighted = key.weighted;
        let acc_mode = key.acc_mode;
        let bins_per_tg = key.bins_per_tg;
        let frames_per_tg = key.frames_per_tg;
        // SAFETY: pointers are valid for the duration of the calls and match the declared MTLDataType.
        unsafe {
            constants.setConstantValue_type_atIndex(
                NonNull::from(&with_var).cast::<c_void>(),
                MTLDataType::Bool,
                0,
            );
            constants.setConstantValue_type_atIndex(
                NonNull::from(&weighted).cast::<c_void>(),
                MTLDataType::Bool,
                1,
            );
            constants.setConstantValue_type_atIndex(
                NonNull::from(&acc_mode).cast::<c_void>(),
                MTLDataType::UInt,
                2,
            );
            constants.setConstantValue_type_atIndex(
                NonNull::from(&bins_per_tg).cast::<c_void>(),
                MTLDataType::UInt,
                3,
            );
            constants.setConstantValue_type_atIndex(
                NonNull::from(&frames_per_tg).cast::<c_void>(),
                MTLDataType::UInt,
                4,
            );
        }

        let fn_name = NSString::from_str(key.pixel.kernel_name());
        let function = self
            .library
            .newFunctionWithName_constantValues_error(&fn_name, &constants)
            .map_err(|e| {
                MetalError::Device(format!(
                    "failed to specialize Metal function {}: {e}",
                    key.pixel.kernel_name()
                ))
            })?;
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                MetalError::Device(format!(
                    "failed to create Metal compute pipeline {}: {e}",
                    key.pixel.kernel_name()
                ))
            })?;
        let handle = Arc::new(MetalHandle(state));
        cache.insert(key, Arc::clone(&handle));
        Ok(handle)
    }

    fn bslz4_pipeline(
        &self,
        kind: Bslz4KernelKind,
    ) -> Result<Arc<MetalHandle<dyn MTLComputePipelineState>>, MetalError> {
        let mut cache = self
            .bslz4_pipelines
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(p) = cache.get(&kind) {
            return Ok(Arc::clone(p));
        }
        let name = match kind {
            Bslz4KernelKind::U16 => "bshuf_lz4_decode_u16",
            Bslz4KernelKind::U32 => "bshuf_lz4_decode_u32",
        };
        let fn_name = NSString::from_str(name);
        let function = self.library.newFunctionWithName(&fn_name).ok_or_else(|| {
            MetalError::Device(format!("Metal BSLZ4 kernel {name} not found in library"))
        })?;
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                MetalError::Device(format!("failed to create Metal BSLZ4 pipeline {name}: {e}"))
            })?;
        let handle = Arc::new(MetalHandle(state));
        cache.insert(kind, Arc::clone(&handle));
        Ok(handle)
    }

    fn alloc_shared_buffer(&self, bytes: usize) -> Result<MetalHandle<dyn MTLBuffer>, MetalError> {
        let len = bytes.max(16);
        let buf = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                MetalError::Device(format!(
                    "Metal failed to allocate shared buffer of {len} bytes"
                ))
            })?;
        Ok(MetalHandle(buf))
    }

    fn alloc_private_buffer(&self, bytes: usize) -> Result<MetalHandle<dyn MTLBuffer>, MetalError> {
        let len = bytes.max(16);
        let buf = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModePrivate)
            .ok_or_else(|| {
                MetalError::Device(format!(
                    "Metal failed to allocate private buffer of {len} bytes"
                ))
            })?;
        Ok(MetalHandle(buf))
    }

    fn upload_private_slice<T: Copy>(
        &self,
        data: &[T],
    ) -> Result<MetalHandle<dyn MTLBuffer>, MetalError> {
        let bytes = std::mem::size_of_val(data);
        if bytes == 0 {
            return self.alloc_private_buffer(16);
        }
        // SAFETY: `data.as_ptr()` is valid for `bytes` bytes; Metal copies immediately into staging.
        let staging = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(data.as_ptr() as *mut c_void),
                bytes,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| {
            MetalError::Device(format!(
                "Metal failed to stage private upload of {bytes} bytes"
            ))
        })?;
        let private = self.alloc_private_buffer(bytes)?;
        let cmd = self.queue.commandBuffer().ok_or_else(|| {
            MetalError::Device("failed to create Metal blit command buffer".into())
        })?;
        let blit = cmd
            .blitCommandEncoder()
            .ok_or_else(|| MetalError::Device("failed to create Metal blit encoder".into()))?;
        // SAFETY: both buffers are valid for `bytes` bytes and kept alive until waitUntilCompleted.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &staging, 0, &private, 0, bytes,
            );
        }
        blit.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if let Some(err) = cmd.error() {
            return Err(MetalError::Device(format!(
                "Metal private buffer upload failed: {err}"
            )));
        }
        Ok(private)
    }

    /// Binds a readable host slice zero-copy on Apple Silicon Unified Memory.
    ///
    /// First checks buffers allocated via `alloc_shared_bytes`; otherwise wraps
    /// the enclosing 16 KiB VM page span with `newBufferWithBytesNoCopy` and
    /// returns the byte offset to the slice start. On macOS ARM64, memory
    /// protection is per 16 KiB page, so every page intersecting a valid
    /// readable slice is mapped and readable.
    fn try_zero_copy_buffer(
        &self,
        ptr: *const u8,
        byte_len: usize,
    ) -> Option<(Arc<MetalHandle<dyn MTLBuffer>>, usize)> {
        if byte_len == 0 {
            return None;
        }
        let addr = ptr as usize;
        // 1. Fast path: buffer allocated by `alloc_shared_bytes`.
        {
            let allocs = self
                .shared_allocs
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            for (&base, buf) in allocs.iter() {
                let cap = buf.length();
                if addr >= base
                    && addr
                        .checked_add(byte_len)
                        .is_some_and(|end| end <= base + cap)
                {
                    return Some((Arc::clone(buf), addr - base));
                }
            }
        }
        // 2. General zero-copy path (Unified Memory only): wrap the enclosing VM page span.
        if self.info.has_unified_memory != Some(true) {
            return None;
        }
        let aligned_start = addr & !(PAGE_SIZE - 1);
        let end = addr.checked_add(byte_len)?;
        let aligned_end = end.div_ceil(PAGE_SIZE).checked_mul(PAGE_SIZE)?;
        let aligned_len = aligned_end - aligned_start;
        let offset = addr - aligned_start;
        if !is_rw_mapped(aligned_start, aligned_len) {
            return None;
        }
        // SAFETY: `[addr, end)` is an immutable borrowed slice valid until GPU completion;
        // `is_rw_mapped` verified every page in `[aligned_start, aligned_end)` is mapped
        // RW, and the GPU kernel only reads `[offset, offset + byte_len)`.
        let buf = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    NonNull::new(aligned_start as *mut c_void)?,
                    aligned_len,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }?;
        Some((Arc::new(MetalHandle(buf)), offset))
    }
}

/// Owned Metal shared byte buffer exposed to Python (`alloc_shared_batch`) or Rust callers.
pub(crate) struct SharedBytes {
    device: Arc<Device>,
    buf: Arc<MetalHandle<dyn MTLBuffer>>,
    base_ptr: usize,
    len: usize,
}

impl SharedBytes {
    pub(crate) fn new(device: Arc<Device>, bytes: usize) -> Result<Self, MetalError> {
        let padded = bytes.max(16).div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let raw = device.alloc_shared_buffer(padded)?;
        let base_ptr = raw.contents().as_ptr() as usize;
        let buf = Arc::new(raw);
        device
            .shared_allocs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(base_ptr, Arc::clone(&buf));
        Ok(Self {
            device,
            buf,
            base_ptr,
            len: bytes,
        })
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buf.contents().as_ptr().cast::<u8>()
    }

    #[allow(dead_code)]
    pub(crate) fn as_slice_mut(&mut self) -> &mut [u8] {
        // SAFETY: `self` owns unique mutable access to the buffer region of `self.len` bytes.
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }

    #[allow(dead_code)]
    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.buf` is backed by `>= self.len` valid shared bytes.
        unsafe { std::slice::from_raw_parts(self.buf.contents().as_ptr().cast::<u8>(), self.len) }
    }
}

impl Drop for SharedBytes {
    fn drop(&mut self) {
        self.device
            .shared_allocs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.base_ptr);
    }
}

/// Returns the cached state of Metal device `ordinal` (0 = system default device).
pub(super) fn device(ordinal: usize) -> Result<Arc<Device>, MetalInfo> {
    static DEVICES: OnceLock<Mutex<HashMap<usize, Arc<Device>>>> = OnceLock::new();
    let mut devices = DEVICES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(dev) = devices.get(&ordinal) {
        return Ok(Arc::clone(dev));
    }
    let mut info = MetalInfo::unavailable(ordinal, true, String::new());
    match initialise(ordinal, &mut info) {
        Ok(dev) => {
            let dev = Arc::new(dev);
            devices.insert(ordinal, Arc::clone(&dev));
            Ok(dev)
        }
        Err(reason) => {
            info.available = false;
            info.reason = Some(reason);
            Err(info)
        }
    }
}

fn initialise(ordinal: usize, info: &mut MetalInfo) -> Result<Device, String> {
    if ordinal != 0 {
        return Err(format!(
            "Metal device {ordinal} is not supported; use device 0 (system default GPU)"
        ));
    }
    let mtl_device =
        MTLCreateSystemDefaultDevice().ok_or_else(|| "no Metal GPU device found".to_string())?;
    info.device_name = Some(mtl_device.name().to_string());
    info.has_unified_memory = Some(mtl_device.hasUnifiedMemory());
    info.recommended_max_working_set_size =
        Some(mtl_device.recommendedMaxWorkingSetSize() as usize);
    info.max_threadgroup_memory_length = Some(mtl_device.maxThreadgroupMemoryLength());

    let queue = mtl_device
        .newCommandQueue()
        .ok_or_else(|| "failed to create Metal command queue".to_string())?;

    let start = Instant::now();
    let options = MTLCompileOptions::new();
    if options.respondsToSelector(sel!(setMathMode:)) {
        options.setMathMode(MTLMathMode::Safe);
    } else {
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
    }
    let source = NSString::from_str(include_str!("kernels.metal"));
    let library = mtl_device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|e| format!("failed to compile Metal kernels: {e}"))?;
    let compile_ms = start.elapsed().as_secs_f64() * 1_000.0;

    info.available = true;
    info.reason = None;
    Ok(Device {
        device: MetalHandle(mtl_device),
        queue: MetalHandle(queue),
        library: MetalHandle(library),
        pipelines: Mutex::new(HashMap::new()),
        bslz4_pipelines: Mutex::new(HashMap::new()),
        shared_allocs: Mutex::new(HashMap::new()),
        info: info.clone(),
        compile_ms,
    })
}

/// Host CSR table in 32-bit device index types.
struct HostTable<'a> {
    pixels: u32,
    bins: u32,
    entries: u32,
    rows: Vec<u32>,
    columns: Cow<'a, [u32]>,
    weights: Option<&'a [f64]>,
    denominator: Vec<f64>,
    radial: Vec<f64>,
}

fn remap_columns_for_order<'a>(
    columns: Cow<'a, [u32]>,
    shape: (usize, usize),
    order: InputOrder,
) -> Cow<'a, [u32]> {
    match order {
        InputOrder::Detector => columns,
        InputOrder::FlipVertical => {
            let rows = shape.0 as u32;
            let cols = shape.1 as u32;
            Cow::Owned(
                columns
                    .iter()
                    .map(|&idx| {
                        let r = idx / cols;
                        let c = idx % cols;
                        (rows - 1 - r) * cols + c
                    })
                    .collect(),
            )
        }
    }
}

fn with_table_1d<R>(
    integrator: &Integrator,
    shape: (usize, usize),
    bins: usize,
    order: InputOrder,
    use_table: impl FnOnce(&HostTable<'_>) -> Result<R, MetalError>,
) -> Result<R, MetalError> {
    let too_large = || MetalError::Unsupported("the Metal table exceeds 32-bit indices".into());
    let pixels = u32::try_from(shape.0 * shape.1).map_err(|_| too_large())?;
    let n_bins = u32::try_from(bins).map_err(|_| too_large())?;
    let view = integrator.for_shape(shape.0, shape.1);
    let norm = view.normalization();
    let key = integrator.one_d_key(bins);
    let lut = view.ensure_lut1d(key);
    let mask = integrator.mask.as_ref().map(|m| m.data.as_slice());
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
        .ok_or_else(|| MetalError::Device("the Metal table has no rows".into()))?;
    if entries > u32::MAX - 128 {
        return Err(MetalError::Unsupported(format!(
            "the Metal table has {entries} entries; at most {} are supported",
            u32::MAX - 128
        )));
    }
    let columns = remap_columns_for_order(
        match &table.col_idx {
            Columns::Compact(c) => Cow::Borrowed(c.as_slice()),
            Columns::Wide(c) => Cow::Owned(
                c.iter()
                    .map(|&col| u32::try_from(col))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| too_large())?,
            ),
        },
        shape,
        order,
    );
    use_table(&HostTable {
        pixels,
        bins: n_bins,
        entries,
        rows,
        columns,
        weights: table.values.as_deref(),
        denominator: denominator.to_vec(),
        radial: lut.bins(),
    })
}

fn with_table_2d<R>(
    integrator: &Integrator,
    shape: (usize, usize),
    n_radial: usize,
    n_azimuthal: usize,
    azimuthal_range: Option<(f64, f64)>,
    order: InputOrder,
    use_table: impl FnOnce(&HostTable<'_>, Option<Vec<f64>>, Option<Vec<f64>>) -> Result<R, MetalError>,
) -> Result<R, MetalError> {
    let too_large = || MetalError::Unsupported("the Metal table exceeds 32-bit indices".into());
    let pixels = u32::try_from(shape.0 * shape.1).map_err(|_| too_large())?;
    let total_bins = n_radial.checked_mul(n_azimuthal).ok_or_else(too_large)?;
    let n_bins = u32::try_from(total_bins).map_err(|_| too_large())?;
    let view = integrator.for_shape(shape.0, shape.1);
    let norm = view.normalization();
    let (key, radial_axis, azimuthal_axis) = match azimuthal_range {
        None => {
            let key = integrator
                .try_lut2d_key(&view, n_radial, n_azimuthal)
                .map_err(MetalError::InvalidInput)?;
            let radial = integrator.radial_range().map(|(lo, hi)| {
                crate::math::histogram::bins(n_radial, lo, (hi - lo) / n_radial as f64)
            });
            (key, radial, None)
        }
        Some(range) => {
            let grid = integrator
                .azimuthal_range_grid(&view, n_radial, n_azimuthal, range)
                .map_err(MetalError::InvalidInput)?;
            let rad = grid.radial_centers();
            let az = grid.azimuthal_centers();
            (integrator.shared_grid_key(grid), Some(rad), Some(az))
        }
    };
    let lut = view.ensure_lut2d(key);
    let mask = integrator.mask.as_ref().map(|m| m.data.as_slice());
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
        .ok_or_else(|| MetalError::Device("the Metal table has no rows".into()))?;
    if entries > u32::MAX - 128 {
        return Err(MetalError::Unsupported(format!(
            "the Metal table has {entries} entries; at most {} are supported",
            u32::MAX - 128
        )));
    }
    let columns = remap_columns_for_order(
        match &table.col_idx {
            Columns::Compact(c) => Cow::Borrowed(c.as_slice()),
            Columns::Wide(c) => Cow::Owned(
                c.iter()
                    .map(|&col| u32::try_from(col))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| too_large())?,
            ),
        },
        shape,
        order,
    );
    use_table(
        &HostTable {
            pixels,
            bins: n_bins,
            entries,
            rows,
            columns,
            weights: table.values.as_deref(),
            denominator: denominator.to_vec(),
            radial: Vec::new(),
        },
        radial_axis,
        azimuthal_axis,
    )
}

fn choose_layout(entries: u32, bins: u32, allow_subwarp: bool) -> Layout {
    let avg = (entries as usize) / (bins as usize).max(1);
    if allow_subwarp && avg <= SUBWARP_MAX_ENTRIES_PER_BIN {
        Layout::SubWarp
    } else if avg <= WARP_MAX_ENTRIES_PER_BIN {
        Layout::Warp
    } else {
        Layout::Block
    }
}

struct SlotScratch {
    input_buf: Option<MetalHandle<dyn MTLBuffer>>,
    input_cap: usize,
    output_buf: Option<MetalHandle<dyn MTLBuffer>>,
    output_cap: usize,
    bslz4_comp_buf: Option<MetalHandle<dyn MTLBuffer>>,
    bslz4_comp_cap: usize,
    bslz4_desc_buf: Option<MetalHandle<dyn MTLBuffer>>,
    bslz4_desc_cap: usize,
    bslz4_decoded_buf: Option<MetalHandle<dyn MTLBuffer>>,
    bslz4_decoded_cap: usize,
    bslz4_err_buf: Option<MetalHandle<dyn MTLBuffer>>,
}

impl SlotScratch {
    fn new() -> Self {
        Self {
            input_buf: None,
            input_cap: 0,
            output_buf: None,
            output_cap: 0,
            bslz4_comp_buf: None,
            bslz4_comp_cap: 0,
            bslz4_desc_buf: None,
            bslz4_desc_cap: 0,
            bslz4_decoded_buf: None,
            bslz4_decoded_cap: 0,
            bslz4_err_buf: None,
        }
    }

    fn ensure_input(
        &mut self,
        device: &Device,
        bytes: usize,
    ) -> Result<&MetalHandle<dyn MTLBuffer>, MetalError> {
        if self.input_cap < bytes || self.input_buf.is_none() {
            let alloc = bytes.max(16);
            self.input_buf = Some(device.alloc_shared_buffer(alloc)?);
            self.input_cap = alloc;
        }
        Ok(self.input_buf.as_ref().unwrap())
    }

    fn ensure_output(
        &mut self,
        device: &Device,
        bytes: usize,
    ) -> Result<&MetalHandle<dyn MTLBuffer>, MetalError> {
        if self.output_cap < bytes || self.output_buf.is_none() {
            let alloc = bytes.max(16);
            self.output_buf = Some(device.alloc_shared_buffer(alloc)?);
            self.output_cap = alloc;
        }
        Ok(self.output_buf.as_ref().unwrap())
    }

    fn ensure_bslz4(
        &mut self,
        device: &Device,
        comp_bytes: usize,
        desc_bytes: usize,
        decoded_bytes: usize,
    ) -> Result<(), MetalError> {
        if self.bslz4_comp_cap < comp_bytes || self.bslz4_comp_buf.is_none() {
            let alloc = comp_bytes.max(16);
            self.bslz4_comp_buf = Some(device.alloc_shared_buffer(alloc)?);
            self.bslz4_comp_cap = alloc;
        }
        if self.bslz4_desc_cap < desc_bytes || self.bslz4_desc_buf.is_none() {
            let alloc = desc_bytes.max(16);
            self.bslz4_desc_buf = Some(device.alloc_shared_buffer(alloc)?);
            self.bslz4_desc_cap = alloc;
        }
        if self.bslz4_err_buf.is_none() {
            self.bslz4_err_buf = Some(device.alloc_shared_buffer(16)?);
        }
        if self.bslz4_decoded_cap < decoded_bytes || self.bslz4_decoded_buf.is_none() {
            let alloc = decoded_bytes.max(16);
            self.bslz4_decoded_buf = Some(device.alloc_private_buffer(alloc)?);
            self.bslz4_decoded_cap = alloc;
        }
        Ok(())
    }
}

struct Scratch {
    slots: [SlotScratch; 2],
}

impl Scratch {
    fn new() -> Self {
        Self {
            slots: [SlotScratch::new(), SlotScratch::new()],
        }
    }
}

/// Raw per-frame `(intensity, optional_errors)` output before wrapping into 1D/2D results.
type RawFrameResults = Vec<(Vec<f64>, Option<Vec<f64>>)>;

/// Shared execution engine for both `Metal1dPlan` and `Metal2dPlan`.
pub(super) struct Engine {
    device: Arc<Device>,
    pixels: usize,
    bins: usize,
    entries: usize,
    weighted: bool,
    precision: MetalPrecision,
    layout: Layout,
    rows: MetalHandle<dyn MTLBuffer>,
    columns: MetalHandle<dyn MTLBuffer>,
    weights: MetalHandle<dyn MTLBuffer>,
    denominator: Vec<f64>,
    max_batch_frames: usize,
    scratch: Mutex<Scratch>,
}

impl Engine {
    fn build(
        device: Arc<Device>,
        table: &HostTable<'_>,
        precision: MetalPrecision,
        max_batch_bytes: Option<usize>,
        allow_subwarp: bool,
    ) -> Result<Self, MetalError> {
        let rows = device.upload_private_slice(&table.rows)?;
        let columns = if table.columns.is_empty() {
            device.upload_private_slice(&[0u32])?
        } else {
            device.upload_private_slice(table.columns.as_ref())?
        };
        let (weights, weighted) = match table.weights {
            Some(w) if !w.is_empty() => {
                let w_f32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
                (device.upload_private_slice(&w_f32)?, true)
            }
            _ => (device.upload_private_slice(&[1.0_f32])?, false),
        };

        let pixels = table.pixels as usize;
        let bins = table.bins as usize;
        let per_frame_bytes = pixels
            .checked_mul(size_of::<f32>())
            .and_then(|b| b.checked_add(bins.checked_mul(4 * size_of::<f32>())?))
            .ok_or_else(|| MetalError::InvalidInput("frame byte size overflow".into()))?;
        let budget = match max_batch_bytes {
            Some(b) => b,
            None => {
                let working_set = device
                    .info
                    .recommended_max_working_set_size
                    .unwrap_or(DEFAULT_BATCH_BYTES);
                DEFAULT_BATCH_BYTES
                    .min(working_set / 4)
                    .max(per_frame_bytes)
            }
        };
        let max_batch_frames = budget / per_frame_bytes.max(1);
        if max_batch_frames == 0 {
            return Err(MetalError::InvalidInput(format!(
                "max_batch_bytes ({budget}) is too small for one frame ({per_frame_bytes} bytes)"
            )));
        }

        let layout = choose_layout(table.entries, table.bins, allow_subwarp);

        // Pre-warm the default f32 intensity-only pipeline so first call has no stall.
        let _ = device.pipeline(PipelineKey {
            pixel: PixelKind::F32,
            acc_mode: precision.acc_mode(),
            bins_per_tg: layout.bins_per_tg(),
            frames_per_tg: 8,
            weighted,
            with_variance: false,
        })?;

        Ok(Self {
            device,
            pixels,
            bins,
            entries: table.entries as usize,
            weighted,
            precision,
            layout,
            rows,
            columns,
            weights,
            denominator: table.denominator.clone(),
            max_batch_frames,
            scratch: Mutex::new(Scratch::new()),
        })
    }

    pub(super) fn alloc_shared_bytes(&self, bytes: usize) -> Result<SharedBytes, MetalError> {
        SharedBytes::new(Arc::clone(&self.device), bytes)
    }

    #[cfg(test)]
    pub(super) fn layout(&self) -> Layout {
        self.layout
    }

    fn tile_frames(&self, total: usize, elem_bytes: usize) -> usize {
        let max_by_slot = (SLOT_BYTES / (self.pixels * elem_bytes).max(1)).max(1);
        let tile = total
            .min(self.max_batch_frames)
            .min(FRAMES_PER_LAUNCH as usize)
            .min(max_by_slot);
        // Round down to a multiple of 8 when >= 8 so 8-frame tiling stays full.
        if tile >= 8 && tile < total {
            (tile / 8) * 8
        } else if tile >= 4 && tile < total {
            (tile / 4) * 4
        } else {
            tile.max(1)
        }
    }

    fn encode_integrate(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        pixel: PixelKind,
        input_buf: &ProtocolObject<dyn MTLBuffer>,
        input_offset: usize,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        num_frames: u32,
        errors: bool,
        layout_override: Option<Layout>,
    ) -> Result<(), MetalError> {
        let layout = layout_override.unwrap_or(self.layout);
        let acc_mode = self.precision.acc_mode();
        let frames_per_tg = if acc_mode == 0 || !errors { 8u32 } else { 4u32 };
        let pipeline = self.device.pipeline(PipelineKey {
            pixel,
            acc_mode,
            bins_per_tg: layout.bins_per_tg(),
            frames_per_tg,
            weighted: self.weighted,
            with_variance: errors,
        })?;

        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::Device("failed to create Metal compute encoder".into()))?;
        encoder.setComputePipelineState(&pipeline);

        let params = KernelParams {
            pixels: self.pixels as u32,
            bins: self.bins as u32,
            num_frames,
        };
        // SAFETY: buffers are alive until command buffer finishes; indices match kernels.metal.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&*self.rows), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&*self.columns), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&*self.weights), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(input_buf), input_offset, 3);
            encoder.setBuffer_offset_atIndex(Some(output_buf), 0, 4);
            encoder.setBytes_length_atIndex(
                NonNull::from(&params).cast::<c_void>(),
                size_of::<KernelParams>(),
                5,
            );
        }

        let grid_x = self.bins.div_ceil(layout.bins_per_tg() as usize);
        let grid_y = (num_frames as usize).div_ceil(frames_per_tg as usize);
        let tg_grid = MTLSize {
            width: grid_x,
            height: grid_y,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(tg_grid, tg_size);
        encoder.endEncoding();
        Ok(())
    }

    fn drain_output(
        &self,
        output_buf: &ProtocolObject<dyn MTLBuffer>,
        frame_count: usize,
        errors: bool,
        out_intensity: &mut [f64],
        out_errors: Option<&mut [f64]>,
    ) {
        let stride = if errors { 4usize } else { 2usize };
        let total_floats = frame_count * self.bins * stride;
        // SAFETY: `output_buf` is StorageModeShared and GPU execution has completed.
        let raw = unsafe {
            std::slice::from_raw_parts(output_buf.contents().as_ptr().cast::<f32>(), total_floats)
        };
        let bins = self.bins;
        let den = &self.denominator;
        let parallel = frame_count * bins >= 32_768;

        if let Some(err_slice) = out_errors {
            if parallel {
                out_intensity
                    .par_chunks_exact_mut(bins)
                    .zip(err_slice.par_chunks_exact_mut(bins))
                    .enumerate()
                    .for_each(|(f, (int_row, err_row))| {
                        let base = f * bins * 4;
                        for b in 0..bins {
                            let d = den[b];
                            if d == 0.0 {
                                int_row[b] = 0.0;
                                err_row[b] = 0.0;
                            } else {
                                let i = base + b * 4;
                                let sig = raw[i] as f64 + raw[i + 1] as f64;
                                let var = raw[i + 2] as f64 + raw[i + 3] as f64;
                                int_row[b] = sig / d;
                                err_row[b] = var.max(0.0).sqrt() / d.abs();
                            }
                        }
                    });
            } else {
                for f in 0..frame_count {
                    let base = f * bins * 4;
                    let int_row = &mut out_intensity[f * bins..(f + 1) * bins];
                    let err_row = &mut err_slice[f * bins..(f + 1) * bins];
                    for b in 0..bins {
                        let d = den[b];
                        if d == 0.0 {
                            int_row[b] = 0.0;
                            err_row[b] = 0.0;
                        } else {
                            let i = base + b * 4;
                            let sig = raw[i] as f64 + raw[i + 1] as f64;
                            let var = raw[i + 2] as f64 + raw[i + 3] as f64;
                            int_row[b] = sig / d;
                            err_row[b] = var.max(0.0).sqrt() / d.abs();
                        }
                    }
                }
            }
        } else if parallel {
            out_intensity
                .par_chunks_exact_mut(bins)
                .enumerate()
                .for_each(|(f, int_row)| {
                    let base = f * bins * 2;
                    for b in 0..bins {
                        let d = den[b];
                        if d == 0.0 {
                            int_row[b] = 0.0;
                        } else {
                            let i = base + b * 2;
                            let sig = raw[i] as f64 + raw[i + 1] as f64;
                            int_row[b] = sig / d;
                        }
                    }
                });
        } else {
            for f in 0..frame_count {
                let base = f * bins * 2;
                let int_row = &mut out_intensity[f * bins..(f + 1) * bins];
                for b in 0..bins {
                    let d = den[b];
                    if d == 0.0 {
                        int_row[b] = 0.0;
                    } else {
                        let i = base + b * 2;
                        let sig = raw[i] as f64 + raw[i + 1] as f64;
                        int_row[b] = sig / d;
                    }
                }
            }
        }
    }

    pub(super) fn run_into(
        &self,
        frames: Frames<'_, '_>,
        stacked: bool,
        out_intensity: &mut [f64],
        out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        self.run_into_with_layout(frames, stacked, out_intensity, out_errors, None)
    }

    pub(super) fn run_into_with_layout(
        &self,
        frames: Frames<'_, '_>,
        stacked: bool,
        out_intensity: &mut [f64],
        mut out_errors: Option<&mut [f64]>,
        layout_override: Option<Layout>,
    ) -> Result<(), MetalError> {
        let total = frames
            .count(self.pixels, stacked)
            .map_err(MetalError::InvalidInput)?;
        if total == 0 {
            return Ok(());
        }
        let errors = out_errors.is_some();
        let pixel_kind = match frames {
            Frames::F64(_) | Frames::F32(_) => PixelKind::F32,
            Frames::U16(_) => PixelKind::U16,
            Frames::U32(_) => PixelKind::U32,
            Frames::I32(_) => PixelKind::I32,
        };
        let elem_bytes = pixel_kind.elem_bytes();
        let tile = self.tile_frames(total, elem_bytes);
        let out_stride_bytes = if errors { 4 } else { 2 } * size_of::<f32>();

        let mut scratch = self.scratch.lock().unwrap_or_else(PoisonError::into_inner);

        /// RAII guard for a committed command buffer: guarantees `waitUntilCompleted`
        /// runs even on early `?` returns so GPU work can never outlive borrowed
        /// zero-copy caller memory or the scratch mutex lock.
        struct InFlightTile {
            cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
            slot_idx: usize,
            start_frame: usize,
            count: usize,
            _keep_alive: Option<Arc<MetalHandle<dyn MTLBuffer>>>,
            waited: bool,
        }

        impl InFlightTile {
            fn wait(mut self) -> Result<(usize, usize, usize), MetalError> {
                self.cmd_buf.waitUntilCompleted();
                self.waited = true;
                if let Some(err) = self.cmd_buf.error() {
                    return Err(MetalError::Device(format!(
                        "Metal command buffer failed: {err}"
                    )));
                }
                Ok((self.slot_idx, self.start_frame, self.count))
            }
        }

        impl Drop for InFlightTile {
            fn drop(&mut self) {
                if !self.waited {
                    self.cmd_buf.waitUntilCompleted();
                }
            }
        }

        let mut pending: Option<InFlightTile> = None;

        let mut offset = 0usize;
        let mut slot_idx = 0usize;

        while offset < total {
            let count = (total - offset).min(tile);
            let out_bytes = count * self.bins * out_stride_bytes;
            let in_bytes = count * self.pixels * elem_bytes;

            let zero_copy = if stacked || total == 1 {
                frames
                    .contiguous_bytes(offset, count, self.pixels)
                    .and_then(|(ptr, len)| self.device.try_zero_copy_buffer(ptr, len))
            } else {
                None
            };

            let slot = &mut scratch.slots[slot_idx];
            slot.ensure_output(&self.device, out_bytes)?;
            if zero_copy.is_none() {
                slot.ensure_input(&self.device, in_bytes)?;
                let dst = slot.input_buf.as_ref().unwrap().contents().as_ptr();
                stage_frames_into(frames, stacked, offset, count, self.pixels, dst);
            }

            let cmd_buf = self.device.queue.commandBuffer().ok_or_else(|| {
                MetalError::Device("failed to create Metal command buffer".into())
            })?;

            let kept_zc = match zero_copy {
                Some((zc_buf, zc_off)) => {
                    self.encode_integrate(
                        &cmd_buf,
                        pixel_kind,
                        &zc_buf,
                        zc_off,
                        slot.output_buf.as_ref().unwrap(),
                        count as u32,
                        errors,
                        layout_override,
                    )?;
                    Some(zc_buf)
                }
                None => {
                    self.encode_integrate(
                        &cmd_buf,
                        pixel_kind,
                        slot.input_buf.as_ref().unwrap(),
                        0,
                        slot.output_buf.as_ref().unwrap(),
                        count as u32,
                        errors,
                        layout_override,
                    )?;
                    None
                }
            };
            cmd_buf.commit();

            let next_tile = InFlightTile {
                cmd_buf,
                slot_idx,
                start_frame: offset,
                count,
                _keep_alive: kept_zc,
                waited: false,
            };
            if let Some(prev) = pending.replace(next_tile) {
                let (prev_slot, prev_off, prev_count) = prev.wait()?;
                let s_int =
                    &mut out_intensity[prev_off * self.bins..(prev_off + prev_count) * self.bins];
                let s_err = out_errors
                    .as_deref_mut()
                    .map(|e| &mut e[prev_off * self.bins..(prev_off + prev_count) * self.bins]);
                self.drain_output(
                    scratch.slots[prev_slot].output_buf.as_ref().unwrap(),
                    prev_count,
                    errors,
                    s_int,
                    s_err,
                );
            }

            offset += count;
            slot_idx ^= 1;
        }

        if let Some(prev) = pending {
            let (prev_slot, prev_off, prev_count) = prev.wait()?;
            let s_int =
                &mut out_intensity[prev_off * self.bins..(prev_off + prev_count) * self.bins];
            let s_err = out_errors
                .map(|e| &mut e[prev_off * self.bins..(prev_off + prev_count) * self.bins]);
            self.drain_output(
                scratch.slots[prev_slot].output_buf.as_ref().unwrap(),
                prev_count,
                errors,
                s_int,
                s_err,
            );
        }

        Ok(())
    }

    pub(super) fn run_raw(
        &self,
        frames: Frames<'_, '_>,
        stacked: bool,
        errors: bool,
    ) -> Result<RawFrameResults, MetalError> {
        let total = frames
            .count(self.pixels, stacked)
            .map_err(MetalError::InvalidInput)?;
        if total == 0 {
            return Ok(Vec::new());
        }
        let mut intensity = vec![0.0_f64; total * self.bins];
        let mut errs = errors.then(|| vec![0.0_f64; total * self.bins]);
        self.run_into(frames, stacked, &mut intensity, errs.as_deref_mut())?;
        Ok(split_results(intensity, errs, total, self.bins))
    }

    pub(super) fn run_1d(
        &self,
        frames: Frames<'_, '_>,
        stacked: bool,
        errors: bool,
        radial: &[f64],
    ) -> Result<Vec<Fixed1dResult>, MetalError> {
        let raw = self.run_raw(frames, stacked, errors)?;
        Ok(raw
            .into_iter()
            .map(|(intensity, errors)| Fixed1dResult {
                radial: radial.to_vec(),
                intensity,
                errors,
            })
            .collect())
    }

    pub(super) fn run_bslz4_into(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        out_intensity: &mut [f64],
        mut out_errors: Option<&mut [f64]>,
    ) -> Result<(), MetalError> {
        if chunks.is_empty() {
            return Ok(());
        }
        let errors = out_errors.is_some();
        let elem_bytes = dtype.elem_bytes();
        let pixel_kind = match dtype {
            Bslz4Dtype::U16 => PixelKind::U16,
            Bslz4Dtype::U32 => PixelKind::U32,
            Bslz4Dtype::I32 => PixelKind::I32,
        };
        let bslz4_kind = match dtype {
            Bslz4Dtype::U16 => Bslz4KernelKind::U16,
            Bslz4Dtype::U32 | Bslz4Dtype::I32 => Bslz4KernelKind::U32,
        };
        let bslz4_pipeline = self.device.bslz4_pipeline(bslz4_kind)?;
        let max_tg_mem = self
            .device
            .info
            .max_threadgroup_memory_length
            .unwrap_or(32_768);
        let tile = self.tile_frames(chunks.len(), elem_bytes);
        let out_stride_bytes = if errors { 4 } else { 2 } * size_of::<f32>();

        let mut scratch = self.scratch.lock().unwrap_or_else(PoisonError::into_inner);

        let mut offset = 0usize;
        while offset < chunks.len() {
            let count = (chunks.len() - offset).min(tile);
            let tile_chunks = &chunks[offset..offset + count];

            let (descs, total_comp, max_block_bytes) =
                parse_bslz4_tile(tile_chunks, self.pixels, elem_bytes, max_tg_mem)?;

            let decoded_bytes = count * self.pixels * elem_bytes;
            let out_bytes = count * self.bins * out_stride_bytes;
            let desc_bytes = descs.len() * size_of::<u32>();

            let slot = &mut scratch.slots[0];
            slot.ensure_bslz4(&self.device, total_comp, desc_bytes, decoded_bytes)?;
            slot.ensure_output(&self.device, out_bytes)?;

            // Copy compressed chunks into shared Metal buffer.
            {
                let comp_ptr = slot
                    .bslz4_comp_buf
                    .as_ref()
                    .unwrap()
                    .contents()
                    .as_ptr()
                    .cast::<u8>();
                let mut pos = 0usize;
                for c in tile_chunks {
                    // SAFETY: `comp_buf` has capacity >= `total_comp`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(c.as_ptr(), comp_ptr.add(pos), c.len());
                    }
                    pos += c.len();
                }
            }
            // Copy descriptors and clear error flag.
            {
                let desc_ptr = slot
                    .bslz4_desc_buf
                    .as_ref()
                    .unwrap()
                    .contents()
                    .as_ptr()
                    .cast::<u32>();
                // SAFETY: `desc_buf` has capacity >= `desc_bytes`.
                unsafe {
                    std::ptr::copy_nonoverlapping(descs.as_ptr(), desc_ptr, descs.len());
                }
                let err_ptr = slot
                    .bslz4_err_buf
                    .as_ref()
                    .unwrap()
                    .contents()
                    .as_ptr()
                    .cast::<u32>();
                // SAFETY: `err_buf` has capacity >= 16 bytes.
                unsafe {
                    *err_ptr = 0;
                }
            }

            let cmd_buf = self.device.queue.commandBuffer().ok_or_else(|| {
                MetalError::Device("failed to create Metal command buffer".into())
            })?;

            // Pass 1: GPU BSLZ4 decode into `slot.input_buf`.
            let n_blocks = descs.len() / 4;
            if n_blocks > 0 {
                let enc = cmd_buf.computeCommandEncoder().ok_or_else(|| {
                    MetalError::Device("failed to create Metal BSLZ4 compute encoder".into())
                })?;
                enc.setComputePipelineState(&bslz4_pipeline);
                // SAFETY: buffers are valid for the lifetime of the command buffer.
                unsafe {
                    enc.setBuffer_offset_atIndex(
                        Some(&**slot.bslz4_comp_buf.as_ref().unwrap()),
                        0,
                        0,
                    );
                    enc.setBuffer_offset_atIndex(
                        Some(&**slot.bslz4_desc_buf.as_ref().unwrap()),
                        0,
                        1,
                    );
                    enc.setBuffer_offset_atIndex(
                        Some(&**slot.bslz4_decoded_buf.as_ref().unwrap()),
                        0,
                        2,
                    );
                    enc.setBuffer_offset_atIndex(
                        Some(&**slot.bslz4_err_buf.as_ref().unwrap()),
                        0,
                        3,
                    );
                    let tg_mem = (max_block_bytes as usize).max(16).div_ceil(16) * 16;
                    enc.setThreadgroupMemoryLength_atIndex(tg_mem, 0);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: n_blocks,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.endEncoding();
            }

            // Pass 2: Integrate decoded frames in-place within the same command buffer.
            self.encode_integrate(
                &cmd_buf,
                pixel_kind,
                slot.bslz4_decoded_buf.as_ref().unwrap(),
                0,
                slot.output_buf.as_ref().unwrap(),
                count as u32,
                errors,
                None,
            )?;

            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
            if let Some(err) = cmd_buf.error() {
                return Err(MetalError::Device(format!(
                    "Metal BSLZ4 command buffer failed: {err}"
                )));
            }
            // SAFETY: GPU execution has completed and `err_buf` is StorageModeShared.
            let err_val = unsafe {
                *slot
                    .bslz4_err_buf
                    .as_ref()
                    .unwrap()
                    .contents()
                    .as_ptr()
                    .cast::<u32>()
            };
            if err_val != 0 {
                return Err(MetalError::InvalidInput(
                    "corrupt or invalid Bitshuffle/LZ4 stream".into(),
                ));
            }

            let s_int = &mut out_intensity[offset * self.bins..(offset + count) * self.bins];
            let s_err = out_errors
                .as_deref_mut()
                .map(|e| &mut e[offset * self.bins..(offset + count) * self.bins]);
            self.drain_output(
                slot.output_buf.as_ref().unwrap(),
                count,
                errors,
                s_int,
                s_err,
            );
            offset += count;
        }
        Ok(())
    }

    pub(super) fn run_bslz4_raw(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
    ) -> Result<RawFrameResults, MetalError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        let mut intensity = vec![0.0_f64; chunks.len() * self.bins];
        let mut errs = errors.then(|| vec![0.0_f64; chunks.len() * self.bins]);
        self.run_bslz4_into(chunks, dtype, &mut intensity, errs.as_deref_mut())?;
        Ok(split_results(intensity, errs, chunks.len(), self.bins))
    }

    pub(super) fn run_bslz4_1d(
        &self,
        chunks: &[&[u8]],
        dtype: Bslz4Dtype,
        errors: bool,
        radial: &[f64],
    ) -> Result<Vec<Fixed1dResult>, MetalError> {
        let raw = self.run_bslz4_raw(chunks, dtype, errors)?;
        Ok(raw
            .into_iter()
            .map(|(intensity, errors)| Fixed1dResult {
                radial: radial.to_vec(),
                intensity,
                errors,
            })
            .collect())
    }
}

fn split_results(
    intensity: Vec<f64>,
    errs: Option<Vec<f64>>,
    total: usize,
    bins: usize,
) -> Vec<(Vec<f64>, Option<Vec<f64>>)> {
    let mut out = Vec::with_capacity(total);
    match errs {
        Some(err_vec) => {
            for f in 0..total {
                let s = f * bins;
                out.push((
                    intensity[s..s + bins].to_vec(),
                    Some(err_vec[s..s + bins].to_vec()),
                ));
            }
        }
        None => {
            for f in 0..total {
                let s = f * bins;
                out.push((intensity[s..s + bins].to_vec(), None));
            }
        }
    }
    out
}

fn stage_frames_into(
    frames: Frames<'_, '_>,
    stacked: bool,
    offset: usize,
    count: usize,
    pixels: usize,
    dst: *mut c_void,
) {
    match frames {
        Frames::F64(slices) => {
            // SAFETY: `dst` has capacity for `count * pixels` f32 elements.
            let out = unsafe { std::slice::from_raw_parts_mut(dst.cast::<f32>(), count * pixels) };
            if stacked {
                let src = &slices[0][offset * pixels..(offset + count) * pixels];
                if src.len() >= PARALLEL_STAGE_THRESHOLD {
                    out.par_iter_mut()
                        .zip(src.par_iter())
                        .for_each(|(d, &s)| *d = s as f32);
                } else {
                    for (d, &s) in out.iter_mut().zip(src.iter()) {
                        *d = s as f32;
                    }
                }
            } else {
                out.par_chunks_exact_mut(pixels)
                    .zip(slices[offset..offset + count].par_iter())
                    .for_each(|(dst_frame, src_frame)| {
                        for (d, &s) in dst_frame.iter_mut().zip(src_frame.iter()) {
                            *d = s as f32;
                        }
                    });
            }
        }
        Frames::F32(slices) => {
            copy_typed(slices, stacked, offset, count, pixels, dst.cast::<f32>())
        }
        Frames::U16(slices) => {
            copy_typed(slices, stacked, offset, count, pixels, dst.cast::<u16>())
        }
        Frames::U32(slices) => {
            copy_typed(slices, stacked, offset, count, pixels, dst.cast::<u32>())
        }
        Frames::I32(slices) => {
            copy_typed(slices, stacked, offset, count, pixels, dst.cast::<i32>())
        }
    }
}

fn copy_typed<T: Copy + Send + Sync>(
    slices: &[&[T]],
    stacked: bool,
    offset: usize,
    count: usize,
    pixels: usize,
    dst: *mut T,
) {
    // SAFETY: `dst` has capacity for `count * pixels` elements of `T`.
    let out = unsafe { std::slice::from_raw_parts_mut(dst, count * pixels) };
    if stacked {
        let src = &slices[0][offset * pixels..(offset + count) * pixels];
        if src.len() >= PARALLEL_STAGE_THRESHOLD * 4 {
            out.par_chunks_mut(pixels)
                .zip(src.par_chunks(pixels))
                .for_each(|(d, s)| d.copy_from_slice(s));
        } else {
            out.copy_from_slice(src);
        }
    } else if count >= 2 {
        out.par_chunks_exact_mut(pixels)
            .zip(slices[offset..offset + count].par_iter())
            .for_each(|(d, s)| d.copy_from_slice(s));
    } else {
        out.copy_from_slice(slices[offset]);
    }
}

fn parse_bslz4_tile(
    chunks: &[&[u8]],
    pixels: usize,
    elem_bytes: usize,
    max_tg_mem: usize,
) -> Result<(Vec<u32>, usize, u32), MetalError> {
    let expected_uncomp = pixels * elem_bytes;
    let total_comp: usize = chunks.iter().map(|c| c.len()).sum();
    if total_comp > u32::MAX as usize {
        return Err(MetalError::InvalidInput(
            "compressed tile exceeds 4 GiB".into(),
        ));
    }
    let mut descs: Vec<u32> = Vec::with_capacity(chunks.len() * (expected_uncomp / 8192 + 2) * 4);
    let mut max_block_bytes: u32 = 0;
    let mut base_off: u32 = 0;
    let u32_overflow =
        || MetalError::InvalidInput("Bitshuffle/LZ4 offset exceeds 32-bit range".into());

    for (frame_idx, chunk) in chunks.iter().enumerate() {
        if chunk.len() < 12 {
            return Err(MetalError::InvalidInput(
                "Bitshuffle/LZ4 chunk shorter than 12-byte header".into(),
            ));
        }
        let uncomp_bytes = u64::from_be_bytes(chunk[0..8].try_into().unwrap()) as usize;
        if uncomp_bytes != expected_uncomp {
            return Err(MetalError::InvalidInput(format!(
                "Bitshuffle/LZ4 chunk uncompressed size {uncomp_bytes} does not match \
                 plan frame size {expected_uncomp} bytes"
            )));
        }
        let raw_block = u32::from_be_bytes(chunk[8..12].try_into().unwrap()) as usize;
        let block_bytes = if raw_block == 0 { 8192 } else { raw_block };
        if block_bytes == 0
            || block_bytes > max_tg_mem
            || !block_bytes.is_multiple_of(8 * elem_bytes)
        {
            return Err(MetalError::InvalidInput(format!(
                "unsupported Bitshuffle/LZ4 block size {block_bytes} bytes (max {max_tg_mem})"
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
                return Err(MetalError::InvalidInput(
                    "truncated Bitshuffle/LZ4 block header".into(),
                ));
            }
            let comp_len = u32::from_be_bytes(chunk[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            if comp_len > max_full_comp {
                return Err(MetalError::InvalidInput(format!(
                    "Bitshuffle/LZ4 compressed block length {comp_len} exceeds LZ4 bound {max_full_comp}"
                )));
            }
            if off
                .checked_add(comp_len)
                .is_none_or(|end| end > chunk.len())
            {
                return Err(MetalError::InvalidInput(
                    "truncated Bitshuffle/LZ4 block payload".into(),
                ));
            }
            let src_off = base_off
                .checked_add(u32::try_from(off).map_err(|_| u32_overflow())?)
                .ok_or_else(u32_overflow)?;
            let comp_len_u32 = u32::try_from(comp_len).map_err(|_| u32_overflow())?;
            let dst_off = u32::try_from(frame_dst + b * block_elems).map_err(|_| u32_overflow())?;
            descs.push(src_off);
            descs.push(comp_len_u32);
            descs.push(dst_off);
            descs.push(block_bytes_u32);
            off += comp_len;
        }
        if rem_comp_elems > 0 {
            if off.checked_add(4).is_none_or(|end| end > chunk.len()) {
                return Err(MetalError::InvalidInput(
                    "truncated Bitshuffle/LZ4 partial block header".into(),
                ));
            }
            let comp_len = u32::from_be_bytes(chunk[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            let rem_bytes = rem_comp_elems * elem_bytes;
            let max_rem_comp = rem_bytes + rem_bytes / 255 + 16;
            if comp_len > max_rem_comp {
                return Err(MetalError::InvalidInput(format!(
                    "Bitshuffle/LZ4 partial compressed block length {comp_len} exceeds LZ4 bound {max_rem_comp}"
                )));
            }
            if off
                .checked_add(comp_len)
                .is_none_or(|end| end > chunk.len())
            {
                return Err(MetalError::InvalidInput(
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
                return Err(MetalError::InvalidInput(
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
    Ok((descs, total_comp, max_block_bytes))
}

pub(super) fn prepare(
    integrator: &Integrator,
    shape: (usize, usize),
    _pixels: usize,
    n_points: usize,
    options: &Metal1dOptions,
) -> Result<Metal1dPlan, MetalError> {
    let dev = device(options.device()).map_err(|info| {
        MetalError::Unavailable(info.reason.unwrap_or_else(|| "unknown reason".into()))
    })?;
    let device_name = dev.info.device_name.clone().unwrap_or_default();
    with_table_1d(
        integrator,
        shape,
        n_points,
        options.input_order(),
        |table| {
            let engine = Engine::build(
                Arc::clone(&dev),
                table,
                options.precision(),
                options.max_batch_bytes(),
                false,
            )?;
            Ok(Metal1dPlan {
                shape,
                n_points,
                precision: options.precision(),
                device: options.device(),
                device_name,
                radial: table.radial.clone(),
                max_batch_frames: engine.max_batch_frames,
                table_entries: engine.entries,
                engine,
            })
        },
    )
}

pub(super) fn prepare_2d(
    integrator: &Integrator,
    shape: (usize, usize),
    _pixels: usize,
    n_radial: usize,
    n_azimuthal: usize,
    azimuthal_range: Option<(f64, f64)>,
    options: &Metal2dOptions,
) -> Result<Metal2dPlan, MetalError> {
    let dev = device(options.device()).map_err(|info| {
        MetalError::Unavailable(info.reason.unwrap_or_else(|| "unknown reason".into()))
    })?;
    let device_name = dev.info.device_name.clone().unwrap_or_default();
    with_table_2d(
        integrator,
        shape,
        n_radial,
        n_azimuthal,
        azimuthal_range,
        options.input_order(),
        |table, radial_axis, azimuthal_axis| {
            let engine = Engine::build(
                Arc::clone(&dev),
                table,
                options.precision(),
                options.max_batch_bytes(),
                true,
            )?;
            Ok(Metal2dPlan {
                shape,
                n_radial,
                n_azimuthal,
                precision: options.precision(),
                device: options.device(),
                device_name,
                radial: radial_axis,
                azimuthal: azimuthal_axis,
                max_batch_frames: engine.max_batch_frames,
                table_entries: engine.entries,
                engine,
            })
        },
    )
}
