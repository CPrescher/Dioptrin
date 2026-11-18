//! Internal fixed-mask CUDA prototype, exercised through its tests and benchmark.
use std::{borrow::Cow, error::Error, sync::Arc, time::Instant};

use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, HostSlice, LaunchConfig,
        PushKernelArg, sys::CUresult,
    },
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

use super::{IntegrationMethod, Integrator, MaskMode};
use crate::lut::prepared::Columns;

mod bench;
mod tests;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug)]
enum Accumulation {
    Plain,
    Compensated,
    Double,
}

impl Accumulation {
    const ALL: [Self; 3] = [Self::Plain, Self::Compensated, Self::Double];

    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Compensated => "compensated",
            Self::Double => "double",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    Block,
    Warp,
}

impl Layout {
    const ALL: [Self; 2] = [Self::Block, Self::Warp];

    fn name(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::Warp => "warp",
        }
    }

    fn bins_per_block(self) -> u32 {
        match self {
            Self::Block => 1,
            Self::Warp => 4,
        }
    }
}

struct Cuda {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    compile_ms: f64,
    module_ms: f64,
}

impl Cuda {
    fn open() -> Result<Option<Self>> {
        // In a CPU-only environment, probing avoids cudarc's missing-library panic.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Ok(None);
        }
        match CudaContext::device_count() {
            Ok(0) => return Ok(None),
            Err(err) if err.0 == CUresult::CUDA_ERROR_NO_DEVICE => return Ok(None),
            result => result?,
        };
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Ok(None);
        }
        let context = CudaContext::new(0)?;
        let stream = context.new_stream()?;
        let (major, minor) = context.compute_capability()?;
        let start = Instant::now();
        let ptx = compile_ptx_with_opts(
            include_str!("cuda/kernels.cu"),
            CompileOptions {
                fmad: Some(false),
                options: vec![
                    "--std=c++17".into(),
                    format!("--gpu-architecture=compute_{major}{minor}"),
                ],
                ..Default::default()
            },
        )?;
        let compile_ms = elapsed_ms(start);
        let start = Instant::now();
        let module = context.load_module(ptx)?;
        let module_ms = elapsed_ms(start);
        Ok(Some(Self {
            context,
            stream,
            module,
            compile_ms,
            module_ms,
        }))
    }

    fn prepare(
        &self,
        integrator: &Integrator,
        shape: (usize, usize),
        bins: u32,
    ) -> Result<Plan<'_>> {
        let pixels = shape.0.checked_mul(shape.1).ok_or("image shape overflow")?;
        let pixels = u32::try_from(pixels)?;
        if pixels == 0 || bins == 0 || bins > i32::MAX as u32 {
            return Err(
                "CUDA requires positive image and bin dimensions within its index range".into(),
            );
        }
        integrator.check_frame_shape(shape.0, shape.1)?;
        if integrator.mask_mode != MaskMode::Fixed
            || !matches!(
                integrator.method,
                IntegrationMethod::Counting
                    | IntegrationMethod::PixelSplit
                    | IntegrationMethod::PyfaiFull
            )
        {
            return Err(
                "CUDA requires a fixed-mask counting, pixel_split or pyfai_full table".into(),
            );
        }
        let view = integrator.for_shape(shape.0, shape.1);
        let norm = view.normalization();
        let key = integrator
            .one_d_table_key(bins as usize)
            .ok_or("missing LUT key")?;
        let lut = view.ensure_lut1d(key);
        let mask = integrator.mask.as_ref().map(|mask| mask.data.as_slice());
        let (table, denominator) = lut.fixed_csr_parts(mask, &norm);
        let rows = table
            .row_ptr
            .iter()
            .copied()
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let entries = rows.last().copied().ok_or("missing CUDA table rows")?;
        if entries > u32::MAX - 128 {
            return Err("CUDA table has too many entries".into());
        }
        let columns = match &table.col_idx {
            Columns::Compact(columns) => Cow::Borrowed(columns.as_slice()),
            Columns::Wide(columns) => Cow::Owned(
                columns
                    .iter()
                    .copied()
                    .map(u32::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            ),
        };
        let weights = table
            .values
            .as_ref()
            .map(|weights| weights.iter().map(|&w| w as f32).collect::<Vec<_>>());
        let start = Instant::now();
        let rows = self.stream.clone_htod(&rows)?;
        // Use one dummy entry for fully masked tables and implicit unit weights.
        let columns = self.stream.clone_htod(if columns.is_empty() {
            &[0u32][..]
        } else {
            columns.as_ref()
        })?;
        let w = weights
            .as_deref()
            .filter(|w| !w.is_empty())
            .unwrap_or(&[1.0]);
        let weights_device = self.stream.clone_htod(w)?;
        let denominator = self.stream.clone_htod(denominator)?;
        self.stream.synchronize()?;
        Ok(Plan {
            cuda: self,
            pixels,
            bins,
            entries,
            rows,
            columns,
            weights: weights_device,
            weighted: u32::from(weights.is_some()),
            denominator,
            upload_ms: elapsed_ms(start),
        })
    }
}

struct Plan<'a> {
    cuda: &'a Cuda,
    pixels: u32,
    bins: u32,
    entries: u32,
    rows: CudaSlice<u32>,
    columns: CudaSlice<u32>,
    weights: CudaSlice<f32>,
    weighted: u32,
    denominator: CudaSlice<f64>,
    upload_ms: f64,
}

struct Batch<'a> {
    plan: &'a Plan<'a>,
    input: CudaSlice<f32>,
    output: CudaSlice<f64>,
    frames: u32,
    errors: bool,
}

struct Kernel {
    function: CudaFunction,
    layout: Layout,
    errors: bool,
}

impl<'a> Plan<'a> {
    fn batch(&'a self, frames: u32, errors: bool) -> Result<Batch<'a>> {
        if frames == 0 || frames > i32::MAX as u32 / self.bins {
            return Err("CUDA batch exceeds the grid limit or is empty".into());
        }
        let elements = |per_frame: usize, element_bytes: usize| {
            per_frame
                .checked_mul(frames as usize)
                .filter(|&len| len <= isize::MAX as usize / element_bytes)
                .ok_or("CUDA batch allocation exceeds the addressable range")
        };
        let input_len = elements(self.pixels as usize, size_of::<f32>())?;
        let output_len = elements(
            self.bins as usize * (1 + usize::from(errors)),
            size_of::<f64>(),
        )?;
        let stream = &self.cuda.stream;
        Ok(Batch {
            plan: self,
            input: stream.alloc_zeros(input_len)?,
            output: stream.alloc_zeros(output_len)?,
            frames,
            errors,
        })
    }

    fn kernel(&self, mode: Accumulation, layout: Layout, errors: bool) -> Result<Kernel> {
        let suffix = if errors { "iv" } else { "i" };
        Ok(Kernel {
            function: self.cuda.module.load_function(&format!(
                "{}_{}_{suffix}",
                mode.name(),
                layout.name()
            ))?,
            layout,
            errors,
        })
    }
}

impl Batch<'_> {
    fn upload(&mut self, offset: usize, frames: &(impl HostSlice<f32> + ?Sized)) -> Result<()> {
        let pixels = self.plan.pixels as usize;
        let end = offset
            .checked_add(frames.len())
            .filter(|&end| end <= self.input.len());
        if !offset.is_multiple_of(pixels) || !frames.len().is_multiple_of(pixels) {
            return Err("CUDA upload must cover whole frames".into());
        }
        let mut destination = self
            .input
            .slice_mut(offset..end.ok_or("CUDA upload exceeds the batch")?);
        Ok(self
            .plan
            .cuda
            .stream
            .memcpy_htod(frames, &mut destination)?)
    }

    fn run(&mut self, kernel: &Kernel) -> Result<()> {
        if self.errors != kernel.errors {
            return Err("CUDA kernel and output strides differ".into());
        }
        let plan = self.plan;
        let output_per_frame = plan.bins as usize * (1 + usize::from(self.errors));
        let pixels = plan.pixels as usize;
        // Bound the work in each launch on display GPUs with a watchdog.
        for first in (0..self.frames).step_by(256) {
            let end = (first + 256).min(self.frames);
            let input = self
                .input
                .slice(first as usize * pixels..end as usize * pixels);
            let mut output = self
                .output
                .slice_mut(first as usize * output_per_frame..end as usize * output_per_frame);
            let config = LaunchConfig {
                grid_dim: (
                    plan.bins.div_ceil(kernel.layout.bins_per_block()) * (end - first),
                    1,
                    1,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = plan.cuda.stream.launch_builder(&kernel.function);
            launch
                .arg(&plan.rows)
                .arg(&plan.columns)
                .arg(&plan.weights)
                .arg(&plan.denominator)
                .arg(&input)
                .arg(&mut output)
                .arg(&plan.pixels)
                .arg(&plan.bins)
                .arg(&plan.weighted);
            // SAFETY: the views span exactly the launched frames and the
            // kernel carries its layout and the matching variance specialization.
            unsafe { launch.launch(config)? };
        }
        Ok(())
    }

    fn download(&self, output: &mut (impl HostSlice<f64> + ?Sized)) -> Result<()> {
        if output.len() != self.output.len() {
            return Err("host output does not match CUDA batch shape".into());
        }
        Ok(self.plan.cuda.stream.memcpy_dtoh(&self.output, output)?)
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}
