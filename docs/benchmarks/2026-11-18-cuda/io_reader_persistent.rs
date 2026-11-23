//! Full Eiger scans through one resident context, lookup table, and decoder pool.
use super::{reader::ParallelDecoder, *};
use std::io::BufRead;

#[derive(serde::Deserialize)]
struct ScanRequest {
    source: String,
    output: PathBuf,
    plan_key: String,
    frames: usize,
}

#[test]
#[ignore = "dioptrin-io benchmark; use hdf5_persistent.py"]
fn eiger_hdf5_persistent() -> Result<()> {
    let stdin = std::io::stdin();
    let mut requests = stdin.lock().lines();
    let first = requests.next().ok_or("missing scan request")??;
    let first: ScanRequest = serde_json::from_str(&first)?;
    let geometry: ScanGeometry = serde_json::from_str(&env::var("DIOPTRIN_HDF5_GEOMETRY")?)?;
    let dataset = env::var("DIOPTRIN_HDF5_DATASET")?;
    let frames = env::var("DIOPTRIN_HDF5_FRAMES")?.parse()?;
    let tile = env::var("DIOPTRIN_HDF5_TILE")?.parse::<usize>()?;
    let threads = env::var("DIOPTRIN_HDF5_DECODERS")?.parse::<usize>()?;
    let layout = match env::var("DIOPTRIN_HDF5_LAYOUT")?.as_str() {
        "block" => Layout::Block,
        "warp" => Layout::Warp,
        _ => return Err("unknown CUDA layout".into()),
    };
    let open = Instant::now();
    let api = LibraryApi;
    let source = api.dataset(&first.source, &dataset)?;
    let opened_ms = elapsed_ms(open);
    let [available, rows, columns] = source.dims;
    if frames == 0
        || frames > available
        || !(1..=64).contains(&tile)
        || !(1..=14).contains(&threads)
    {
        return Err("invalid frame, tile, or decoder count".into());
    }
    let work = Work {
        first: 0,
        frames,
        tile: tile.min(frames),
        shape: (rows, columns),
    };

    let setup = Instant::now();
    let mut cpu = geometry.integrator(work.shape, IntegrationMethod::PyfaiFull)?;
    let serve = |mut integrate: Box<
        dyn FnMut(&mut Dataset, &Work) -> Result<(Vec<f64>, Phases)> + '_,
    >| {
        let mut run = |mut source: Dataset,
                       request: &ScanRequest,
                       iteration,
                       open_ms,
                       plan_ms|
         -> Result<()> {
            if source.dims[1..] != [rows, columns]
                || request.frames == 0
                || request.frames > source.dims[0]
            {
                return Err(
                    "scan shape or requested frame count is incompatible with the plan".into(),
                );
            }
            let current = Work {
                frames: request.frames,
                ..work
            };
            let start = Instant::now();
            let (patterns, phases) = integrate(&mut source, &current)?;
            let batch_ms = elapsed_ms(start);
            let writing = Instant::now();
            write_patterns(&request.output, &patterns)?;
            let row = json!({
                "iteration": iteration, "open_ms": open_ms, "prepare_ms": plan_ms, "batch_ms": batch_ms,
                "write_patterns_ms": elapsed_ms(writing), "phases": phases, "pid": std::process::id(),
            });
            let mut stdout = std::io::stdout().lock();
            writeln!(stdout, "DIOPTRIN_HDF5_BATCH {row}")?;
            Ok(stdout.flush()?)
        };
        run(source, &first, 0, opened_ms, elapsed_ms(setup))?;
        for (i, request) in requests.enumerate() {
            let request: ScanRequest = serde_json::from_str(&request?)?;
            if request.plan_key != first.plan_key {
                return Err(
                    "cannot reuse an integration plan with different preparation settings".into(),
                );
            }
            let open = Instant::now();
            let source = api.dataset(&request.source, &dataset)?;
            let opened_ms = elapsed_ms(open);
            run(source, &request, i + 1, opened_ms, 0.)?;
        }
        Ok(())
    };
    if env::var("DIOPTRIN_HDF5_BACKEND")?.as_str() == "cpu_csr" {
        cpu.set_sparse_layout(SparseLayout::Csr);
        let pool = ThreadPoolBuilder::new().num_threads(14).build()?;
        pool.install(|| {
            black_box(integrate_pixels(
                &cpu,
                work.shape,
                &vec![0u32; work.pixels()],
                true,
            ))
        });
        let mut buffers = std::array::from_fn(|_| vec![0u32; work.pixels() * work.tile]);
        serve(Box::new(|source, current| {
            cpu_batch_hdf5(source, current, &cpu, &pool, &mut buffers)
        }))
    } else {
        let cuda = Cuda::open()?.ok_or("CUDA driver, GPU or NVRTC unavailable")?;
        let plan = cuda.prepare_f64(&cpu, work.shape, BINS)?;
        let kernel = plan.kernel_u32(layout, true)?;
        let mut slots = slots(&plan, &work)?;
        let mut decoder = ParallelDecoder::new(threads)?;
        serve(Box::new(|source, current| {
            cuda_batch_hdf5(
                source,
                current,
                &kernel,
                &mut slots,
                |source, first, dst, phases| {
                    let start = Instant::now();
                    decoder.read(source, first, dst)?;
                    phases.read_decode_ms += elapsed_ms(start);
                    Ok(())
                },
            )
        }))
    }
}
