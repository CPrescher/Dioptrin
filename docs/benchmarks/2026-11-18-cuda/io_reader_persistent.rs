//! Full Eiger scans through one resident context, lookup table, and decoder pool.
use super::{reader::ParallelDecoder, *};
use std::io::BufRead;

#[derive(serde::Deserialize)]
struct ScanRequest {
    source: String,
    output: PathBuf,
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
    let cpu = geometry.integrator(work.shape, IntegrationMethod::PyfaiFull)?;
    let cuda = Cuda::open()?.ok_or("CUDA driver, GPU or NVRTC unavailable")?;
    let plan = cuda.prepare_f64(&cpu, work.shape, BINS)?;
    let kernel = plan.kernel_u32(layout, true)?;
    let mut slots = slots(&plan, &work)?;
    let mut decoder = ParallelDecoder::new(threads)?;
    let prepare_ms = elapsed_ms(setup);
    let mut run = |mut source: Dataset,
                   output: PathBuf,
                   iteration,
                   open_ms,
                   plan_ms|
     -> Result<()> {
        let start = Instant::now();
        let (patterns, phases) = cuda_batch_hdf5(
            &mut source,
            &work,
            &kernel,
            &mut slots,
            |source, first, dst, phases| {
                let start = Instant::now();
                decoder.read(source, first, dst)?;
                phases.read_decode_ms += elapsed_ms(start);
                Ok(())
            },
        )?;
        let batch_ms = elapsed_ms(start);
        let writing = Instant::now();
        write_patterns(&output, &patterns)?;
        let row = json!({
            "iteration": iteration, "open_ms": open_ms, "prepare_ms": plan_ms, "batch_ms": batch_ms,
            "write_patterns_ms": elapsed_ms(writing), "phases": phases, "pid": std::process::id(),
        });
        let mut stdout = std::io::stdout().lock();
        writeln!(stdout, "DIOPTRIN_HDF5_BATCH {row}")?;
        Ok(stdout.flush()?)
    };
    run(source, first.output, 0, opened_ms, prepare_ms)?;
    for (i, request) in requests.enumerate() {
        let request: ScanRequest = serde_json::from_str(&request?)?;
        let open = Instant::now();
        let source = api.dataset(&request.source, &dataset)?;
        let opened_ms = elapsed_ms(open);
        if source.dims != [available, rows, columns] {
            return Err("cannot reuse a plan for a different scan shape".into());
        }
        run(source, request.output, i + 1, opened_ms, 0.)?;
    }
    Ok(())
}
