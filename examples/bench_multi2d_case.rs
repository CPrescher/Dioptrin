//! Native compute-only companion to python_examples/bench_multi2d_startup.py.
//! Arguments: INPUT_DIR LABEL REPEATS THREADS. Inputs are predecoded little-endian
//! f64 frames plus PONI/mask files prepared by the Python driver.
use dioptrin::{
    geometry::Geometry,
    integration::{IntegrationMethod, Integrator, PixelSplitMode, RadialUnit},
    mask::Mask,
    multi::{Multi2dOptions, MultiIntegrator, MultiNormalization},
    poni::Poni,
};
use std::{io::Write, path::Path, time::Instant};
fn write_values(path: &Path, values: &[f64]) -> std::io::Result<()> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("expected INPUT_DIR LABEL REPEATS THREADS".into());
    }
    let dir = Path::new(&args[1]);
    let label = &args[2];
    let repeats: usize = args[3].parse()?;
    let threads = args[4].parse()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let text = std::fs::read_to_string(dir.join("config.txt"))?;
    let mut words = text.split_whitespace();
    let n_det: usize = words.next().ok_or("detectors")?.parse()?;
    let n_frames: usize = words.next().ok_or("frames")?.parse()?;
    let nr = words.next().ok_or("radial bins")?.parse()?;
    let na = words.next().ok_or("azimuthal bins")?.parse()?;
    let lo = words.next().ok_or("radial min")?.parse()?;
    let hi = words.next().ok_or("radial max")?.parse()?;
    let mut geometries = Vec::new();
    let mut masks = Vec::new();
    let mut shapes = Vec::new();
    let mut monitors = Vec::new();
    for d in 0..n_det {
        let rows = words.next().ok_or("rows")?.parse()?;
        let cols = words.next().ok_or("cols")?.parse()?;
        monitors.push(words.next().ok_or("monitor")?.parse()?);
        shapes.push((rows, cols));
        geometries.push(Geometry::new(
            rows,
            cols,
            Poni::load(
                dir.join(format!("det{d}.poni"))
                    .to_str()
                    .ok_or("PONI path")?,
            )?,
        ));
        masks.push(std::fs::read(dir.join(format!("det{d}.mask")))?);
    }
    let mut frames = Vec::new();
    for f in 0..n_frames {
        let mut detectors = Vec::new();
        for d in 0..n_det {
            let bytes = std::fs::read(dir.join(format!("frame{f}-det{d}.f64")))?;
            if bytes.len() % 8 != 0 {
                return Err("invalid float buffer length".into());
            }
            detectors.push(
                bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|v| f64::from_le_bytes(*v))
                    .collect::<Vec<_>>(),
            );
        }
        frames.push(detectors);
    }
    let method = match label.as_str() {
        "polygon_fractional" | "polygon_pyfai" => IntegrationMethod::PixelSplit,
        "subpixels21" => IntegrationMethod::Supersampled { n_ss: 21 },
        "subpixels41" => IntegrationMethod::Supersampled { n_ss: 41 },
        _ => return Err("unknown method label".into()),
    };
    let start = Instant::now();
    let integrators = geometries
        .into_iter()
        .zip(masks)
        .map(|(g, m)| {
            let rows = g.dim1;
            let cols = g.dim2;
            Ok(Integrator::new(g)
                .with_method(method)
                .with_radial_unit(RadialUnit::QAngstrom)
                .with_radial_range(lo, hi)
                .with_polarization_factor(0.6)
                .with_mask(Mask::from_vec(m, rows, cols)?))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let multi =
        MultiIntegrator::try_new(integrators)?.with_pixel_split_mode(if label == "polygon_pyfai" {
            PixelSplitMode::PyfaiHistogram
        } else {
            PixelSplitMode::Fractional
        });
    let construct_ms = start.elapsed().as_secs_f64() * 1000.0;
    let options = Multi2dOptions {
        errors: true,
        normalization: MultiNormalization::Absolute,
        normalization_factors: Some(monitors),
        ..Default::default()
    };
    let integrate = |f: usize| {
        let slices: Vec<_> = frames[f]
            .iter()
            .zip(&shapes)
            .map(|(data, &(rows, cols))| (data.as_slice(), rows, cols))
            .collect();
        multi.integrate2d_from_slices(&slices, nr, na, &options)
    };
    let first_start = Instant::now();
    let first = integrate(0)?;
    let first_ms = first_start.elapsed().as_secs_f64() * 1000.0;
    let mut samples = Vec::with_capacity(repeats);
    for i in 0..repeats {
        let t = Instant::now();
        let result = std::hint::black_box(integrate((i + 1) % n_frames)?);
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
        drop(result);
    }
    write_values(
        &dir.join(format!("{label}-intensity.f64")),
        &first.intensity,
    )?;
    write_values(
        &dir.join(format!("{label}-errors.f64")),
        first.errors.as_ref().ok_or("missing errors")?,
    )?;
    write_values(&dir.join(format!("{label}-count.f64")), &first.raw.count)?;
    println!(
        "{{\"construct_ms\":{construct_ms},\"first_integration_ms\":{first_ms},\"first_frame_ms\":{},\"samples_ms\":{samples:?}}}",
        construct_ms + first_ms
    );
    Ok(())
}
