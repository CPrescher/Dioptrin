use dioptrin::geometry::Geometry;
use dioptrin::integration::{IntegrationMethod, Integrator, RadialUnit};
use dioptrin::mask::Mask;
use dioptrin::poni::Poni;
use memmap2::MmapOptions;
use rayon::prelude::*;
use std::fs::File;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads: usize = args[1].parse().unwrap();
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    let root = "/tmp/dioptrin-bridge-investigation-20261105";
    let h = 3262;
    let w = 3108;
    let file = File::open(format!("{root}/frames.f64")).unwrap();
    // The prepared input file is immutable throughout this measurement. Its
    // mapping is page aligned, has whole f64 elements and outlives every slice.
    let mapping = unsafe { MmapOptions::new().map(&file).unwrap() };
    assert!(cfg!(target_endian = "little"));
    assert_eq!(mapping.as_ptr().align_offset(std::mem::align_of::<f64>()), 0);
    assert_eq!(mapping.len() % 8, 0);
    let data = unsafe { std::slice::from_raw_parts(mapping.as_ptr().cast::<f64>(), mapping.len()/8) };
    let frames: Vec<&[f64]> = data.chunks_exact(h*w).collect();
    assert_eq!(frames.len(), 16);
    let mask = std::fs::read(format!("{root}/mask.u8")).unwrap();
    let poni = Poni::load("/Users/clemens/Programming/Dioptrin/docs/benchmarks/2026-11-05-bslz4/calibration.poni").unwrap();
    let ai = Integrator::new(Geometry::new(h,w,poni))
        .with_method(IntegrationMethod::PixelSplit)
        .with_radial_unit(RadialUnit::TthDeg)
        .with_polarization_factor(0.965)
        .with_mask(Mask::from_vec(mask,h,w).unwrap());
    ai.integrate1d_from_slice(frames[0],h,w,2000);
    let run = || frames.par_iter().map(|f| ai.integrate1d_from_slice(f,h,w,2000)).collect::<Vec<_>>();
    for _ in 0..3 { black_box(run()); }
    let reference = run();
    let mut output = Vec::new();
    for (_,intensity) in &reference {
        for v in intensity { output.extend_from_slice(&v.to_le_bytes()); }
    }
    std::fs::write(format!("{root}/native-reference-w{threads}.f64"), output).unwrap();
    for rep in 0..5 {
        let start=Instant::now();
        let mut n=0;
        loop {
            black_box(run());
            n+=1;
            if start.elapsed().as_secs_f64() >= 1.0 { break; }
        }
        let seconds=start.elapsed().as_secs_f64();
        println!("{{\"workflow\":\"native_rust\",\"threads\":{threads},\"repeat\":{rep},\"loops\":{n},\"seconds\":{seconds},\"fps\":{}}}",n as f64*frames.len() as f64/seconds);
    }
}
