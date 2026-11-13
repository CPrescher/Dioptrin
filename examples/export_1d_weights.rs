//! Export a single detector's per-pixel 1D weights for numerical diagnostics.
//! Usage: export_1d_weights PONI ROWS COLS NR R0 R1 OUTPUT METHOD [N_SS]
//! METHOD is counting, pixel_split, pyfai_full or supersampled; R0/R1 are 2θ in
//! radians. OUTPUT is little-endian f64, shaped (ROWS*COLS, NR): row p holds the
//! raw `sum_signal` of an image that is 1 in pixel p and 0 elsewhere. No mask or
//! corrections are applied. Tables that are normalized per bin (counting,
//! pixel_split, supersampled) therefore report w[p,b] / sum_p w[p,b].
use dioptrin::{
    geometry::Geometry,
    integration::{IntegrationMethod, Integrator, RadialUnit},
    poni::Poni,
};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if !(9..=10).contains(&a.len()) {
        return Err("expected PONI ROWS COLS NR R0 R1 OUTPUT METHOD [N_SS]".into());
    }
    let (rows, cols, nr): (usize, usize, usize) = (a[2].parse()?, a[3].parse()?, a[4].parse()?);
    let method = match a[8].as_str() {
        "counting" => IntegrationMethod::Counting,
        "pixel_split" => IntegrationMethod::PixelSplit,
        "pyfai_full" => IntegrationMethod::PyfaiFull,
        "supersampled" => IntegrationMethod::Supersampled {
            n_ss: a.get(9).ok_or("supersampled needs N_SS")?.parse()?,
        },
        _ => return Err("unknown method".into()),
    };
    let ai = Integrator::new(Geometry::new(rows, cols, Poni::load(&a[1])?))
        .with_method(method)
        .with_radial_unit(RadialUnit::TthRad)
        .with_radial_range(a[5].parse()?, a[6].parse()?);
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[7])?);
    let mut image = vec![0.0; rows * cols];
    for p in 0..image.len() {
        image[p] = 1.0;
        for w in ai
            .integrate1d_raw_from_slice(&image, rows, cols, nr)
            .sum_signal
        {
            out.write_all(&w.to_le_bytes())?;
        }
        image[p] = 0.0;
    }
    Ok(())
}
