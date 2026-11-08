//! Export a single detector's fractional 2D weights for numerical diagnostics.
//! Usage: inspect_split_weights PONI ROWS COLS NR NA R0 R1 A0 A1 UNIT OUTPUT [fractional|pyfai_histogram|supersampled] [N_SS]
//! OUTPUT is little-endian f64, shaped (ROWS*COLS, NA, NR); no masks/corrections.
use dioptrin::{
    geometry::Geometry,
    integration::{IntegrationGrid2d, IntegrationMethod, Integrator, PixelSplitMode, RadialUnit},
    poni::Poni,
};
use std::io::Write;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if !(12..=14).contains(&a.len()) {
        return Err("expected PONI ROWS COLS NR NA R0 R1 A0 A1 UNIT OUTPUT [fractional|pyfai_histogram|supersampled] [N_SS]".into());
    }
    let rows = a[2].parse()?;
    let cols = a[3].parse()?;
    let nr = a[4].parse()?;
    let na = a[5].parse()?;
    let unit = match a[10].as_str() {
        "2th_rad" => RadialUnit::TthRad,
        "2th_deg" => RadialUnit::TthDeg,
        "q_A^-1" => RadialUnit::QAngstrom,
        "q_nm^-1" => RadialUnit::QNm,
        _ => return Err("unknown unit".into()),
    };
    let mode = a.get(12).map_or("fractional", String::as_str);
    let method = match mode {
        "fractional" | "pyfai_histogram" => IntegrationMethod::PixelSplit,
        "supersampled" => IntegrationMethod::Supersampled {
            n_ss: a.get(13).map_or(Ok(128), |s| s.parse::<usize>())?,
        },
        _ => return Err("unknown splitting mode".into()),
    };
    let ai = Integrator::new(Geometry::new(rows, cols, Poni::load(&a[1])?))
        .with_method(method)
        .with_radial_unit(unit);
    let grid = IntegrationGrid2d::new(
        nr,
        na,
        (a[6].parse()?, a[7].parse()?),
        (a[8].parse()?, a[9].parse()?),
    )?;
    let grid = if a.get(12).is_some_and(|s| s == "pyfai_histogram") {
        grid.with_pixel_split_mode(PixelSplitMode::PyfaiHistogram)
    } else {
        grid
    };
    let mut output = std::io::BufWriter::new(std::fs::File::create(&a[11])?);
    let mut image = vec![0.0; rows * cols];
    for pixel in 0..image.len() {
        image[pixel] = 1.0;
        let raw = ai.integrate2d_raw_on_grid_from_slice(&image, rows, cols, &grid, false)?;
        for weight in raw.sum_signal {
            output.write_all(&weight.to_le_bytes())?;
        }
        image[pixel] = 0.0;
    }
    Ok(())
}
