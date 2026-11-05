//! Export the public 1D LUT for the matched bslz4 benchmark, without changing the engine.
//! Arguments: PONI OUT_DIR HEIGHT WIDTH BINS [counting|pixel_split].
use dioptrin::geometry::Geometry;
use dioptrin::integration::RadialUnit;
use dioptrin::lut::{build_lut1d_csr, build_lut1d_ps_csr};
use dioptrin::poni::Poni;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

fn write_f64(path: &Path, values: &[f64]) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        return Err("expected PONI OUT_DIR HEIGHT WIDTH BINS [counting|pixel_split]".into());
    }
    let out = Path::new(&args[2]);
    fs::create_dir_all(out)?;
    let height: usize = args[3].parse()?;
    let width: usize = args[4].parse()?;
    let bins: usize = args[5].parse()?;
    let method = args.get(6).map(String::as_str).unwrap_or("pixel_split");
    let poni = Poni::load(&args[1])?;
    assert!(poni.pixel1 > 0.0 && poni.pixel2 > 0.0);
    let geometry = Geometry::new(height, width, poni);
    let tth = geometry.calc_tth_all();
    let radial = RadialUnit::TthDeg.convert_from_tth(&tth, geometry.poni.wavelength);
    let lut = match method {
        "counting" => build_lut1d_csr(&radial, bins),
        "pixel_split" => build_lut1d_ps_csr(&geometry, &radial, bins, RadialUnit::TthDeg),
        _ => return Err("method must be counting or pixel_split".into()),
    };
    let mut pointers = BufWriter::new(File::create(out.join("indptr.u64"))?);
    for &value in &lut.row_ptr {
        pointers.write_all(&(value as u64).to_le_bytes())?;
    }
    pointers.flush()?;
    let mut indices = BufWriter::new(File::create(out.join("indices.u32"))?);
    for &value in &lut.col_idx {
        indices.write_all(&u32::try_from(value)?.to_le_bytes())?;
    }
    indices.flush()?;
    if let Some(weights) = &lut.values {
        write_f64(&out.join("weights.f64"), weights)?;
    }
    write_f64(&out.join("radial.f64"), &lut.bins())?;
    let sa = geometry.calc_solid_angle();
    let azimuthal = geometry.calc_azimuth_all();
    let pol = geometry.calc_polarization(0.965, &tth, &azimuthal);
    let normalization: Vec<f64> = sa.iter().zip(&pol).map(|(s, p)| s * p).collect();
    write_f64(&out.join("normalization.f64"), &normalization)?;
    println!(
        "exported {method}: {height}x{width}, {bins} bins, {} entries",
        lut.col_idx.len()
    );
    Ok(())
}
