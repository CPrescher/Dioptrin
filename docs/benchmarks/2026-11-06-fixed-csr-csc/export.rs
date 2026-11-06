use dioptrin::{geometry::Geometry, integration::RadialUnit, poni::Poni};
use std::{fs, path::Path};
fn write_f64(p: &Path, values: &[f64]) {
    fs::write(
        p,
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
}
fn main() {
    let a: Vec<_> = std::env::args().collect();
    let size: usize = a[2].parse().unwrap();
    let y: usize = a[3].parse().unwrap();
    let x: usize = a[4].parse().unwrap();
    let bins: usize = a[5].parse().unwrap();
    let out = Path::new(&a[6]);
    fs::create_dir_all(out).unwrap();
    let mut poni = Poni::load(&a[1]).unwrap();
    poni.poni1 -= y as f64 * poni.pixel1;
    poni.poni2 -= x as f64 * poni.pixel2;
    let geo = Geometry::new(size, size, poni);
    let tth = geo.calc_tth_all();
    let azi = geo.calc_azimuth_all();
    let sa = geo.calc_solid_angle();
    let pol = geo.calc_polarization(0.965, &tth, &azi);
    let normalization: Vec<_> = sa.iter().zip(&pol).map(|(a, b)| a * b).collect();
    let radial = RadialUnit::TthDeg.convert_from_tth(&tth, geo.poni.wavelength);
    let lut = dioptrin::lut::build_lut1d_ps_csr(&geo, &radial, bins, RadialUnit::TthDeg);
    fs::write(
        out.join("indptr.u64"),
        lut.row_ptr
            .iter()
            .flat_map(|&v| (v as u64).to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    fs::write(
        out.join("indices.u32"),
        lut.col_idx
            .iter()
            .flat_map(|&v| (u32::try_from(v).unwrap()).to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    write_f64(&out.join("weights.f64"), lut.values.as_ref().unwrap());
    write_f64(&out.join("normalization.f64"), &normalization);
    write_f64(&out.join("radial.f64"), &lut.bins());
}
