//! Fractional pixel contributions on explicit shared edges. Legacy builders
//! retain their historical row normalization and implicit detector-local grids.
use super::Lut2d;
use super::pixel_split::{clip_polygon_rect_scratch, polygon_area};
use crate::geometry::Geometry;
use crate::integration::{
    IntegrationGrid2d, LutSampling, PixelSplitMode, RadialConverter, RadialUnit,
};
use rayon::prelude::*;
use std::f64::consts::TAU;

pub(crate) fn build(
    geometry: &Geometry,
    radial: &[f64],
    azimuthal: &[f64],
    grid: &IntegrationGrid2d,
    sampling: LutSampling,
    unit: RadialUnit,
) -> Lut2d {
    let corners = if matches!(sampling, LutSampling::PixelSplit(_)) {
        let (tth, azimuthal) = geometry.angular_corners();
        Some((
            unit.convert_from_tth(&tth, geometry.poni.wavelength),
            azimuthal,
        ))
    } else {
        None
    };
    let corner_sign = match (&corners, sampling) {
        (Some((r, a)), LutSampling::PixelSplit(PixelSplitMode::PyfaiHistogram)) => {
            super::pyfai_1d::corner_sign(
                geometry,
                (0..radial.len()).into_par_iter().map(|pixel| {
                    let [i00, i01, i11, i10] = geometry.corner_indices(pixel);
                    [i00, i10, i11, i01].map(|i| (r[i], a[i]))
                }),
            )
        }
        _ => 1.0,
    };
    let transform = geometry.subpixel_transform();
    let converter = (!matches!(sampling, LutSampling::Counting))
        .then(|| RadialConverter::new(unit, geometry.poni.wavelength));
    let pixels: Vec<Vec<(usize, f64)>> = (0..radial.len())
        .into_par_iter()
        .map(|pixel| {
            let mut entries = Vec::new();
            match sampling {
                LutSampling::Counting => {
                    if let Some(bin) = grid.bin(radial[pixel], azimuthal[pixel]) {
                        entries.push((bin, 1.0));
                    }
                }
                LutSampling::Supersampled { n_ss } => {
                    entries = sample_pixel(
                        geometry,
                        &transform,
                        pixel,
                        grid,
                        n_ss,
                        &[(0.5, 0.5)],
                        converter.as_ref().unwrap(),
                    );
                }
                LutSampling::PixelSplit(split_mode) => {
                    debug_assert_eq!(split_mode, grid.pixel_split_mode);
                    let (r, a) = corners.as_ref().unwrap();
                    let indices = geometry.corner_indices(pixel);
                    // A pixel enclosing/touching the polar origin has no single
                    // simple polar quad. Use detector-plane samples, coalesced by
                    // pixel/bin before the variance kernel squares each weight.
                    let mut axis_on_edge = false;
                    let winding: f64 = (0..4)
                        .map(|i| {
                            let delta = a[indices[(i + 1) % 4]] - a[indices[i]];
                            let delta = (delta + std::f64::consts::PI).rem_euclid(TAU)
                                - std::f64::consts::PI;
                            axis_on_edge |=
                                delta.abs() >= std::f64::consts::PI - 64.0 * f64::EPSILON;
                            delta
                        })
                        .sum();
                    if winding.abs() > std::f64::consts::PI
                        || axis_on_edge
                        || indices.iter().any(|&i| r[i] == 0.0)
                    {
                        return sample_pixel(
                            geometry,
                            &transform,
                            pixel,
                            grid,
                            128,
                            &[(0.25, 0.5), (0.5, 0.25), (0.75, 0.5), (0.5, 0.75)],
                            converter.as_ref().unwrap(),
                        );
                    }
                    if split_mode == PixelSplitMode::PyfaiHistogram {
                        // Reverse to pyFAI's corner order, preserving raw angles.
                        let [i00, i01, i11, i10] = indices;
                        let poly = [i00, i10, i11, i01].map(|i| (r[i], a[i]));
                        if let Some(entries) = super::pyfai_split::split(poly, grid, corner_sign) {
                            return entries;
                        }
                    }
                    let (a0, _) = grid.azimuthal_range();
                    let center = a0 + (azimuthal[pixel] - a0).rem_euclid(TAU);
                    let poly = indices.map(|i| {
                        (
                            r[i],
                            center + (a[i] - center + std::f64::consts::PI).rem_euclid(TAU)
                                - std::f64::consts::PI,
                        )
                    });
                    let area = polygon_area(&poly).abs();
                    if area > 0.0 {
                        split_polygon(&poly, area, grid, &mut entries);
                    } else if let Some(bin) = grid.bin(radial[pixel], azimuthal[pixel]) {
                        entries.push((bin, 1.0));
                    }
                }
            }
            entries
        })
        .collect();

    let rows = grid.n_radial * grid.n_azimuthal;
    let mut row_ptr = vec![0; rows + 1];
    for entries in &pixels {
        for &(bin, _) in entries {
            row_ptr[bin + 1] += 1;
        }
    }
    for i in 0..rows {
        row_ptr[i + 1] += row_ptr[i];
    }
    let mut next = row_ptr[..rows].to_vec();
    let mut col_idx = vec![0; row_ptr[rows]];
    let mut values = vec![0.0; row_ptr[rows]];
    for (pixel, entries) in pixels.into_iter().enumerate() {
        for (bin, weight) in entries {
            let slot = next[bin];
            next[bin] += 1;
            col_idx[slot] = pixel;
            values[slot] = weight;
        }
    }
    Lut2d {
        row_ptr,
        col_idx,
        values: Some(values),
        n_tth: grid.n_radial,
        n_azimuthal: grid.n_azimuthal,
    }
}

fn split_polygon(
    poly: &[(f64, f64); 4],
    area: f64,
    grid: &IntegrationGrid2d,
    entries: &mut Vec<(usize, f64)>,
) {
    let (r0, _) = grid.radial_range();
    let (a0, _) = grid.azimuthal_range();
    let dr = grid.radial_step();
    let da = grid.azimuthal_step();
    let r_min = poly.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let r_max = poly.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let r_start = (((r_min - r0) / dr).floor() as isize).max(0);
    let r_end = (((r_max - r0) / dr).floor() as isize).min(grid.n_radial as isize - 1);
    let mut scratch1 = Vec::with_capacity(8);
    let mut scratch2 = Vec::with_capacity(8);
    // Periodic copies handle the ±π seam even for a clipped azimuthal sector.
    for shift in [-TAU, 0.0, TAU] {
        let poly = poly.map(|(r, a)| (r, a + shift));
        let a_min = poly.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let a_max = poly.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let a_start = (((a_min - a0) / da).floor() as isize).max(0);
        let a_end = (((a_max - a0) / da).floor() as isize).min(grid.n_azimuthal as isize - 1);
        for ar in a_start..=a_end {
            for rr in r_start..=r_end {
                let r = r0 + rr as f64 * dr;
                let a = a0 + ar as f64 * da;
                let clipped = clip_polygon_rect_scratch(
                    &poly,
                    r,
                    r + dr,
                    a,
                    a + da,
                    &mut scratch1,
                    &mut scratch2,
                );
                let weight = polygon_area(clipped).abs() / area;
                if weight > 0.0 {
                    let bin = ar as usize * grid.n_radial + rr as usize;
                    if let Some((_, w)) = entries.iter_mut().find(|(b, _)| *b == bin) {
                        *w += weight;
                    } else {
                        entries.push((bin, weight));
                    }
                }
            }
        }
    }
}

fn sample_pixel(
    geometry: &Geometry,
    transform: &crate::geometry::SubpixelTransform,
    pixel: usize,
    grid: &IntegrationGrid2d,
    n_ss: usize,
    offsets: &[(f64, f64)],
    converter: &crate::integration::RadialConverter,
) -> Vec<(usize, f64)> {
    let row = pixel / geometry.dim2();
    let col = pixel % geometry.dim2();
    let weight = 1.0 / ((n_ss * n_ss) as f64 * offsets.len() as f64);
    let mut entries = Vec::new();
    for i in 0..n_ss {
        for j in 0..n_ss {
            for &(off1, off2) in offsets {
                let d1 = row as f64 + (i as f64 + off1) / n_ss as f64;
                let d2 = ((col * n_ss + j) as f64 + off2) / n_ss as f64;
                let (tth, azimuthal) = transform.tth_and_azimuth(d1, d2);
                if let Some(bin) = grid.bin(converter.convert(tth), azimuthal) {
                    if let Some((_, w)) = entries.iter_mut().find(|(b, _)| *b == bin) {
                        *w += weight;
                    } else {
                        entries.push((bin, weight));
                    }
                }
            }
        }
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::{Detector, DetectorTransform, PixelMap};
    use crate::integration::{IntegrationMethod, Integrator};
    use crate::poni::Poni;
    use std::sync::Arc;

    fn test_geometry(dim1: usize, dim2: usize, detector: Detector) -> Geometry {
        let mut poni = Poni::new();
        poni.distance = 0.2;
        poni.poni1 = 0.0;
        poni.poni2 = 0.0;
        poni.pixel1 = detector.pixel1;
        poni.pixel2 = detector.pixel2;
        poni.wavelength = 1e-10;
        Geometry::new(dim1, dim2, poni)
            .with_detector(Arc::new(detector))
            .unwrap()
    }

    fn gapped_separable_detector() -> Detector {
        let p = 1e-3;
        let gap = 10.0 * p;
        let slow: Vec<[[f64; 3]; 2]> = (0..8)
            .map(|r| {
                let shift = if r >= 4 { gap } else { 0.0 };
                let lo = (r as f64 + 2.0) * p + shift;
                let hi = lo + p;
                [[0.0, lo, 0.0], [0.0, hi, 0.0]]
            })
            .collect();
        let fast: Vec<[[f64; 3]; 2]> = (0..8)
            .map(|c| {
                let lo = (c as f64 + 2.0) * p;
                let hi = lo + p;
                [[0.0, 0.0, lo], [0.0, 0.0, hi]]
            })
            .collect();
        let mut det = Detector::custom(p, p);
        det.max_shape = Some([8, 8]);
        det.map = PixelMap::Separable { slow, fast };
        det
    }

    fn corners_detector() -> Detector {
        let p = 1e-3;
        let mut values = Vec::with_capacity(64);
        for r in 0..8 {
            for c in 0..8 {
                let y0 = (r as f64 + 3.0) * p;
                let y1 = y0 + p;
                let x0 = (c as f64 + 3.0) * p;
                let x1 = x0 + p;
                // Slight parabolic curvature in z so pixels are non-coplanar.
                let z = |y: f64, x: f64| 0.5 * (y * y + x * x);
                values.push([
                    [z(y0, x0), y0, x0],
                    [z(y1, x0), y1, x0],
                    [z(y1, x1), y1, x1],
                    [z(y0, x1), y0, x1],
                ]);
            }
        }
        let mut det = Detector::custom(p, p);
        det.max_shape = Some([8, 8]);
        det.map = PixelMap::Corners {
            shape: [8, 8],
            values: Arc::new(values),
        };
        det
    }

    fn assert_2d_matches_1d_totals(detector: Detector, method: IntegrationMethod) {
        assert!(!detector.is_contiguous());
        let geometry = test_geometry(8, 8, detector);
        let integrator = Integrator::new(geometry)
            .with_method(method)
            .with_radial_unit(RadialUnit::TthDeg);
        let frame: Vec<f64> = (0..64).map(|i| 10.0 + (i as f64) * 0.5).collect();
        let raw1d = integrator.integrate1d_raw_from_slice(&frame, 8, 8, 24);
        let r_min = raw1d.bins[0] - 0.5 * (raw1d.bins[1] - raw1d.bins[0]);
        let r_max = raw1d.bins[23] + 0.5 * (raw1d.bins[1] - raw1d.bins[0]);
        let mut grid = IntegrationGrid2d::new(
            24,
            12,
            (r_min, r_max),
            (-std::f64::consts::PI, std::f64::consts::PI),
        )
        .unwrap();
        if method == IntegrationMethod::PyfaiFull {
            grid = grid.with_pixel_split_mode(PixelSplitMode::PyfaiHistogram);
        }
        let raw2d = integrator
            .integrate2d_raw_on_grid_from_slice(&frame, 8, 8, &grid, false)
            .unwrap();

        let sum_sig_1d: f64 = raw1d.sum_signal.iter().sum();
        let sum_sig_2d: f64 = raw2d.sum_signal.iter().sum();
        let sum_norm_1d: f64 = raw1d.sum_norm.iter().sum();
        let sum_norm_2d: f64 = raw2d.sum_norm.iter().sum();
        let sum_count_2d: f64 = raw2d.count.iter().sum();
        let expected_sig: f64 = frame.iter().sum();

        assert!(
            (sum_count_2d - 64.0).abs() < 1e-6,
            "pixel count conservation failed for {method:?}: 2d={sum_count_2d}"
        );
        assert!(
            (sum_norm_2d - sum_norm_1d).abs() < 1e-6,
            "norm conservation failed for {method:?}: 2d={sum_norm_2d}, 1d={sum_norm_1d}"
        );
        assert!(
            (sum_sig_2d - expected_sig).abs() < 1e-4,
            "signal conservation failed for {method:?}: 2d={sum_sig_2d}, expected={expected_sig}"
        );
        assert!(
            (sum_sig_2d - sum_sig_1d).abs() < 1e-4,
            "2d vs 1d total signal mismatch for {method:?}: 2d={sum_sig_2d}, 1d={sum_sig_1d}"
        );
    }

    #[test]
    fn shared_grid_gapped_separable_conserves_and_leaves_gap_empty() {
        for method in [IntegrationMethod::PixelSplit, IntegrationMethod::PyfaiFull] {
            assert_2d_matches_1d_totals(gapped_separable_detector(), method);
        }

        // Verify no quad straddles the physical gap between rows 3 and 4 when
        // cols are aligned near x=0 so radial distance cleanly separates modules.
        let p = 1e-3;
        let slow: Vec<[[f64; 3]; 2]> = (0..4)
            .map(|r| {
                let shift = if r >= 2 { 20.0 * p } else { 0.0 };
                let lo = (r as f64 + 1.0) * p + shift;
                [[0.0, lo, 0.0], [0.0, lo + p, 0.0]]
            })
            .collect();
        let fast = vec![[[0.0, 0.0, 0.0], [0.0, 0.0, p]]];
        let mut det = Detector::custom(p, p);
        det.max_shape = Some([4, 1]);
        det.map = PixelMap::Separable { slow, fast };
        assert!(!det.is_contiguous());

        let integrator = Integrator::new(test_geometry(4, 1, det))
            .with_method(IntegrationMethod::PixelSplit)
            .with_radial_unit(RadialUnit::TthRad);
        // Module 0 covers y in [1mm, 3mm] (tth <= ~0.016 rad);
        // Module 1 covers y in [23mm, 25mm] (tth >= ~0.114 rad).
        // Bins in [0.04, 0.08] rad lie strictly inside the physical gap.
        let grid = IntegrationGrid2d::new(
            14,
            4,
            (0.0, 0.14),
            (-std::f64::consts::PI, std::f64::consts::PI),
        )
        .unwrap();
        let raw = integrator
            .integrate2d_raw_on_grid_from_slice(&[1.0; 4], 4, 1, &grid, false)
            .unwrap();
        let sum_norm = &raw.sum_norm;
        let gap_norm: f64 = (0..4)
            .flat_map(|a| (4..8).map(move |r| sum_norm[a * 14 + r]))
            .sum();
        assert_eq!(gap_norm, 0.0, "gap bins must receive zero pixel weight");
        let total_count: f64 = raw.count.iter().sum();
        assert!((total_count - 4.0).abs() < 1e-9);
    }

    #[test]
    fn shared_grid_corners_detector_conserves_intensity() {
        for method in [IntegrationMethod::PixelSplit, IntegrationMethod::PyfaiFull] {
            assert_2d_matches_1d_totals(corners_detector(), method);
        }
    }

    #[test]
    fn shared_grid_transformed_detectors_conserve_intensity() {
        for op in [DetectorTransform::Right, DetectorTransform::Horizontal] {
            for method in [IntegrationMethod::PixelSplit, IntegrationMethod::PyfaiFull] {
                let det = gapped_separable_detector().transformed(op).unwrap();
                assert_2d_matches_1d_totals(det, method);
            }
        }
    }
}
