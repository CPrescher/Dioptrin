//! Native pyFAI-compatible full splitting for 1D patterns and 2D cakes.
//! The 1D kernel follows pyFAI fullSplit1D_engine (radial signed-line
//! integration). 2D uses Dioptrin's existing shared-grid compatibility kernel.
//! Singular beam pixels use detector-plane sampling; negative numerical
//! fractions are discarded before normalization, as in the 2D safety policy.
//!
//! Algorithm adapted from pyFAI/ext/{splitPixel.pyx,regrid_common.pxi} (MIT):
//! Copyright (C) 2012-2024 European Synchrotron Radiation Facility, France.
//! Principal author: Jérôme Kieffer (Jerome.Kieffer@ESRF.eu).
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in
//! all copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//! THE SOFTWARE.
use crate::{
    geometry::Geometry,
    integration::{IntegrationGrid2d, RadialUnit},
    lut::Lut1d,
};
use std::f64::consts::{PI, TAU};

pub fn pyfai_radial_range(g: &Geometry, unit: RadialUnit) -> [f64; 2] {
    let (tth, _) = g.angular_corners();
    let radial = unit.convert_from_tth(&tth, g.poni.wavelength);
    let lo = radial
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .max(0.0);
    let hi = radial.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    [lo, hi * (1.0 + f32::EPSILON as f64)]
}

/// Build fractional 1D weights following pyFAI's full histogram convention.
/// Radial edges use `unit`; azimuthal edges are radians on an increasing interval
/// of at most one full turn. None selects automatic corner bounds/full azimuth.
/// Range-clipped pixels are renormalized. Singular beam pixels use detector-plane
/// fractions; negative numerical fractions are rejected (approximate compatibility).
pub fn build_lut1d_pyfai_csr(
    g: &Geometry,
    unit: RadialUnit,
    bins: usize,
    azimuth: Option<[f64; 2]>,
    radial: Option<[f64; 2]>,
) -> Result<Lut1d, String> {
    let radial = radial.unwrap_or_else(|| pyfai_radial_range(g, unit));
    let angles = azimuth.unwrap_or([-PI, PI]);
    let grid = IntegrationGrid2d::new(bins, 1, (radial[0], radial[1]), (angles[0], angles[1]))?;
    let (tth, azi) = g.angular_corners();
    let r = unit.convert_from_tth(&tth, g.poni.wavelength);
    let mut buckets = vec![Vec::<(usize, f64)>::new(); bins];
    let mut entries = 0usize;
    for pixel in 0..g.size() {
        let [a, b, c, d] = g.corner_indices(pixel);
        let poly = [a, d, c, b].map(|i| (r[i], azi[i]));
        let contributions = if singular(&poly) {
            sample(g, pixel, unit, &grid)
        } else {
            split1d(poly, &grid, azimuth.is_some())
        };
        entries += contributions.len();
        if entries > 64 * 1024 * 1024 {
            return Err("pyFAI-compatible splitting exceeds the 64-million-entry memory bound; reduce binning.".into());
        }
        for (bin, weight) in contributions {
            buckets[bin].push((pixel, weight));
        }
    }
    let mut row_ptr = vec![0];
    let mut col_idx = Vec::with_capacity(entries);
    let mut weights = Vec::with_capacity(entries);
    for bucket in buckets {
        for (pixel, weight) in bucket {
            col_idx.push(pixel);
            weights.push(weight);
        }
        row_ptr.push(col_idx.len());
    }
    Ok(Lut1d {
        row_ptr,
        col_idx,
        values: Some(weights),
        hist_min: radial[0],
        hist_step: (radial[1] - radial[0]) / bins as f64,
    })
}

fn singular(poly: &[(f64, f64); 4]) -> bool {
    let mut winding = 0.0;
    for i in 0..4 {
        if poly[i].0 == 0.0 {
            return true;
        }
        let delta = (poly[(i + 1) % 4].1 - poly[i].1 + PI).rem_euclid(TAU) - PI;
        if delta.abs() >= PI - 64.0 * f64::EPSILON {
            return true;
        }
        winding += delta;
    }
    winding.abs() > PI
}

// Match the upstream safety convention: singular beam pixels use nonnegative
// detector-plane fractions. These exceptional weights are not bitwise pyFAI.
fn sample(
    g: &Geometry,
    pixel: usize,
    unit: RadialUnit,
    grid: &IntegrationGrid2d,
) -> Vec<(usize, f64)> {
    const N: usize = 128;
    let offsets = [(0.25, 0.5), (0.5, 0.25), (0.75, 0.5), (0.5, 0.75)];
    let count = N * N * offsets.len();
    let mut rows = Vec::with_capacity(count);
    let mut cols = Vec::with_capacity(count);
    for row in 0..N {
        for col in 0..N {
            for (dy, dx) in offsets {
                rows.push((pixel / g.dim2) as f64 + (row as f64 + dy) / N as f64);
                cols.push((pixel % g.dim2) as f64 + (col as f64 + dx) / N as f64);
            }
        }
    }
    let (tth, azimuth) = g.calc_tth_and_azimuth(&rows, &cols);
    let radial = unit.convert_from_tth(&tth, g.poni.wavelength);
    let (r0, r1) = grid.radial_range();
    let (a0, a1) = grid.azimuthal_range();
    let (na, nr) = grid.shape();
    let mut hits = std::collections::BTreeMap::<usize, usize>::new();
    for (r, a) in radial.into_iter().zip(azimuth) {
        let a = a0 + (a - a0).rem_euclid(TAU);
        if r < r0 || r >= r1 || a >= a1 {
            continue;
        }
        let ir = ((r - r0) / (r1 - r0) * nr as f64) as usize;
        let ia = ((a - a0) / (a1 - a0) * na as f64) as usize;
        if ir < nr && ia < na {
            *hits.entry(ia * nr + ir).or_default() += 1;
        }
    }
    hits.into_iter()
        .map(|(bin, hits)| (bin, hits as f64 / count as f64))
        .collect()
}

// fullSplit1D_engine integrates signed line areas along the radial axis, then
// renormalizes each included pixel. Its sector selection includes a pixel if
// its azimuthal extent intersects the sector (no partial-azimuth attenuation).
fn split1d(mut poly: [(f64, f64); 4], grid: &IntegrationGrid2d, sector: bool) -> Vec<(usize, f64)> {
    let (a0, a1) = grid.azimuthal_range();
    let offset = if a0 >= -PI && a1 <= PI {
        0.0
    } else if a0 >= 0.0 && a1 <= TAU {
        PI
    } else {
        a0 + PI
    };
    for p in &mut poly {
        p.1 = (p.1 - offset + PI).rem_euclid(TAU) - PI;
    }
    let [a, b, c, d] = poly;
    let area = 0.5 * ((c.0 - a.0) * (d.1 - b.1) - (c.1 - a.1) * (d.0 - b.0));
    if area > 0.0 {
        for p in &mut poly {
            if p.1 < 0.0 {
                p.1 += TAU;
            }
        }
        if poly.iter().map(|p| p.1).sum::<f64>() * 0.25 > PI {
            for p in &mut poly {
                p.1 -= TAU;
            }
        }
    }
    for p in &mut poly {
        p.1 += offset;
    }
    if sector && (poly.iter().all(|p| p.1 < a0) || poly.iter().all(|p| p.1 > a1)) {
        return vec![];
    }
    let (r0, r1) = grid.radial_range();
    let bins = grid.shape().1;
    let dr = (r1 - r0) / bins as f64;
    for p in &mut poly {
        p.0 = (p.0 - r0) / dr;
    }
    let lo = poly.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let hi = poly.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    if hi < 0.0 || lo >= bins as f64 {
        return vec![];
    }
    let first = lo.floor().max(0.0) as usize;
    let end = (hi.floor() as isize + 1).max(0).min(bins as isize) as usize;
    if lo.floor() == hi.floor() {
        return vec![(first, 1.0)];
    }
    let mut buffer = vec![0.0_f32; end - first];
    for i in 0..4 {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % 4];
        if x0 == x1 {
            continue;
        }
        let slope = (y1 - y0) / (x1 - x0);
        let intercept = y0 - slope * x0;
        let sign = if x1 > x0 { 1.0 } else { -1.0 };
        for (j, value) in buffer.iter_mut().enumerate() {
            let left = x0.min(x1).max((first + j) as f64);
            let right = x0.max(x1).min((first + j + 1) as f64);
            if right > left {
                *value +=
                    (sign * (right - left) * (0.5 * slope * (right + left) + intercept)) as f32;
            }
        }
    }
    let sum = buffer.iter().map(|v| *v as f64).sum::<f64>();
    if !sum.is_finite() || sum == 0.0 {
        return vec![];
    }
    // Normal cases have consistently signed areas. Reject numerical negative
    // fractions instead of reproducing pathological negative photon weights.
    let mut weights: Vec<_> = buffer
        .into_iter()
        .enumerate()
        .filter_map(|(j, v)| {
            let w = v as f64 / sum;
            (w.is_finite() && w > 0.0).then_some((first + j, w))
        })
        .collect();
    let total: f64 = weights.iter().map(|(_, w)| w).sum();
    for (_, w) in &mut weights {
        *w /= total;
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radial_line_weights_match_independent_pyfai_golden() {
        // pyFAI 2026.9.0 fullSplit1D_engine, one unit-count pixel, variance=1,
        // explicit [0,2] effective radial edges and four bins.
        let grid = IntegrationGrid2d::new(4, 1, (0., 2.), (-PI, PI)).unwrap();
        let weights = split1d(
            [(0.1, 0.2), (1.2, 0.3), (1.8, 1.5), (0.4, 1.7)],
            &grid,
            false,
        );
        let expected = [
            0.21462747684427036,
            0.4066985688044358,
            0.32228298342690204,
            0.05639097092439181,
        ];
        assert_eq!(weights.len(), expected.len());
        for (i, ((bin, w), expected)) in weights.into_iter().zip(expected).enumerate() {
            assert_eq!(bin, i);
            assert!((w - expected).abs() < 1e-7, "{w} != {expected}");
        }
    }

    #[test]
    fn partial_radial_pixels_renormalize_and_disjoint_sectors_exclude() {
        let grid = IntegrationGrid2d::new(4, 1, (0., 2.), (0.2, 0.4)).unwrap();
        let poly = [(-0.5, 0.1), (-0.5, 0.8), (0.75, 0.8), (0.75, 0.1)];
        let weights = split1d(poly, &grid, true);
        assert!((weights.iter().map(|(_, w)| w).sum::<f64>() - 1.).abs() < 1e-12);
        assert!(weights.iter().all(|(_, w)| w.is_finite() && *w > 0.));
        let outside = IntegrationGrid2d::new(4, 1, (0., 2.), (1., 2.)).unwrap();
        assert!(split1d(poly, &outside, true).is_empty());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::{
        frame::Frame,
        integration::{InputOrder, IntegrationMethod, Integrator, SparseLayout},
        mask::Mask,
        poni::Poni,
    };
    fn geometry() -> Geometry {
        let mut poni = Poni::new();
        poni.distance = 0.08;
        poni.pixel1 = 1e-4;
        poni.pixel2 = 1.2e-4;
        poni.poni1 = -0.0013;
        poni.poni2 = -0.0021;
        poni.wavelength = 5e-11;
        Geometry::new(12, 16, poni)
    }
    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).abs() < 1e-10 * (1. + b.abs()), "{a} != {b}");
        }
    }
    #[test]
    fn pyfai_method_reuses_weighted_errors_and_fixed_dynamic_plans() {
        let g = geometry();
        let data: Vec<_> = (0..g.size()).map(|i| 30. + i as f64).collect();
        let frame = Frame {
            name: "test".into(),
            data: data.clone(),
            dim1: g.dim1,
            dim2: g.dim2,
        };
        let mut ai = Integrator::new(g)
            .with_method(IntegrationMethod::PyfaiFull)
            .with_radial_unit(RadialUnit::TthDeg)
            .with_sparse_layout(SparseLayout::Csc);
        let (x, y, e) = ai.integrate1d_with_errors(&frame, 32);
        let plan = ai.prepare_fixed1d((12, 16), 32).unwrap();
        let fixed = plan.integrate_frame(&data, true).unwrap();
        close(&x, &fixed.radial);
        close(&y, &fixed.intensity);
        close(&e, fixed.errors.as_ref().unwrap());
        let mut worker = plan.worker(true, InputOrder::Detector);
        worker.begin_range(0..data.len()).unwrap();
        worker.push_f64_block(0, &data).unwrap();
        let blocks = plan.finish_ranges(&mut [worker]).unwrap();
        close(&blocks.intensity, &y);
        let dynamic = ai.prepare_dynamic1d((12, 16), 32).unwrap();
        let mask: Vec<u8> = (0..data.len()).map(|i| u8::from(i % 7 == 0)).collect();
        ai.set_mask(Mask {
            data: mask.clone(),
            dim1: 12,
            dim2: 16,
        });
        let reference = ai.integrate1d_with_errors(&frame, 32);
        let result = dynamic.integrate(&data, Some(&mask), true, false);
        close(&result.intensity, &reference.1);
        close(result.errors.as_ref().unwrap(), &reference.2);
        ai.set_method(IntegrationMethod::PixelSplit);
        let polygon = ai.integrate1d(&frame, 32).1;
        ai.set_method(IntegrationMethod::PyfaiFull);
        close(&ai.integrate1d(&frame, 32).1, &reference.1);
        assert!(
            polygon
                .iter()
                .zip(&reference.1)
                .any(|(a, b)| (a - b).abs() > 1e-7)
        );
    }
    #[test]
    fn singular_beam_pixel_retains_nonnegative_fractions_and_count_variance() {
        let mut p = Poni::new();
        p.pixel1 = 1.;
        p.pixel2 = 1.;
        p.poni1 = 0.5;
        p.poni2 = 0.5;
        p.distance = 100.;
        let g = Geometry::new(1, 1, p);
        let lut =
            build_lut1d_pyfai_csr(&g, RadialUnit::TthRad, 16, None, Some([0., 0.01])).unwrap();
        let weights = lut.values.as_ref().unwrap();
        assert!(weights.iter().all(|w| w.is_finite() && *w >= 0.));
        assert!((weights.iter().sum::<f64>() - 1.).abs() < 1e-12);
        let (_, y, e) = crate::lut::integrate1d_with_errors(&[100.], &[100.], &lut, None, &[1.]);
        for (y, e) in y.iter().zip(e) {
            if *y > 0. {
                assert!((*y - 100.).abs() < 1e-12);
                assert!((e - 10.).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn ordinary_cake_and_shared_grid_use_the_same_pyfai_kernel() {
        let g = geometry();
        let radial = pyfai_radial_range(&g, RadialUnit::TthDeg);
        let ai = Integrator::new(g)
            .with_method(IntegrationMethod::PyfaiFull)
            .with_radial_unit(RadialUnit::TthDeg);
        let data: Vec<_> = (0..192).map(|i| 30. + i as f64).collect();
        let cake = ai.integrate2d_from_slice(&data, 12, 16, 32, 12);
        let grid = IntegrationGrid2d::new(32, 12, (radial[0], radial[1]), (-PI, PI)).unwrap();
        let raw = ai
            .integrate2d_raw_on_grid_from_slice(&data, 12, 16, &grid, false)
            .unwrap();
        let expected: Vec<_> = raw
            .sum_signal
            .iter()
            .zip(raw.sum_norm)
            .map(|(s, n)| if n == 0. { 0. } else { s / n })
            .collect();
        close(&cake, &expected);
    }
}
