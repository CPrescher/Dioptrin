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
use rayon::prelude::*;
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
/// On the automatic radial range (`radial` is None) the beam pixel's samples below
/// the first edge (the smallest pixel corner) go to bin 0, so it is conserved.
pub fn build_lut1d_pyfai_csr(
    g: &Geometry,
    unit: RadialUnit,
    bins: usize,
    azimuth: Option<[f64; 2]>,
    radial: Option<[f64; 2]>,
) -> Result<Lut1d, String> {
    // The automatic range starts at the smallest pixel corner, but the beam pixel
    // reaches r = 0 below it; its samples there belong to the first bin.
    let clamp_low = radial.is_none();
    let radial = radial.unwrap_or_else(|| pyfai_radial_range(g, unit));
    let angles = azimuth.unwrap_or([-PI, PI]);
    let grid = IntegrationGrid2d::new(bins, 1, (radial[0], radial[1]), (angles[0], angles[1]))?;
    let (tth, azi) = g.angular_corners();
    let r = unit.convert_from_tth(&tth, g.poni.wavelength);
    let polygon = |pixel: usize| {
        let [a, b, c, d] = g.corner_indices(pixel);
        [a, d, c, b].map(|i| (r[i], azi[i]))
    };
    let sign = corner_sign(g, (0..g.size()).into_par_iter().map(polygon));
    let mut buckets = vec![Vec::<(usize, f64)>::new(); bins];
    let mut entries = 0usize;
    for pixel in 0..g.size() {
        let poly = polygon(pixel);
        let contributions = if singular(&poly) {
            sample(g, pixel, unit, &grid, clamp_low)
        } else {
            split1d(poly, &grid, azimuth.is_some(), sign)
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

/// pyFAI's `_corner_sign` (regrid_common.pxi): +1 when the pixel corners, in
/// pyFAI order A, B, C, D, trace the (radial, azimuth) quadrilateral the way an
/// unmirrored detector does, -1 when they trace it the other way round.
///
/// pyFAI multiplies `area4p` by this sign in its seam test and derives it from
/// the detector orientation alone (-1 for 2 and 4, which mirror one axis).
/// Here it is measured on the corners the kernels actually receive, which for
/// uniform and separable detectors gives exactly pyFAI's rule. Transformed
/// maps (`Detector::transformed`) report orientation 3 whatever they do: their
/// per-pixel corners (1D) keep the base detector's order, while the shared
/// corner grid (2D) follows the transformed image. The 1D winding therefore
/// follows the base orientation, the 2D winding the base orientation combined
/// with the transform (a Horizontal flip of an orientation-3 base is mirrored
/// in 2D only). The orientation cannot express that; the corners can.
///
/// Each non-singular pixel votes with the sign of its true signed area
/// (azimuth unwrapped along the contour, so seam pixels vote correctly);
/// unmirrored pixels have negative area. A geometry without a detector is
/// pyFAI's default orientation and keeps +1 without a pass over the pixels.
/// The votes are integers, so the parallel count is exact in any order.
pub(super) fn corner_sign(
    g: &Geometry,
    polys: impl ParallelIterator<Item = [(f64, f64); 4]>,
) -> f64 {
    if g.detector().is_none() {
        return 1.0;
    }
    let votes: i64 = polys
        .map(|poly| {
            if singular(&poly) {
                return 0;
            }
            let area = unwrapped_area(&poly);
            if area < 0.0 {
                1
            } else if area > 0.0 {
                -1
            } else {
                0
            }
        })
        .sum();
    if votes < 0 { -1.0 } else { 1.0 }
}

/// Signed area of a quadrilateral whose azimuths are unwrapped along the
/// contour (consecutive differences taken in (-π, π]).
///
/// Computed as half the cross product of the diagonals, which equals the
/// shoelace sum for any quadrilateral but works on coordinate differences
/// (pixel-sized) instead of products of absolute coordinates (of order one).
/// In the shoelace sum single votes turned to rounding noise below a pixel
/// span of about 3e-8 rad, and the majority could flip further down (issue
/// #44); this form keeps every vote until neighbouring corner angles are a
/// few ulps apart (pixel / distance about 1e-15).
fn unwrapped_area(poly: &[(f64, f64); 4]) -> f64 {
    let mut chi = poly[0].1;
    let mut pts = [(poly[0].0, chi); 4];
    for i in 1..4 {
        chi += wrap_tau(poly[i].1 - poly[i - 1].1 + PI) - PI;
        pts[i] = (poly[i].0, chi);
    }
    let [a, b, c, d] = pts;
    0.5 * ((c.0 - a.0) * (d.1 - b.1) - (c.1 - a.1) * (d.0 - b.0))
}

/// `x.rem_euclid(TAU)`, bit for bit, without the `fmod` call for values
/// already in `[0, TAU)` (the common case), which it returns unchanged.
#[inline]
pub(super) fn wrap_tau(x: f64) -> f64 {
    if (0.0..TAU).contains(&x) {
        x
    } else {
        x.rem_euclid(TAU)
    }
}

pub(super) fn singular(poly: &[(f64, f64); 4]) -> bool {
    let mut winding = 0.0;
    for i in 0..4 {
        if poly[i].0 == 0.0 {
            return true;
        }
        let delta = wrap_tau(poly[(i + 1) % 4].1 - poly[i].1 + PI) - PI;
        if delta.abs() >= PI - 64.0 * f64::EPSILON {
            return true;
        }
        winding += delta;
    }
    winding.abs() > PI
}

// Match the upstream safety convention: singular beam pixels use nonnegative
// detector-plane fractions. These exceptional weights are not bitwise pyFAI.
// With `clamp_low` (automatic radial range, whose lower edge is the smallest
// pixel corner), samples below the first radial edge are counted in bin 0
// instead of being dropped; the azimuthal and upper radial limits still apply.
pub(super) fn sample(
    g: &Geometry,
    pixel: usize,
    unit: RadialUnit,
    grid: &IntegrationGrid2d,
    clamp_low: bool,
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
        if (r < r0 && !clamp_low) || r >= r1 || a >= a1 {
            continue;
        }
        let ir = if r < r0 {
            0
        } else {
            ((r - r0) / (r1 - r0) * nr as f64) as usize
        };
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
// `sign` is `corner_sign`; the weights themselves are winding-independent
// because each pixel is normalized by its own signed sum, as in pyFAI.
fn split1d(
    mut poly: [(f64, f64); 4],
    grid: &IntegrationGrid2d,
    sector: bool,
    sign: f64,
) -> Vec<(usize, f64)> {
    let (a0, a1) = grid.azimuthal_range();
    let offset = if a0 >= -PI && a1 <= PI {
        0.0
    } else if a0 >= 0.0 && a1 <= TAU {
        PI
    } else {
        a0 + PI
    };
    for p in &mut poly {
        p.1 = wrap_tau(p.1 - offset + PI) - PI;
    }
    let [a, b, c, d] = poly;
    // Deliberately pyFAI's `area4p` (regrid_common.pxi), operator precedence
    // included: `0.5 * X - Y`, not the signed diagonal area `0.5 * (X - Y)`.
    // Only the sign is used; the 2D port in `pyfai_split.rs` uses the same form.
    // pyFAI's `_recenter` multiplies it by the orientation's corner sign.
    let area = sign * (0.5 * (c.0 - a.0) * (d.1 - b.1) - (c.1 - a.1) * (d.0 - b.0));
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
    fn wrap_tau_is_rem_euclid_bit_for_bit() {
        let mut values = vec![
            0.0,
            -0.0,
            TAU,
            -TAU,
            PI,
            -PI,
            TAU.next_down(),
            TAU.next_up(),
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            1e300,
            -1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.push((state >> 11) as f64 / (1u64 << 53) as f64 * 8.0 * TAU - 4.0 * TAU);
        }
        for x in values {
            assert_eq!(wrap_tau(x).to_bits(), x.rem_euclid(TAU).to_bits(), "{x:e}");
        }
    }

    #[test]
    fn radial_line_weights_match_independent_pyfai_golden() {
        // pyFAI 2026.9.0 fullSplit1D_engine, one unit-count pixel, variance=1,
        // explicit [0,2] effective radial edges and four bins.
        let grid = IntegrationGrid2d::new(4, 1, (0., 2.), (-PI, PI)).unwrap();
        let weights = split1d(
            [(0.1, 0.2), (1.2, 0.3), (1.8, 1.5), (0.4, 1.7)],
            &grid,
            false,
            1.0,
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
        let weights = split1d(poly, &grid, true, 1.0);
        assert!((weights.iter().map(|(_, w)| w).sum::<f64>() - 1.).abs() < 1e-12);
        assert!(weights.iter().all(|(_, w)| w.is_finite() && *w > 0.));
        let outside = IntegrationGrid2d::new(4, 1, (0., 2.), (1., 2.)).unwrap();
        assert!(split1d(poly, &outside, true, 1.0).is_empty());
    }
    /// pyFAI's seam test is `0.5 * X - Y`. For this folded quadrilateral it is
    /// negative (no unwrapping) while the true signed area `0.5 * (X - Y)` is
    /// positive, which would unwrap two corners to above 2π - 1 and pull the
    /// pixel into the sector.
    #[test]
    fn seam_test_uses_pyfai_operator_precedence() {
        let poly = [(1.0, -0.5), (1.2, -0.4), (2.0, 0.25), (2.2, 0.6)];
        let grid = IntegrationGrid2d::new(10, 1, (0.0, 3.0), (1.0, 2.0)).unwrap();
        assert!(split1d(poly, &grid, true, 1.0).is_empty());
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

    /// 9x11 detector, 100 µm pixels, 5 cm distance, beam in pixel (4, 5).
    fn beam_geometry(poni1: f64, poni2: f64) -> Geometry {
        let mut p = Poni::new();
        p.pixel1 = 1e-4;
        p.pixel2 = 1e-4;
        p.distance = 0.05;
        p.poni1 = poni1;
        p.poni2 = poni2;
        p.wavelength = 1e-10;
        Geometry::new(9, 11, p)
    }

    fn pixel_total(lut: &Lut1d, pixel: usize) -> f64 {
        lut.col_idx
            .iter()
            .zip(lut.values.as_ref().unwrap())
            .filter(|(p, _)| **p == pixel)
            .map(|(_, w)| w)
            .sum()
    }

    const BEAM_PIXEL: usize = 4 * 11 + 5;

    #[test]
    fn automatic_range_conserves_the_beam_pixel() {
        for (poni1, poni2) in [(4.5e-4, 5.5e-4), (4.3e-4, 5.6e-4)] {
            let g = beam_geometry(poni1, poni2);
            let lut = build_lut1d_pyfai_csr(&g, RadialUnit::TthDeg, 200, None, None).unwrap();
            let total = pixel_total(&lut, BEAM_PIXEL);
            assert!((total - 1.0).abs() < 1e-12, "({poni1}, {poni2}): {total}");
            for pixel in 0..g.size() {
                let total = pixel_total(&lut, pixel);
                assert!((total - 1.0).abs() < 1e-9, "pixel {pixel}: {total}");
            }
        }
    }

    #[test]
    fn explicit_range_and_sector_still_clip_the_beam_pixel() {
        let g = beam_geometry(4.5e-4, 5.5e-4);
        let auto = pyfai_radial_range(&g, RadialUnit::TthDeg);
        // Range starting at the inscribed circle: 1 - pi/4 of the pixel remains.
        let min = (0.5e-4_f64 / 0.05).atan().to_degrees();
        let lut =
            build_lut1d_pyfai_csr(&g, RadialUnit::TthDeg, 200, None, Some([min, auto[1]])).unwrap();
        let total = pixel_total(&lut, BEAM_PIXEL);
        assert!((total - (1.0 - PI / 4.0)).abs() < 2e-3, "{total}");
        // The automatic range passed explicitly is not clamped: the centred
        // beam pixel lies entirely below its smallest corner.
        let lut = build_lut1d_pyfai_csr(&g, RadialUnit::TthDeg, 200, None, Some(auto)).unwrap();
        assert_eq!(pixel_total(&lut, BEAM_PIXEL), 0.0);
        // Automatic radial range with an azimuthal quadrant: samples outside
        // the sector are still dropped.
        let lut = build_lut1d_pyfai_csr(&g, RadialUnit::TthDeg, 200, Some([0.0, PI / 2.0]), None)
            .unwrap();
        let total = pixel_total(&lut, BEAM_PIXEL);
        assert!((total - 0.25).abs() < 2e-3, "{total}");
    }

    /// The corner sign is pyFAI's orientation rule for uniform detectors:
    /// every pixel, the seam pixels included, has the expected winding.
    #[test]
    fn corner_sign_follows_the_mirrored_orientations() {
        use crate::detector::Detector;
        use std::sync::Arc;
        let sign = |g: &Geometry, unanimous: bool| {
            let (tth, azi) = g.angular_corners();
            let polygons: Vec<_> = (0..g.size())
                .map(|pixel| {
                    let [a, b, c, d] = g.corner_indices(pixel);
                    [a, d, c, b].map(|i| (tth[i], azi[i]))
                })
                .collect();
            if unanimous {
                let regular: Vec<_> = polygons.iter().filter(|p| !singular(p)).collect();
                let n = regular.iter().filter(|p| unwrapped_area(p) < 0.0).count();
                assert!(regular.len() > 90);
                assert!(
                    n == 0 || n == regular.len(),
                    "{n} of {} pixels unmirrored",
                    regular.len()
                );
            }
            corner_sign(g, polygons.into_par_iter())
        };
        let g = beam_geometry(4.5e-4, 5.5e-4);
        assert_eq!(sign(&g, true), 1.0);
        for (orientation, expected) in [(0, 1.0), (1, 1.0), (2, -1.0), (3, 1.0), (4, -1.0)] {
            let mut d = Detector::custom(1e-4, 1e-4);
            d.orientation = orientation;
            let g = g.clone().with_detector(Arc::new(d)).unwrap();
            assert_eq!(sign(&g, true), expected, "orientation {orientation}");
        }
    }

    /// A tie keeps +1, no votes at all included. Every pixel of a 2 × 2 image
    /// with the beam on its centre node is singular (and sampled, so the sign
    /// is unused); the mirrored orientation 2 would give -1 by pyFAI's rule.
    #[test]
    fn corner_sign_tie_keeps_plus_one() {
        use crate::detector::Detector;
        use std::sync::Arc;
        let polygons = |g: &Geometry| {
            let (tth, azi) = g.angular_corners();
            (0..g.size())
                .map(|pixel| {
                    let [a, b, c, d] = g.corner_indices(pixel);
                    [a, d, c, b].map(|i| (tth[i], azi[i]))
                })
                .collect::<Vec<_>>()
        };
        let mut d = Detector::custom(1e-4, 1e-4);
        d.orientation = 2;
        let poni = beam_geometry(1e-4, 1e-4).poni;
        let g = Geometry::new(2, 2, poni)
            .with_detector(Arc::new(d))
            .unwrap();
        let all_singular = polygons(&g);
        assert!(all_singular.iter().all(singular));
        assert_eq!(corner_sign(&g, all_singular.into_par_iter()), 1.0);
        assert_eq!(corner_sign(&g, rayon::iter::empty()), 1.0);
        // One mirrored pixel against one unmirrored pixel.
        let regular = polygons(&beam_geometry(-3e-4, -2e-4))[0];
        let mut mirrored = regular;
        mirrored.reverse();
        assert!(unwrapped_area(&regular) < 0.0 && unwrapped_area(&mirrored) > 0.0);
        assert_eq!(corner_sign(&g, vec![mirrored].into_par_iter()), -1.0);
        assert_eq!(
            corner_sign(&g, vec![regular, mirrored].into_par_iter()),
            1.0
        );
        // A collapsed pixel is not singular but has no area: it abstains.
        let point = [(0.1, 0.2); 4];
        assert!(!singular(&point) && unwrapped_area(&point) == 0.0);
        assert_eq!(corner_sign(&g, vec![point; 3].into_par_iter()), 1.0);
        assert_eq!(
            corner_sign(&g, vec![point, point, mirrored].into_par_iter()),
            -1.0
        );
    }

    /// Pixel polygons as `build_lut1d_pyfai_csr` (per-pixel corners) and the
    /// shared-grid builder (corner grid) hand them to `corner_sign`.
    fn corner_polygons(g: &Geometry) -> [Vec<[(f64, f64); 4]>; 2] {
        let (tth, azi) = g.angular_corners();
        let one_d = (0..g.size())
            .map(|pixel| {
                let [a, b, c, d] = g.corner_indices(pixel);
                [a, d, c, b].map(|i| (tth[i], azi[i]))
            })
            .collect();
        let (d1, d2) = g.corner_grid();
        let (tth, azi) = g.calc_tth_and_azimuth(&d1, &d2);
        let w = g.dim2 + 1;
        let two_d = (0..g.size())
            .map(|pixel| {
                let b = pixel / g.dim2 * w + pixel % g.dim2;
                [b, b + w, b + w + 1, b + 1].map(|i| (tth[i], azi[i]))
            })
            .collect();
        [one_d, two_d]
    }

    /// `corner_sign` of `polygons`, checking that every pixel that votes
    /// (non-singular, non-zero area) votes for it.
    #[track_caller]
    fn unanimous_sign(g: &Geometry, polygons: &[[(f64, f64); 4]], label: &str) -> f64 {
        let sign = corner_sign(g, polygons.par_iter().copied());
        let voting = polygons.iter().filter(|p| !singular(p)).count();
        let agree = polygons
            .iter()
            .filter(|p| !singular(p) && unwrapped_area(p) * sign < 0.0)
            .count();
        assert!(voting > 0, "{label}: no votes");
        assert_eq!(
            agree, voting,
            "{label}: {agree} of {voting} pixels vote {sign}"
        );
        sign
    }

    fn mirrored(orientation: u8) -> bool {
        matches!(orientation, 2 | 4)
    }

    /// Issue #44: pixel / distance of 1e-12 and 1e-14, the PONI 0.5 mm beside
    /// the detector and rot1 = 1.2. The shoelace sum of absolute (2θ, χ)
    /// products cancelled to rounding noise; with 1e-12 m pixels orientations
    /// 0 and 3 voted -1 (974 against 962 pixels, 2160 areas exactly zero).
    #[test]
    fn corner_sign_is_unanimous_for_unphysically_small_pixels() {
        use crate::{detector::Detector, poni::Poni};
        use std::sync::Arc;
        for pixel in [1e-12, 1e-14] {
            for orientation in 0..=4 {
                let mut p = Poni::new();
                (p.pixel1, p.pixel2, p.distance, p.wavelength) = (pixel, pixel, 1.0, 1e-10);
                (p.poni1, p.poni2, p.rot1) = (4.5e-4, 5.5e-4, 1.2);
                let mut d = Detector::custom(pixel, pixel);
                d.orientation = orientation;
                let g = Geometry::new(64, 64, p).with_detector(Arc::new(d)).unwrap();
                let expected = if mirrored(orientation) { -1.0 } else { 1.0 };
                for (path, polygons) in ["1D", "2D"].iter().zip(corner_polygons(&g)) {
                    let label = format!("{pixel:e} m, orientation {orientation}, {path}");
                    assert_eq!(unanimous_sign(&g, &polygons, &label), expected, "{label}");
                }
            }
        }
    }

    /// The sign for every pixel-map kind, orientation and transform, binned
    /// and unbinned, with the beam on the detector (the seam crosses it).
    /// Per-pixel corners (1D) keep the base detector's order, so their sign is
    /// pyFAI's rule for the base orientation. The corner grid (2D) follows the
    /// transformed image, so a transform that mirrors (Transpose, Transverse,
    /// Horizontal, Vertical) flips it. A uniform map without spline is not
    /// remapped by `transformed` and keeps the rule in both.
    #[test]
    fn corner_sign_for_every_map_kind_orientation_and_transform() {
        use crate::{
            detector::{Detector, DetectorTransform::*, PixelMap, spline::Spline},
            poni::Poni,
        };
        use std::sync::Arc;
        const FULL: [usize; 2] = [36, 72];
        const PIXEL: f64 = 1e-4;
        let edges = |n: usize, axis: usize, gaps: bool, curved: bool| -> Vec<[[f64; 3]; 2]> {
            let mut lo = 0.;
            (0..n)
                .map(|i| {
                    if gaps && (i == 7 || i == 20) {
                        lo += 0.4 * PIXEL;
                    }
                    let hi = lo + PIXEL * (1. + 0.05 * (i as f64 * 0.7).sin());
                    let z = |x: f64| {
                        if curved {
                            2e-4 * (x / 3e-3).powi(2)
                        } else {
                            0.
                        }
                    };
                    let (mut a, mut b) = ([z(lo), 0., 0.], [z(hi), 0., 0.]);
                    (a[axis], b[axis]) = (lo, hi);
                    lo = hi;
                    [a, b]
                })
                .collect()
        };
        let separable = |gaps, curved| PixelMap::Separable {
            slow: edges(FULL[0], 1, gaps, curved),
            fast: edges(FULL[1], 2, gaps, curved),
        };
        let corners = {
            let PixelMap::Separable { slow, fast } = separable(true, false) else {
                unreachable!()
            };
            let add = |a: [f64; 3], b: [f64; 3]| std::array::from_fn(|k| a[k] + b[k]);
            let values = (0..FULL[0] * FULL[1])
                .map(|i| {
                    let (s, f) = (slow[i / FULL[1]], fast[i % FULL[1]]);
                    [
                        add(s[0], f[0]),
                        add(s[1], f[0]),
                        add(s[1], f[1]),
                        add(s[0], f[1]),
                    ]
                })
                .collect();
            PixelMap::Corners {
                shape: FULL,
                values: Arc::new(values),
            }
        };
        let spline = Arc::new(
            Spline::parse(include_str!("../../test_data/detector_reference.spline")).unwrap(),
        );
        let mut cases = 0;
        for orientation in 0..=4 {
            let base = |map: PixelMap, with_spline: bool| {
                let mut d = Detector::custom(PIXEL, PIXEL);
                d.max_shape = Some(FULL);
                d.orientation = orientation;
                d.map = map;
                d.spline = with_spline.then(|| spline.clone());
                d
            };
            let bases = [
                ("uniform", base(PixelMap::Uniform, false)),
                ("spline", base(PixelMap::Uniform, true)),
                ("separable", base(separable(false, false), false)),
                ("gapped", base(separable(true, false), false)),
                ("curved", base(separable(true, true), false)),
                ("corners", base(corners.clone(), false)),
            ];
            for (name, base) in bases {
                for op in [
                    Identity, Half, Transpose, Transverse, Left, Right, Horizontal, Vertical,
                ] {
                    let t = base.transformed(op).unwrap();
                    let rule = if mirrored(orientation) { -1.0 } else { 1.0 };
                    let flip = matches!(t.map, PixelMap::Transformed { .. })
                        && matches!(op, Transpose | Transverse | Horizontal | Vertical);
                    let expected = [rule, if flip { -rule } else { rule }];
                    let full = t.max_shape.unwrap();
                    for bin in [[1, 1], [2, 3]] {
                        let n = [full[0] / bin[0], full[1] / bin[1]];
                        let px = t.effective_pixels(n);
                        let mut p = Poni::new();
                        (p.pixel1, p.pixel2, p.distance, p.wavelength) =
                            (px[0], px[1], 0.03, 1e-10);
                        p.poni1 = (n[0] as f64 * 0.47 + 0.3) * px[0];
                        p.poni2 = (n[1] as f64 * 0.53 + 0.2) * px[1];
                        let g = Geometry::new(n[0], n[1], p)
                            .with_detector(Arc::new(t.clone()))
                            .unwrap();
                        for ((path, polygons), expected) in
                            ["1D", "2D"].iter().zip(corner_polygons(&g)).zip(expected)
                        {
                            let label = format!("{name} o{orientation} {op:?} {bin:?} {path}");
                            assert_eq!(unanimous_sign(&g, &polygons, &label), expected, "{label}");
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 5 * 6 * 8 * 2 * 2);
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
