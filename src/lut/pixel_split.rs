//! Pixel-splitting CSR LUT builders (1D/2D) producing weighted `Lut1d`/`Lut2d`.
//! Integrate with `crate::lut::integrate1d` / `integrate2d`.
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::geometry::Geometry;
use crate::integration::{IntegrationGrid2d, RadialUnit};
use crate::lut::counting_csr::{Lut1d, Lut2d};
use crate::lut::pyfai_1d;
use crate::math::{self, histogram};

/// Compute the signed area of a polygon given as (x, y) vertices.
pub(super) fn polygon_area(poly: &[(f64, f64)]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..poly.len() {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % poly.len()];
        area += x0 * y1 - x1 * y0;
    }
    0.5 * area
}

/// Intersection of a segment with a vertical line at x = x_edge.
fn intersect_with_x(p0: (f64, f64), p1: (f64, f64), x_edge: f64) -> (f64, f64) {
    let dx = p1.0 - p0.0;
    if dx.abs() < f64::EPSILON {
        return (x_edge, p0.1);
    }
    let t = (x_edge - p0.0) / dx;
    (x_edge, p0.1 + t * (p1.1 - p0.1))
}

/// Intersection of a segment with a horizontal line at y = y_edge.
fn intersect_with_y(p0: (f64, f64), p1: (f64, f64), y_edge: f64) -> (f64, f64) {
    let dy = p1.1 - p0.1;
    if dy.abs() < f64::EPSILON {
        return (p0.0, y_edge);
    }
    let t = (y_edge - p0.1) / dy;
    (p0.0 + t * (p1.0 - p0.0), y_edge)
}

/// Clip polygon against x >= x_min.
fn clip_left_into(poly: &[(f64, f64)], x_min: f64, out: &mut Vec<(f64, f64)>) {
    out.clear();
    if poly.is_empty() {
        return;
    }
    let mut prev = poly[poly.len() - 1];
    let mut prev_inside = prev.0 >= x_min;
    for &curr in poly {
        let curr_inside = curr.0 >= x_min;
        if curr_inside {
            if !prev_inside {
                out.push(intersect_with_x(prev, curr, x_min));
            }
            out.push(curr);
        } else if prev_inside {
            out.push(intersect_with_x(prev, curr, x_min));
        }
        prev = curr;
        prev_inside = curr_inside;
    }
}

/// Clip polygon against x <= x_max.
fn clip_right_into(poly: &[(f64, f64)], x_max: f64, out: &mut Vec<(f64, f64)>) {
    out.clear();
    if poly.is_empty() {
        return;
    }
    let mut prev = poly[poly.len() - 1];
    let mut prev_inside = prev.0 <= x_max;
    for &curr in poly {
        let curr_inside = curr.0 <= x_max;
        if curr_inside {
            if !prev_inside {
                out.push(intersect_with_x(prev, curr, x_max));
            }
            out.push(curr);
        } else if prev_inside {
            out.push(intersect_with_x(prev, curr, x_max));
        }
        prev = curr;
        prev_inside = curr_inside;
    }
}

/// Clip polygon against y >= y_min.
fn clip_bottom_into(poly: &[(f64, f64)], y_min: f64, out: &mut Vec<(f64, f64)>) {
    out.clear();
    if poly.is_empty() {
        return;
    }
    let mut prev = poly[poly.len() - 1];
    let mut prev_inside = prev.1 >= y_min;
    for &curr in poly {
        let curr_inside = curr.1 >= y_min;
        if curr_inside {
            if !prev_inside {
                out.push(intersect_with_y(prev, curr, y_min));
            }
            out.push(curr);
        } else if prev_inside {
            out.push(intersect_with_y(prev, curr, y_min));
        }
        prev = curr;
        prev_inside = curr_inside;
    }
}

/// Clip polygon against y <= y_max.
fn clip_top_into(poly: &[(f64, f64)], y_max: f64, out: &mut Vec<(f64, f64)>) {
    out.clear();
    if poly.is_empty() {
        return;
    }
    let mut prev = poly[poly.len() - 1];
    let mut prev_inside = prev.1 <= y_max;
    for &curr in poly {
        let curr_inside = curr.1 <= y_max;
        if curr_inside {
            if !prev_inside {
                out.push(intersect_with_y(prev, curr, y_max));
            }
            out.push(curr);
        } else if prev_inside {
            out.push(intersect_with_y(prev, curr, y_max));
        }
        prev = curr;
        prev_inside = curr_inside;
    }
}

/// Clip polygon against an axis-aligned rectangle.
pub(super) fn clip_polygon_rect_scratch<'a>(
    poly: &[(f64, f64)],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    buf1: &'a mut Vec<(f64, f64)>,
    buf2: &'a mut Vec<(f64, f64)>,
) -> &'a [(f64, f64)] {
    clip_left_into(poly, x_min, buf1);
    if buf1.is_empty() {
        return &buf1[..];
    }
    clip_right_into(buf1, x_max, buf2);
    if buf2.is_empty() {
        return &buf2[..];
    }
    clip_bottom_into(buf2, y_min, buf1);
    if buf1.is_empty() {
        return &buf1[..];
    }
    clip_top_into(buf1, y_max, buf2);
    &buf2[..]
}

/// Unwrap an angle to stay within +/- PI of a reference value.
fn unwrap_angle(angle: f64, reference: f64) -> f64 {
    let mut a = angle;
    let delta = a - reference;
    if delta > PI {
        a -= 2.0 * PI;
    } else if delta < -PI {
        a += 2.0 * PI;
    }
    a
}

/// Per-bin `(pixel, fraction)` contributions of the 1D polygon builders,
/// before the per-bin normalization applied by the public builders.
///
/// `fraction = area(pixel ∩ bin) / area(pixel)`, both measured on the pixel's
/// quadrilateral in (radial, azimuth) space, so a pixel that lies completely
/// inside the radial range contributes a total weight of one, independent of
/// its distance from the beam. A pixel partly outside `[min, min + n·step)`
/// keeps only its in-range fraction. Pixels that contain or touch the point of
/// normal incidence have no single polar quadrilateral; they use the same
/// detector-plane sampling as the shared-grid 2D and pyFAI-compatible paths.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lut1d_ps_fractions(
    geometry: &Geometry,
    radial: &[f64],
    n_points: usize,
    unit: RadialUnit,
    min: f64,
    step: f64,
    radial_corners: &[f64],
    azimuthal_corners: &[f64],
) -> Vec<Vec<(usize, f64)>> {
    let azimuthal = geometry.calc_azimuth_all();
    // Only used for singular beam pixels; an unrepresentable grid (for example a
    // degenerate range) falls back to the polygon for those pixels as before.
    let sample_grid =
        IntegrationGrid2d::new(n_points, 1, (min, min + n_points as f64 * step), (-PI, PI)).ok();

    // Single pass: build per-row buckets in parallel, then flatten to CSR.
    let (buckets, _, _) = radial
        .par_iter()
        .enumerate()
        .fold(
            || {
                (
                    vec![Vec::<(usize, f64)>::new(); n_points],
                    Vec::with_capacity(8),
                    Vec::with_capacity(8),
                )
            },
            |(mut acc, mut buf1, mut buf2), (idx, &center_radial)| {
                let [i00, i01, i11, i10] = geometry.corner_indices(idx);

                let r00 = radial_corners[i00];
                let r01 = radial_corners[i01];
                let r10 = radial_corners[i10];
                let r11 = radial_corners[i11];

                if let Some(grid) = &sample_grid {
                    let raw =
                        [i00, i01, i11, i10].map(|i| (radial_corners[i], azimuthal_corners[i]));
                    if pyfai_1d::singular(&raw) {
                        for (bin, fraction) in pyfai_1d::sample(geometry, idx, unit, grid) {
                            acc[bin].push((idx, fraction));
                        }
                        return (acc, buf1, buf2);
                    }
                }

                let center_azimuthal = azimuthal[idx];
                let a00 = unwrap_angle(azimuthal_corners[i00], center_azimuthal);
                let a01 = unwrap_angle(azimuthal_corners[i01], center_azimuthal);
                let a10 = unwrap_angle(azimuthal_corners[i10], center_azimuthal);
                let a11 = unwrap_angle(azimuthal_corners[i11], center_azimuthal);

                let poly = [(r00, a00), (r01, a01), (r11, a11), (r10, a10)];
                let area_total = polygon_area(&poly).abs();
                if area_total <= 0.0 {
                    // Degenerate quadrilateral: assign the whole pixel to the bin
                    // of its centre, or drop it if the centre is out of range.
                    let pos = (center_radial - min) / step;
                    if pos < 0.0 {
                        return (acc, buf1, buf2);
                    }
                    let bin = pos.floor() as isize;
                    if bin >= 0 && (bin as usize) < n_points {
                        acc[bin as usize].push((idx, 1.0));
                    }
                    return (acc, buf1, buf2);
                }

                let mut rmin = r00;
                let mut rmax = r00;
                for v in [r01, r10, r11] {
                    if v < rmin {
                        rmin = v;
                    }
                    if v > rmax {
                        rmax = v;
                    }
                }

                let mut start = ((rmin - min) / step).floor() as isize;
                let mut end = ((rmax - min) / step).floor() as isize;
                if end < 0 || start >= n_points as isize {
                    return (acc, buf1, buf2);
                }
                if start < 0 {
                    start = 0;
                }
                if end >= n_points as isize {
                    end = n_points as isize - 1;
                }

                let mut amin = a00;
                let mut amax = a00;
                for v in [a01, a10, a11] {
                    if v < amin {
                        amin = v;
                    }
                    if v > amax {
                        amax = v;
                    }
                }

                for b in start..=end {
                    let bin_min = min + b as f64 * step;
                    let bin_max = bin_min + step;
                    let clipped = clip_polygon_rect_scratch(
                        &poly, bin_min, bin_max, amin, amax, &mut buf1, &mut buf2,
                    );
                    let area = polygon_area(clipped).abs();
                    if area > 0.0 {
                        // Fraction of this pixel, not polar-plane area: areas of
                        // different pixels are not comparable (they scale ~1/r).
                        acc[b as usize].push((idx, area / area_total));
                    }
                }
                (acc, buf1, buf2)
            },
        )
        .reduce(
            || {
                (
                    vec![Vec::<(usize, f64)>::new(); n_points],
                    Vec::new(),
                    Vec::new(),
                )
            },
            |(mut a, _, _), (b, _, _)| {
                for (row, mut v) in b.into_iter().enumerate() {
                    a[row].append(&mut v);
                }
                (a, Vec::new(), Vec::new())
            },
        );
    buckets
}

/// Flatten per-bin fractions to CSR and normalize each bin (row) to unit sum.
fn lut1d_from_fractions(buckets: Vec<Vec<(usize, f64)>>, min: f64, step: f64) -> Lut1d {
    let n_points = buckets.len();
    // Prefix sum
    let mut row_ptr = vec![0usize; n_points + 1];
    for i in 0..n_points {
        row_ptr[i + 1] = row_ptr[i] + buckets[i].len();
    }
    let total = *row_ptr.last().unwrap();
    let mut col_idx = Vec::with_capacity(total);
    let mut values = Vec::with_capacity(total);
    for bucket in &buckets {
        for (idx, val) in bucket {
            col_idx.push(*idx);
            values.push(*val);
        }
    }

    // Normalize per row
    for i in 0..n_points {
        let (s, e) = (row_ptr[i], row_ptr[i + 1]);
        let ssum: f64 = values[s..e].iter().sum();
        if ssum > 0.0 {
            for v in &mut values[s..e] {
                *v /= ssum;
            }
        }
    }

    Lut1d {
        row_ptr,
        col_idx,
        values: Some(values),
        hist_min: min,
        hist_step: step,
    }
}

/// Pixel-split CSR builders using full pixel splitting with unified LUT types.
///
/// Uses pixel corner coordinates from `Geometry` to compute bin overlaps.
/// The `radial` array contains pixel center values in the selected radial unit.
/// The `unit` parameter specifies the radial unit for proper corner conversion.
///
/// Each pixel is weighted by the fraction of its own (radial, azimuth)
/// quadrilateral that falls into a bin; each bin (row) is then normalized to
/// unit sum. Pixels containing or touching the point of normal incidence use
/// detector-plane sampling instead of a polar quadrilateral.
///
/// Example (1D)
/// ```
/// use dioptrin::geometry::Geometry;
/// use dioptrin::poni::Poni;
/// use dioptrin::integration::RadialUnit;
/// use dioptrin::lut::{build_lut1d_ps_csr, integrate1d};
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 100.0;
/// let geometry = Geometry::new(2, 2, poni);
/// let tth = geometry.calc_tth_all();
/// let intensity = vec![1.0; geometry.size()];
/// let lut = build_lut1d_ps_csr(&geometry, &tth, 4, RadialUnit::TthRad);
/// let norm = vec![1.0; geometry.size()];
/// let (_bins, y) = integrate1d(&intensity, &lut, None, &norm);
/// assert_eq!(y.len(), 4);
/// ```
pub fn build_lut1d_ps_csr(
    geometry: &Geometry,
    radial: &[f64],
    n_points: usize,
    unit: RadialUnit,
) -> Lut1d {
    assert_eq!(radial.len(), geometry.size());

    let wavelength = geometry.poni.wavelength;

    let (tth_corners, azimuthal_corners) = geometry.angular_corners();
    let radial_corners = unit.convert_from_tth(&tth_corners, wavelength);

    let min = math::min(&radial_corners);
    let mut max = math::max(&radial_corners);
    max += 10.0 * f64::EPSILON;
    let step = (max - min) / n_points as f64;

    let buckets = lut1d_ps_fractions(
        geometry,
        radial,
        n_points,
        unit,
        min,
        step,
        &radial_corners,
        &azimuthal_corners,
    );
    lut1d_from_fractions(buckets, min, step)
}

/// Build a pixel-splitting 1D CSR LUT with explicit radial range.
///
/// Weights are per-pixel area fractions as in [`build_lut1d_ps_csr`]; a pixel
/// partly outside `[radial_min, radial_max)` contributes only its in-range
/// fraction (it is not renormalized to one) before the per-bin normalization.
pub fn build_lut1d_ps_csr_with_range(
    geometry: &Geometry,
    radial: &[f64],
    n_points: usize,
    unit: RadialUnit,
    radial_min: f64,
    radial_max: f64,
) -> Lut1d {
    assert_eq!(radial.len(), geometry.size());

    let wavelength = geometry.poni.wavelength;

    let (tth_corners, azimuthal_corners) = geometry.angular_corners();
    let radial_corners = unit.convert_from_tth(&tth_corners, wavelength);

    let (min, _max, step) = histogram::min_max_step_from_range(radial_min, radial_max, n_points);

    let buckets = lut1d_ps_fractions(
        geometry,
        radial,
        n_points,
        unit,
        min,
        step,
        &radial_corners,
        &azimuthal_corners,
    );
    lut1d_from_fractions(buckets, min, step)
}

fn build_lut2d_ps_csr_with_azimuthal(
    geometry: &Geometry,
    radial: &[f64],
    azimuthal: &[f64],
    n_radial: usize,
    n_azimuthal: usize,
    azimuthal_range: Option<(f64, f64)>,
    unit: RadialUnit,
) -> Lut2d {
    assert_eq!(radial.len(), geometry.size());
    assert_eq!(azimuthal.len(), geometry.size());

    let wavelength = geometry.poni.wavelength;

    let (tth_corners, azimuthal_corners) = geometry.angular_corners();
    let radial_corners = unit.convert_from_tth(&tth_corners, wavelength);

    let min_radial = math::min(&radial_corners);
    let mut max_radial = math::max(&radial_corners);
    max_radial += 10.0 * f64::EPSILON;
    let step_radial = (max_radial - min_radial) / n_radial as f64;

    let (min_azimuthal, step_azimuthal) = match azimuthal_range {
        Some((start, step)) => {
            assert!(step > 0.0, "azimuthal_step must be positive");
            (start - 0.5 * step, step)
        }
        None => {
            let min_azimuthal = math::min(&azimuthal_corners);
            let mut max_azimuthal = math::max(&azimuthal_corners);
            max_azimuthal += 10.0 * f64::EPSILON;
            let step_azimuthal = (max_azimuthal - min_azimuthal) / n_azimuthal as f64;
            (min_azimuthal, step_azimuthal)
        }
    };

    let rows = n_radial * n_azimuthal;

    let counts: Vec<AtomicUsize> = (0..rows).map(|_| AtomicUsize::new(0)).collect();
    (0..radial.len()).into_par_iter().for_each_init(
        || (Vec::with_capacity(8), Vec::with_capacity(8)),
        |(buf1, buf2), idx| {
            let [i00, i01, i11, i10] = geometry.corner_indices(idx);

            let r00 = radial_corners[i00];
            let r01 = radial_corners[i01];
            let r10 = radial_corners[i10];
            let r11 = radial_corners[i11];

            let center_azimuthal = azimuthal[idx];
            let a00 = unwrap_angle(azimuthal_corners[i00], center_azimuthal);
            let a01 = unwrap_angle(azimuthal_corners[i01], center_azimuthal);
            let a10 = unwrap_angle(azimuthal_corners[i10], center_azimuthal);
            let a11 = unwrap_angle(azimuthal_corners[i11], center_azimuthal);

            let poly = [(r00, a00), (r01, a01), (r11, a11), (r10, a10)];
            let area_total = polygon_area(&poly).abs();
            if area_total <= 0.0 {
                let rb = histogram::get_bin_clamped(radial[idx], min_radial, step_radial, n_radial);
                let ab = histogram::get_bin_wrapped(
                    center_azimuthal,
                    min_azimuthal,
                    step_azimuthal,
                    n_azimuthal,
                );
                counts[ab * n_radial + rb].fetch_add(1, Ordering::Relaxed);
                return;
            }

            let mut rmin = r00;
            let mut rmax = r00;
            for v in [r01, r10, r11] {
                if v < rmin {
                    rmin = v;
                }
                if v > rmax {
                    rmax = v;
                }
            }
            let mut amin = a00;
            let mut amax = a00;
            for v in [a01, a10, a11] {
                if v < amin {
                    amin = v;
                }
                if v > amax {
                    amax = v;
                }
            }

            let mut r_start = ((rmin - min_radial) / step_radial).floor() as isize;
            let mut r_end = ((rmax - min_radial) / step_radial).floor() as isize;
            if r_end < 0 || r_start >= n_radial as isize {
                return;
            }
            if r_start < 0 {
                r_start = 0;
            }
            if r_end >= n_radial as isize {
                r_end = n_radial as isize - 1;
            }

            let a_start = ((amin - min_azimuthal) / step_azimuthal).floor() as isize;
            let a_end = ((amax - min_azimuthal) / step_azimuthal).floor() as isize;
            if a_start > a_end {
                return;
            }

            for br in r_start..=r_end {
                let r0 = min_radial + br as f64 * step_radial;
                let r1 = r0 + step_radial;
                for ba in a_start..=a_end {
                    let a0 = min_azimuthal + ba as f64 * step_azimuthal;
                    let a1 = a0 + step_azimuthal;
                    let clipped = clip_polygon_rect_scratch(&poly, r0, r1, a0, a1, buf1, buf2);
                    let area = polygon_area(clipped).abs();
                    if area <= 0.0 {
                        continue;
                    }
                    let row_azimuthal =
                        ((ba % n_azimuthal as isize) + n_azimuthal as isize) % n_azimuthal as isize;
                    let row = row_azimuthal as usize * n_radial + br as usize;
                    counts[row].fetch_add(1, Ordering::Relaxed);
                }
            }
        },
    );

    let mut row_ptr = vec![0usize; rows + 1];
    for r in 0..rows {
        row_ptr[r + 1] = row_ptr[r] + counts[r].load(Ordering::Relaxed);
    }
    let total = *row_ptr.last().unwrap();
    let col_idx: Vec<AtomicUsize> = (0..total).map(|_| AtomicUsize::new(0)).collect();
    let values: Vec<AtomicU64> = (0..total).map(|_| AtomicU64::new(0)).collect();
    let next: Vec<AtomicUsize> = row_ptr.iter().map(|&v| AtomicUsize::new(v)).collect();

    (0..radial.len()).into_par_iter().for_each_init(
        || (Vec::with_capacity(8), Vec::with_capacity(8)),
        |(buf1, buf2), idx| {
            let [i00, i01, i11, i10] = geometry.corner_indices(idx);

            let r00 = radial_corners[i00];
            let r01 = radial_corners[i01];
            let r10 = radial_corners[i10];
            let r11 = radial_corners[i11];

            let center_azimuthal = azimuthal[idx];
            let a00 = unwrap_angle(azimuthal_corners[i00], center_azimuthal);
            let a01 = unwrap_angle(azimuthal_corners[i01], center_azimuthal);
            let a10 = unwrap_angle(azimuthal_corners[i10], center_azimuthal);
            let a11 = unwrap_angle(azimuthal_corners[i11], center_azimuthal);

            let poly = [(r00, a00), (r01, a01), (r11, a11), (r10, a10)];
            let area_total = polygon_area(&poly).abs();
            if area_total <= 0.0 {
                let rb = histogram::get_bin_clamped(radial[idx], min_radial, step_radial, n_radial);
                let ab = histogram::get_bin_wrapped(
                    center_azimuthal,
                    min_azimuthal,
                    step_azimuthal,
                    n_azimuthal,
                );
                let row = ab * n_radial + rb;
                let p = next[row].fetch_add(1, Ordering::Relaxed);
                col_idx[p].store(idx, Ordering::Relaxed);
                values[p].store(1.0f64.to_bits(), Ordering::Relaxed);
                return;
            }

            let mut rmin = r00;
            let mut rmax = r00;
            for v in [r01, r10, r11] {
                if v < rmin {
                    rmin = v;
                }
                if v > rmax {
                    rmax = v;
                }
            }
            let mut amin = a00;
            let mut amax = a00;
            for v in [a01, a10, a11] {
                if v < amin {
                    amin = v;
                }
                if v > amax {
                    amax = v;
                }
            }

            let mut r_start = ((rmin - min_radial) / step_radial).floor() as isize;
            let mut r_end = ((rmax - min_radial) / step_radial).floor() as isize;
            if r_end < 0 || r_start >= n_radial as isize {
                return;
            }
            if r_start < 0 {
                r_start = 0;
            }
            if r_end >= n_radial as isize {
                r_end = n_radial as isize - 1;
            }

            let a_start = ((amin - min_azimuthal) / step_azimuthal).floor() as isize;
            let a_end = ((amax - min_azimuthal) / step_azimuthal).floor() as isize;
            if a_start > a_end {
                return;
            }

            for br in r_start..=r_end {
                let r0 = min_radial + br as f64 * step_radial;
                let r1 = r0 + step_radial;
                for ba in a_start..=a_end {
                    let a0 = min_azimuthal + ba as f64 * step_azimuthal;
                    let a1 = a0 + step_azimuthal;
                    let clipped = clip_polygon_rect_scratch(&poly, r0, r1, a0, a1, buf1, buf2);
                    let area = polygon_area(clipped).abs();
                    if area <= 0.0 {
                        continue;
                    }
                    let row_azimuthal =
                        ((ba % n_azimuthal as isize) + n_azimuthal as isize) % n_azimuthal as isize;
                    let row = row_azimuthal as usize * n_radial + br as usize;
                    let p = next[row].fetch_add(1, Ordering::Relaxed);
                    col_idx[p].store(idx, Ordering::Relaxed);
                    values[p].store(area.to_bits(), Ordering::Relaxed);
                }
            }
        },
    );

    let col_idx: Vec<usize> = col_idx
        .into_iter()
        .map(|v| v.load(Ordering::Relaxed))
        .collect();
    let mut values: Vec<f64> = values
        .into_iter()
        .map(|v| f64::from_bits(v.load(Ordering::Relaxed)))
        .collect();

    // Normalize per row
    for r in 0..rows {
        let (s, e) = (row_ptr[r], row_ptr[r + 1]);
        let ssum: f64 = values[s..e].iter().sum();
        if ssum > 0.0 {
            for v in &mut values[s..e] {
                *v /= ssum;
            }
        }
    }

    Lut2d {
        row_ptr,
        col_idx,
        values: Some(values),
        n_tth: n_radial,
        n_azimuthal,
    }
}

/// Build a 2D pixel-split LUT with full polygon splitting over radial/azimuthal bins.
///
/// This clips each pixel quad in (radial, azimuthal) space against bin rectangles and
/// accumulates area-based weights.
///
/// The `radial` array contains pixel center values in the selected radial unit.
/// The `unit` parameter specifies the radial unit for proper corner conversion.
///
/// Example (2D)
/// ```
/// use dioptrin::geometry::Geometry;
/// use dioptrin::poni::Poni;
/// use dioptrin::integration::RadialUnit;
/// use dioptrin::lut::{build_lut2d_ps_csr, integrate2d};
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 100.0;
/// let geometry = Geometry::new(2, 2, poni);
/// let tth = geometry.calc_tth_all();
/// let azimuthal = geometry.calc_azimuth_all();
/// let intensity = vec![1.0; geometry.size()];
/// let lut2d = build_lut2d_ps_csr(&geometry, &tth, &azimuthal, 4, 4, RadialUnit::TthRad);
/// let norm = vec![1.0; geometry.size()];
/// let y = integrate2d(&intensity, &lut2d, None, &norm);
/// assert_eq!(y.len(), 4 * 4);
/// ```
pub fn build_lut2d_ps_csr(
    geometry: &Geometry,
    radial: &[f64],
    azimuthal: &[f64],
    n_radial: usize,
    n_azimuthal: usize,
    unit: RadialUnit,
) -> Lut2d {
    build_lut2d_ps_csr_with_azimuthal(
        geometry,
        radial,
        azimuthal,
        n_radial,
        n_azimuthal,
        None,
        unit,
    )
}

/// Build a 2D pixel-split LUT using an explicit azimuthal binning definition.
///
/// `azimuthal_start` is the center of the first azimuthal bin; the effective start edge is
/// `azimuthal_start - azimuthal_step / 2`. Binning wraps periodically so angles outside the
/// total range fold back into `0..n_azimuthal`.
#[allow(clippy::too_many_arguments)]
pub fn build_lut2d_ps_csr_with_azimuthal_range(
    geometry: &Geometry,
    radial: &[f64],
    azimuthal: &[f64],
    n_radial: usize,
    n_azimuthal: usize,
    azimuthal_start: f64,
    azimuthal_step: f64,
    unit: RadialUnit,
) -> Lut2d {
    build_lut2d_ps_csr_with_azimuthal(
        geometry,
        radial,
        azimuthal,
        n_radial,
        n_azimuthal,
        Some((azimuthal_start, azimuthal_step)),
        unit,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Geometry;
    use crate::poni::Poni;

    fn create_test_geometry() -> Geometry {
        let mut poni = Poni::new();
        poni.pixel1 = 0.0001; // 100 µm
        poni.pixel2 = 0.0001;
        poni.distance = 0.1; // 10 cm
        poni.wavelength = 1e-10; // 1 Angstrom
        Geometry::new(10, 10, poni)
    }

    #[test]
    fn test_build_lut1d_ps_csr_tth_rad() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let lut = build_lut1d_ps_csr(&geometry, &tth, 5, RadialUnit::TthRad);

        assert_eq!(lut.n_points(), 5);
        assert!(lut.hist_min >= 0.0);
        assert!(lut.hist_step > 0.0);
    }

    #[test]
    fn test_build_lut1d_ps_csr_tth_deg() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let radial_deg = RadialUnit::TthDeg.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut1d_ps_csr(&geometry, &radial_deg, 5, RadialUnit::TthDeg);

        assert_eq!(lut.n_points(), 5);
        // Degrees should be larger than radians
        assert!(lut.hist_min >= 0.0);
        // hist_min in degrees should be roughly 57x larger than radians (180/π)
    }

    #[test]
    fn test_build_lut1d_ps_csr_q_angstrom() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let radial_q = RadialUnit::QAngstrom.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut1d_ps_csr(&geometry, &radial_q, 5, RadialUnit::QAngstrom);

        assert_eq!(lut.n_points(), 5);
        assert!(lut.hist_min >= 0.0, "q should be non-negative");
    }

    #[test]
    fn test_build_lut1d_ps_csr_q_nm() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let radial_q = RadialUnit::QNm.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut1d_ps_csr(&geometry, &radial_q, 5, RadialUnit::QNm);

        assert_eq!(lut.n_points(), 5);
        assert!(lut.hist_min >= 0.0, "q should be non-negative");
    }

    #[test]
    fn test_build_lut1d_ps_csr_different_units_different_bins() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();

        // Build LUT in radians
        let lut_rad = build_lut1d_ps_csr(&geometry, &tth, 5, RadialUnit::TthRad);

        // Build LUT in degrees
        let radial_deg = RadialUnit::TthDeg.convert_from_tth(&tth, geometry.poni.wavelength);
        let lut_deg = build_lut1d_ps_csr(&geometry, &radial_deg, 5, RadialUnit::TthDeg);

        // The hist_step should differ by factor of 180/π
        // (hist_min might be very close to zero, so compare steps instead)
        let ratio = lut_deg.hist_step / lut_rad.hist_step;
        let expected_ratio = 180.0 / std::f64::consts::PI;
        assert!(
            (ratio - expected_ratio).abs() / expected_ratio < 0.01,
            "Degree step should be 180/π times radian step, got ratio {}",
            ratio
        );
    }

    #[test]
    fn test_build_lut2d_ps_csr_with_q_unit() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let azimuthal = geometry.calc_azimuth_all();
        let radial_q = RadialUnit::QAngstrom.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut2d_ps_csr(
            &geometry,
            &radial_q,
            &azimuthal,
            5,
            4,
            RadialUnit::QAngstrom,
        );

        assert_eq!(lut.n_tth, 5);
        assert_eq!(lut.n_azimuthal, 4);
        assert_eq!(lut.row_ptr.len(), 5 * 4 + 1);
    }

    #[test]
    fn test_lut_integration_same_total_intensity_different_units() {
        use crate::lut::integrate1d;

        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let intensity: Vec<f64> = (0..geometry.size()).map(|i| (i + 1) as f64).collect();

        // Build and integrate in radians
        let lut_rad = build_lut1d_ps_csr(&geometry, &tth, 5, RadialUnit::TthRad);
        let norm = vec![1.0; geometry.size()];
        let (_, int_rad) = integrate1d(&intensity, &lut_rad, None, &norm);

        // Build and integrate in q
        let radial_q = RadialUnit::QAngstrom.convert_from_tth(&tth, geometry.poni.wavelength);
        let lut_q = build_lut1d_ps_csr(&geometry, &radial_q, 5, RadialUnit::QAngstrom);
        let (_, int_q) = integrate1d(&intensity, &lut_q, None, &norm);

        // Total integrated intensity should be similar (not exactly equal due to different binning)
        let total_rad: f64 = int_rad.iter().sum();
        let total_q: f64 = int_q.iter().sum();

        // They should be in the same ballpark (within 20%)
        let rel_diff = (total_rad - total_q).abs() / total_rad;
        assert!(
            rel_diff < 0.2,
            "Total intensity should be similar across units, got {}% difference",
            rel_diff * 100.0
        );
    }

    /// One row of pixels that are long along the radial direction and only 1 µm
    /// wide, starting `offset` pixels away from the point of normal incidence.
    /// Their true bin fractions are plain length fractions along the row.
    fn radial_strip(n: usize, offset: f64, distance: f64) -> Geometry {
        let mut poni = Poni::new();
        poni.pixel1 = 1e-6;
        poni.pixel2 = 1e-4;
        poni.distance = distance;
        poni.poni1 = 0.5e-6;
        poni.poni2 = -offset * 1e-4;
        poni.wavelength = 1e-10;
        Geometry::new(1, n, poni)
    }

    fn corners(geometry: &Geometry) -> (Vec<f64>, Vec<f64>) {
        geometry.angular_corners()
    }

    fn pixel_radial_extent(geometry: &Geometry, pixel: usize) -> (f64, f64) {
        let (tth, _) = corners(geometry);
        let values = geometry.corner_indices(pixel).map(|i| tth[i]);
        (
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    }

    fn fractions(geometry: &Geometry, n: usize, min: f64, step: f64) -> Vec<Vec<(usize, f64)>> {
        let (tth, azimuth) = corners(geometry);
        let radial = geometry.calc_tth_all();
        lut1d_ps_fractions(
            geometry,
            &radial,
            n,
            RadialUnit::TthRad,
            min,
            step,
            &tth,
            &azimuth,
        )
    }

    fn pixel_total(buckets: &[Vec<(usize, f64)>], pixel: usize) -> f64 {
        buckets
            .iter()
            .flatten()
            .filter(|(p, _)| *p == pixel)
            .map(|(_, w)| w)
            .sum()
    }

    #[test]
    fn split_pixel_gets_fractions_of_its_own_area_at_any_radius() {
        // The same 30/70 split 300 and 3000 pixels from the beam: the polar
        // quadrilateral areas differ by an order of magnitude, the fractions
        // must not.
        for (offset, distance) in [(300.0, 1.0), (3000.0, 10.0)] {
            let geometry = radial_strip(1, offset, distance);
            let (lo, hi) = pixel_radial_extent(&geometry, 0);
            let step = hi - lo;
            let buckets = fractions(&geometry, 2, lo - 0.7 * step, step);
            assert_eq!(buckets[0].len(), 1);
            assert_eq!(buckets[1].len(), 1);
            let (w0, w1) = (buckets[0][0].1, buckets[1][0].1);
            assert!((w0 - 0.3).abs() < 1e-3, "offset {offset}: {w0}");
            assert!((w1 - 0.7).abs() < 1e-3, "offset {offset}: {w1}");
            assert!((w0 + w1 - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn fully_included_pixels_have_unit_total_weight_before_bin_normalization() {
        let mut poni = Poni::new();
        poni.pixel1 = 1.7e-4;
        poni.pixel2 = 2.1e-4;
        poni.distance = 0.14;
        poni.poni1 = -0.003;
        poni.poni2 = -0.002;
        poni.rot1 = 0.007;
        poni.rot2 = -0.009;
        poni.wavelength = 1e-10;
        let geometry = Geometry::new(23, 29, poni);
        let (tth, _) = corners(&geometry);
        let lo = math::min(&tth);
        let hi = math::max(&tth) + 10.0 * f64::EPSILON;
        let buckets = fractions(&geometry, 40, lo, (hi - lo) / 40.0);
        for pixel in 0..geometry.size() {
            let total = pixel_total(&buckets, pixel);
            assert!((total - 1.0).abs() < 1e-9, "pixel {pixel}: {total}");
        }
    }

    #[test]
    fn range_clipped_pixel_keeps_only_its_in_range_fraction() {
        let geometry = radial_strip(1, 300.0, 1.0);
        let (lo, hi) = pixel_radial_extent(&geometry, 0);
        // The radial range ends 40% into the pixel.
        let max = lo + 0.4 * (hi - lo);
        let min = lo - (max - lo);
        let buckets = fractions(&geometry, 4, min, (max - min) / 4.0);
        let total = pixel_total(&buckets, 0);
        assert!((total - 0.4).abs() < 1e-3, "{total}");

        // The public builder normalizes the bin but must not hide the pixel.
        let tth = geometry.calc_tth_all();
        let lut = build_lut1d_ps_csr_with_range(&geometry, &tth, 4, RadialUnit::TthRad, min, max);
        assert!(lut.col_idx.iter().all(|&pixel| pixel == 0));
        assert_eq!(lut.row_ptr[4] - lut.row_ptr[3], 1);
        assert_eq!(lut.values.unwrap()[lut.row_ptr[3]], 1.0);
    }

    #[test]
    fn bin_weights_follow_pixel_fractions_not_polar_area() {
        // Pixels 1..=3 of the strip span 11-12, 12-13 and 13-14 pixel lengths
        // from the beam. One bin covers the outer half of the first, all of the
        // second and the inner half of the third: weights 1/4, 1/2, 1/4.
        // Polar-area weights (~1/r) would give 0.267, 0.499, 0.234.
        let geometry = radial_strip(5, 10.0, 0.1);
        let (lo1, hi1) = pixel_radial_extent(&geometry, 1);
        let (lo3, hi3) = pixel_radial_extent(&geometry, 3);
        let min = 0.5 * (lo1 + hi1);
        let max = 0.5 * (lo3 + hi3);
        let tth = geometry.calc_tth_all();
        let lut = build_lut1d_ps_csr_with_range(&geometry, &tth, 1, RadialUnit::TthRad, min, max);
        assert_eq!(lut.col_idx, vec![1, 2, 3]);
        let values = lut.values.unwrap();
        for (value, expected) in values.iter().zip([0.25, 0.5, 0.25]) {
            assert!((value - expected).abs() < 0.01, "{values:?}");
        }
        assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn beam_pixel_uses_detector_plane_fractions() {
        let mut poni = Poni::new();
        poni.pixel1 = 1e-4;
        poni.pixel2 = 1e-4;
        poni.distance = 0.1;
        poni.poni1 = 4.3e-4;
        poni.poni2 = 5.6e-4;
        poni.wavelength = 1e-10;
        let geometry = Geometry::new(9, 11, poni);
        let beam_pixel = 4 * 11 + 5;
        let (tth, _) = corners(&geometry);
        let hi = math::max(&tth) + 10.0 * f64::EPSILON;
        let n = 200;
        let step = hi / n as f64;
        let buckets = fractions(&geometry, n, 0.0, step);
        // Every pixel, including the one containing the beam, is conserved.
        for pixel in 0..geometry.size() {
            let total = pixel_total(&buckets, pixel);
            assert!((total - 1.0).abs() < 1e-9, "pixel {pixel}: {total}");
        }
        // The first bin is a disc of radius distance·tan(step) inside the beam
        // pixel; no polar quadrilateral of that pixel reaches it.
        let radius = 0.1 * step.tan();
        let expected = PI * radius * radius / 1e-8;
        assert_eq!(buckets[0].len(), 1);
        let (pixel, weight) = buckets[0][0];
        assert_eq!(pixel, beam_pixel);
        assert!(
            (weight / expected - 1.0).abs() < 0.05,
            "{weight} {expected}"
        );
    }
}
