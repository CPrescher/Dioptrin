//! Pixel-splitting CSR LUT builders (1D/2D) producing weighted `Lut1d`/`Lut2d`.
//! Integrate with `crate::lut::integrate1d` / `integrate2d`.
use rayon::prelude::*;
use std::f64::consts::PI;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::geometry::Geometry;
use crate::lut::counting_csr::{Lut1d, Lut2d};
use crate::math::{self, histogram};

/// Compute the signed area of a polygon given as (x, y) vertices.
fn polygon_area(poly: &[(f64, f64)]) -> f64 {
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
fn clip_polygon_rect_scratch<'a>(
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

/// Pixel-split CSR builders using full pixel splitting  with unified LUT types.
///
/// Uses pixel corner coordinates from `Geometry` to compute bin overlaps.
///
/// Example (1D)
/// ```
/// use dioptrin::geometry::Geometry;
/// use dioptrin::poni::Poni;
/// use dioptrin::lut::{build_lut1d_ps_csr, integrate1d};
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 100.0;
/// let geometry = Geometry::new(2, 2, poni);
/// let tth = geometry.calc_tth_all();
/// let intensity = vec![1.0; geometry.size()];
/// let lut = build_lut1d_ps_csr(&geometry, &tth, 4);
/// let (_bins, y) = integrate1d(&intensity, &lut, None);
/// assert_eq!(y.len(), 4);
/// ```
pub fn build_lut1d_ps_csr(geometry: &Geometry, tth: &Vec<f64>, n_points: usize) -> Lut1d {
    assert_eq!(tth.len(), geometry.size());

    let (d1, d2) = geometry.corner_grid();
    let (tth_corners, chi_corners) = geometry.calc_tth_and_chi(&d1, &d2);
    let chi = geometry.calc_chi_all();

    let min = math::min(&tth_corners);
    let mut max = math::max(&tth_corners);
    max += 10.0 * f64::EPSILON;
    let step = (max - min) / n_points as f64;

    let dim2 = geometry.dim2;
    let width = dim2 + 1;

    // Single pass: build per-row buckets in parallel, then flatten to CSR.
    let (buckets, _, _) = tth
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
            |(mut acc, mut buf1, mut buf2), (idx, &center_tth)| {
                let row = idx / dim2;
                let col = idx - row * dim2;
                let base = row * width + col;

                let t00 = tth_corners[base];
                let t01 = tth_corners[base + 1];
                let t10 = tth_corners[base + width];
                let t11 = tth_corners[base + width + 1];

                let center_chi = chi[idx];
                let a00 = unwrap_angle(chi_corners[base], center_chi);
                let a01 = unwrap_angle(chi_corners[base + 1], center_chi);
                let a10 = unwrap_angle(chi_corners[base + width], center_chi);
                let a11 = unwrap_angle(chi_corners[base + width + 1], center_chi);

                let poly = [(t00, a00), (t01, a01), (t11, a11), (t10, a10)];
                let area_total = polygon_area(&poly).abs();
                if area_total <= 0.0 {
                    let pos = (center_tth - min) / step;
                    if pos < 0.0 {
                        return (acc, buf1, buf2);
                    }
                    let bin = pos.floor() as isize;
                    if bin >= 0 && (bin as usize) < n_points {
                        acc[bin as usize].push((idx, 1.0));
                    }
                    return (acc, buf1, buf2);
                }

                let mut tmin = t00;
                let mut tmax = t00;
                for v in [t01, t10, t11] {
                    if v < tmin {
                        tmin = v;
                    }
                    if v > tmax {
                        tmax = v;
                    }
                }

                let mut start = ((tmin - min) / step).floor() as isize;
                let mut end = ((tmax - min) / step).floor() as isize;
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
                    let area = polygon_area(&clipped).abs();
                    if area > 0.0 {
                        acc[b as usize].push((idx, area));
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

    // Prefix sum
    let mut row_ptr = vec![0usize; n_points + 1];
    for i in 0..n_points {
        row_ptr[i + 1] = row_ptr[i] + buckets[i].len();
    }
    let total = *row_ptr.last().unwrap();
    let mut col_idx = Vec::with_capacity(total);
    let mut values = Vec::with_capacity(total);
    for row in 0..n_points {
        for (idx, val) in &buckets[row] {
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

/// Build a 2D pixel-split LUT with full polygon splitting over `tth`/`azi` bins.
///
/// This clips each pixel quad in (tth, azi) space against bin rectangles and
/// accumulates area-based weights.
///
/// Example (2D)
/// ```
/// use dioptrin::geometry::Geometry;
/// use dioptrin::poni::Poni;
/// use dioptrin::lut::{build_lut2d_ps_csr, integrate2d};
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 100.0;
/// let geometry = Geometry::new(2, 2, poni);
/// let tth = geometry.calc_tth_all();
/// let chi = geometry.calc_chi_all();
/// let intensity = vec![1.0; geometry.size()];
/// let lut2d = build_lut2d_ps_csr(&geometry, &tth, &chi, 4, 4);
/// let y = integrate2d(&intensity, &lut2d, None);
/// assert_eq!(y.len(), 4 * 4);
/// ```
pub fn build_lut2d_ps_csr(
    geometry: &Geometry,
    tth: &Vec<f64>,
    azi: &Vec<f64>,
    n_tth: usize,
    n_azi: usize,
) -> Lut2d {
    assert_eq!(tth.len(), geometry.size());
    assert_eq!(azi.len(), geometry.size());

    let (d1, d2) = geometry.corner_grid();
    let (tth_corners, azi_corners) = geometry.calc_tth_and_chi(&d1, &d2);

    let min_tth = math::min(&tth_corners);
    let mut max_tth = math::max(&tth_corners);
    max_tth += 10.0 * f64::EPSILON;
    let step_tth = (max_tth - min_tth) / n_tth as f64;

    let min_azi = math::min(&azi_corners);
    let mut max_azi = math::max(&azi_corners);
    max_azi += 10.0 * f64::EPSILON;
    let step_azi = (max_azi - min_azi) / n_azi as f64;

    let rows = n_tth * n_azi;
    let dim2 = geometry.dim2;
    let width = dim2 + 1;

    let counts: Vec<AtomicUsize> = (0..rows).map(|_| AtomicUsize::new(0)).collect();
    (0..tth.len()).into_par_iter().for_each_init(
        || (Vec::with_capacity(8), Vec::with_capacity(8)),
        |(buf1, buf2), idx| {
            let row = idx / dim2;
            let col = idx - row * dim2;
            let base = row * width + col;

            let t00 = tth_corners[base];
            let t01 = tth_corners[base + 1];
            let t10 = tth_corners[base + width];
            let t11 = tth_corners[base + width + 1];

            let center_azi = azi[idx];
            let a00 = unwrap_angle(azi_corners[base], center_azi);
            let a01 = unwrap_angle(azi_corners[base + 1], center_azi);
            let a10 = unwrap_angle(azi_corners[base + width], center_azi);
            let a11 = unwrap_angle(azi_corners[base + width + 1], center_azi);

            let poly = [(t00, a00), (t01, a01), (t11, a11), (t10, a10)];
            let area_total = polygon_area(&poly).abs();
            if area_total <= 0.0 {
                let tb = histogram::get_bin(tth[idx], min_tth, step_tth);
                let ab = histogram::get_bin(center_azi, min_azi, step_azi) % n_azi;
                counts[ab * n_tth + tb].fetch_add(1, Ordering::Relaxed);
                return;
            }

            let mut tmin = t00;
            let mut tmax = t00;
            for v in [t01, t10, t11] {
                if v < tmin {
                    tmin = v;
                }
                if v > tmax {
                    tmax = v;
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

            let mut t_start = ((tmin - min_tth) / step_tth).floor() as isize;
            let mut t_end = ((tmax - min_tth) / step_tth).floor() as isize;
            if t_end < 0 || t_start >= n_tth as isize {
                return;
            }
            if t_start < 0 {
                t_start = 0;
            }
            if t_end >= n_tth as isize {
                t_end = n_tth as isize - 1;
            }

            let a_start = ((amin - min_azi) / step_azi).floor() as isize;
            let a_end = ((amax - min_azi) / step_azi).floor() as isize;
            if a_start > a_end {
                return;
            }

            for bt in t_start..=t_end {
                let t0 = min_tth + bt as f64 * step_tth;
                let t1 = t0 + step_tth;
                for ba in a_start..=a_end {
                    let a0 = min_azi + ba as f64 * step_azi;
                    let a1 = a0 + step_azi;
                    let clipped = clip_polygon_rect_scratch(&poly, t0, t1, a0, a1, buf1, buf2);
                    let area = polygon_area(&clipped).abs();
                    if area <= 0.0 {
                        continue;
                    }
                    let row_azi = ((ba % n_azi as isize) + n_azi as isize) % n_azi as isize;
                    let row = row_azi as usize * n_tth + bt as usize;
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

    (0..tth.len()).into_par_iter().for_each_init(
        || (Vec::with_capacity(8), Vec::with_capacity(8)),
        |(buf1, buf2), idx| {
            let row = idx / dim2;
            let col = idx - row * dim2;
            let base = row * width + col;

            let t00 = tth_corners[base];
            let t01 = tth_corners[base + 1];
            let t10 = tth_corners[base + width];
            let t11 = tth_corners[base + width + 1];

            let center_azi = azi[idx];
            let a00 = unwrap_angle(azi_corners[base], center_azi);
            let a01 = unwrap_angle(azi_corners[base + 1], center_azi);
            let a10 = unwrap_angle(azi_corners[base + width], center_azi);
            let a11 = unwrap_angle(azi_corners[base + width + 1], center_azi);

            let poly = [(t00, a00), (t01, a01), (t11, a11), (t10, a10)];
            let area_total = polygon_area(&poly).abs();
            if area_total <= 0.0 {
                let tb = histogram::get_bin(tth[idx], min_tth, step_tth);
                let ab = histogram::get_bin(center_azi, min_azi, step_azi) % n_azi;
                let row = ab * n_tth + tb;
                let p = next[row].fetch_add(1, Ordering::Relaxed);
                col_idx[p].store(idx, Ordering::Relaxed);
                values[p].store(1.0f64.to_bits(), Ordering::Relaxed);
                return;
            }

            let mut tmin = t00;
            let mut tmax = t00;
            for v in [t01, t10, t11] {
                if v < tmin {
                    tmin = v;
                }
                if v > tmax {
                    tmax = v;
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

            let mut t_start = ((tmin - min_tth) / step_tth).floor() as isize;
            let mut t_end = ((tmax - min_tth) / step_tth).floor() as isize;
            if t_end < 0 || t_start >= n_tth as isize {
                return;
            }
            if t_start < 0 {
                t_start = 0;
            }
            if t_end >= n_tth as isize {
                t_end = n_tth as isize - 1;
            }

            let a_start = ((amin - min_azi) / step_azi).floor() as isize;
            let a_end = ((amax - min_azi) / step_azi).floor() as isize;
            if a_start > a_end {
                return;
            }

            for bt in t_start..=t_end {
                let t0 = min_tth + bt as f64 * step_tth;
                let t1 = t0 + step_tth;
                for ba in a_start..=a_end {
                    let a0 = min_azi + ba as f64 * step_azi;
                    let a1 = a0 + step_azi;
                    let clipped = clip_polygon_rect_scratch(&poly, t0, t1, a0, a1, buf1, buf2);
                    let area = polygon_area(&clipped).abs();
                    if area <= 0.0 {
                        continue;
                    }
                    let row_azi = ((ba % n_azi as isize) + n_azi as isize) % n_azi as isize;
                    let row = row_azi as usize * n_tth + bt as usize;
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
        n_tth,
        n_azi,
    }
}
