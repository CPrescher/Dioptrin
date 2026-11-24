//! Supersampled CSR LUT builders (1D/2D) producing weighted `Lut1d`/`Lut2d`.
//! Integrate with `crate::lut::integrate1d` / `integrate2d`.
#![allow(clippy::items_after_test_module)]
use rayon::prelude::*;

use crate::geometry::Geometry;
use crate::integration::RadialUnit;
use crate::lut::counting_csr::{Lut1d, Lut2d};
use crate::math::histogram;

/// An exact integer count for one unique pixel/output-bin intersection.
struct Contribution {
    bin: usize,
    pixel: usize,
    count: usize,
}

struct RadialConversion {
    unit: RadialUnit,
    q_scale: f64,
}

impl RadialConversion {
    fn new(unit: RadialUnit, wavelength: f64) -> Self {
        let q_scale = match unit {
            RadialUnit::QAngstrom | RadialUnit::QNm => {
                assert!(
                    wavelength > 0.0,
                    "Wavelength must be positive for q conversion"
                );
                let wavelength = wavelength
                    * if unit == RadialUnit::QAngstrom {
                        1e10
                    } else {
                        1e9
                    };
                4.0 * std::f64::consts::PI / wavelength
            }
            _ => 0.0,
        };
        Self { unit, q_scale }
    }

    fn convert(&self, tth: f64) -> f64 {
        match self.unit {
            RadialUnit::TthRad => tth,
            RadialUnit::TthDeg => tth.to_degrees(),
            RadialUnit::QAngstrom | RadialUnit::QNm => self.q_scale * (tth / 2.0).sin(),
        }
    }
}

/// Count samples per pixel/bin immediately, then transpose the compact records
/// into CSR. Detector rows and pixels remain ordered, so each CSR row has exactly
/// the legacy sorted pixel order, integer totals and floating-point weights.
fn build_counted_csr<F>(
    geometry: &Geometry,
    n_ss: usize,
    n_bins: usize,
    normalize_bins: bool,
    sample_bin: F,
) -> (Vec<usize>, Vec<usize>, Vec<f64>)
where
    F: Fn(f64, f64) -> Option<usize> + Sync,
{
    let rows: Vec<Box<[Contribution]>> = (0..geometry.dim1())
        .into_par_iter()
        .map(|row| {
            let mut entries = Vec::with_capacity(geometry.dim2() * 2);
            let mut counts: Vec<(usize, usize)> = Vec::new();
            for col in 0..geometry.dim2() {
                counts.clear();
                for i in 0..n_ss {
                    let d1 = row as f64 + (i as f64 + 0.5) / n_ss as f64;
                    for j in 0..n_ss {
                        // Preserve subgrid's original global-column arithmetic;
                        // col + (j + .5)/n can round differently near an edge.
                        let d2 = ((col * n_ss + j) as f64 + 0.5) / n_ss as f64;
                        // A sample outside the grid belongs to no bin.
                        let Some(bin) = sample_bin(d1, d2) else {
                            continue;
                        };
                        match counts.iter_mut().find(|(b, _)| *b == bin) {
                            Some((_, count)) => *count += 1,
                            None => counts.push((bin, 1)),
                        }
                    }
                }
                let pixel = row * geometry.dim2() + col;
                entries.extend(counts.iter().map(|&(bin, count)| Contribution {
                    bin,
                    pixel,
                    count,
                }));
            }
            entries.into_boxed_slice()
        })
        .collect();

    let mut row_ptr = vec![0usize; n_bins + 1];
    let mut totals = vec![0usize; n_bins];
    for entries in &rows {
        for entry in entries.iter() {
            row_ptr[entry.bin + 1] += 1;
            totals[entry.bin] += entry.count;
        }
    }
    for bin in 0..n_bins {
        row_ptr[bin + 1] += row_ptr[bin];
    }
    let mut col_idx = vec![0usize; row_ptr[n_bins]];
    let mut values = vec![0.0; row_ptr[n_bins]];
    let mut next = row_ptr[..n_bins].to_vec();
    // Consume/free row buffers during assembly. No global record concatenation,
    // sort, unsafe parallel writes, or duplicated subpixel events are needed.
    for entries in rows {
        for entry in entries.iter() {
            let slot = next[entry.bin];
            col_idx[slot] = entry.pixel;
            values[slot] = entry.count as f64;
            next[entry.bin] += 1;
        }
    }
    for bin in 0..n_bins {
        if totals[bin] != 0 {
            // Per-bin totals for the 2D tables; the pixel's own sample count for
            // 1D, so that a weight is the fraction of the pixel in the bin.
            let denom = if normalize_bins {
                totals[bin] as f64
            } else {
                (n_ss * n_ss) as f64
            };
            for value in &mut values[row_ptr[bin]..row_ptr[bin + 1]] {
                *value /= denom;
            }
        }
    }
    (row_ptr, col_idx, values)
}

/// Precomputed `cos(2θ) * |cos(2θ)|` thresholds for every bin edge `0..=n_bins`.
/// Monotonically non-increasing with bin index.
fn radial_edge_cos2(
    hist_min: f64,
    hist_step: f64,
    n_bins: usize,
    conversion: &RadialConversion,
) -> Vec<f64> {
    (0..=n_bins)
        .map(|b| {
            let r_edge = hist_min + (b as f64) * hist_step;
            let tth = match conversion.unit {
                RadialUnit::TthRad => r_edge,
                RadialUnit::TthDeg => r_edge.to_radians(),
                RadialUnit::QAngstrom | RadialUnit::QNm => {
                    let s = r_edge / conversion.q_scale;
                    if s <= 0.0 {
                        -1.0
                    } else if s >= 1.0 {
                        std::f64::consts::PI + 1.0
                    } else {
                        2.0 * s.asin()
                    }
                }
            };
            if tth <= 0.0 {
                2.0
            } else if tth >= std::f64::consts::PI {
                -2.0
            } else {
                let c = tth.cos();
                c * c.abs()
            }
        })
        .collect()
}

/// Sign of `d(tan²2θ)/ds` at point `p` along direction `d` (valid for `p[2] > 0`).
#[inline(always)]
fn edge_grad_sign(p: [f64; 3], d: [f64; 3]) -> f64 {
    (p[0] * d[0] + p[1] * d[1]) * p[2] - (p[0] * p[0] + p[1] * p[1]) * d[2]
}

/// Finds bin index `b` in `0..=n_bins` for `u` using a local `hint` first.
/// Unconditionally returns `0` if `u > c_edges[0]` and `n_bins` if
/// `u <= c_edges[n_bins]`.
#[inline(always)]
fn find_bin_hint(c_edges: &[f64], u: f64, hint: usize) -> usize {
    let n_bins = c_edges.len() - 1;
    let mut b = hint.min(n_bins.saturating_sub(1));
    if u > c_edges[b] {
        while b > 0 && u > c_edges[b] {
            b -= 1;
        }
        return b;
    }
    if u <= c_edges[b + 1] {
        // Walk forward a few steps before falling back to binary search.
        let limit = (b + 4).min(n_bins);
        while b < limit && u <= c_edges[b + 1] {
            b += 1;
            if b == n_bins {
                return n_bins;
            }
        }
        if b < n_bins && u <= c_edges[b + 1] {
            // `c_edges` is non-increasing: find first `i` where `c_edges[i] < u`.
            let pos = c_edges.partition_point(|&ce| ce >= u);
            return pos.saturating_sub(1).min(n_bins);
        }
    }
    b
}

/// Fast 1D supersampled CSR builder using affine corner monotonicity and
/// `cos²(2θ)` bin-edge thresholds for flat detectors, falling back to exact
/// per-subpixel evaluation near boundaries or for distorted detectors.
#[allow(clippy::too_many_arguments)]
fn build_counted_1d_csr<F>(
    geometry: &Geometry,
    n_ss: usize,
    n_bins: usize,
    hist_min: f64,
    hist_step: f64,
    clamped: bool,
    conversion: &RadialConversion,
    sample_bin: F,
) -> (Vec<usize>, Vec<usize>, Vec<f64>)
where
    F: Fn(f64, f64) -> Option<usize> + Sync,
{
    let transform = geometry.subpixel_transform();
    let Some((p0, d1_step, d2_step)) = transform.affine_plane() else {
        return build_counted_csr(geometry, n_ss, n_bins, false, sample_bin);
    };
    if n_bins == 0 {
        return build_counted_csr(geometry, n_ss, n_bins, false, sample_bin);
    }

    const MARGIN: f64 = 1e-9;
    let c_edges = radial_edge_cos2(hist_min, hist_step, n_bins, conversion);
    let ss2 = n_ss * n_ss;
    let inv_ss = 1.0 / n_ss as f64;
    let s1 = [
        d1_step[0] * inv_ss,
        d1_step[1] * inv_ss,
        d1_step[2] * inv_ss,
    ];
    let s2 = [
        d2_step[0] * inv_ss,
        d2_step[1] * inv_ss,
        d2_step[2] * inv_ss,
    ];
    let dim2 = geometry.dim2();

    let rows: Vec<Box<[Contribution]>> = (0..geometry.dim1())
        .into_par_iter()
        .map(|row| {
            let mut entries = Vec::with_capacity(dim2 * 2);
            let mut counts: Vec<(usize, usize)> = Vec::new();
            let rf = row as f64;
            let mut p00 = [
                p0[0] + rf * d1_step[0],
                p0[1] + rf * d1_step[1],
                p0[2] + rf * d1_step[2],
            ];
            let mut p10 = [
                p00[0] + d1_step[0],
                p00[1] + d1_step[1],
                p00[2] + d1_step[2],
            ];
            let mut hint = 0usize;

            for col in 0..dim2 {
                let p01 = [
                    p00[0] + d2_step[0],
                    p00[1] + d2_step[1],
                    p00[2] + d2_step[2],
                ];
                let p11 = [
                    p10[0] + d2_step[0],
                    p10[1] + d2_step[1],
                    p10[2] + d2_step[2],
                ];
                let pixel = row * dim2 + col;

                let fast_done = if p00[2] > 0.0 && p01[2] > 0.0 && p10[2] > 0.0 && p11[2] > 0.0 {
                    let mono = edge_grad_sign(p00, d2_step) * edge_grad_sign(p01, d2_step) > 0.0
                        && edge_grad_sign(p10, d2_step) * edge_grad_sign(p11, d2_step) > 0.0
                        && edge_grad_sign(p00, d1_step) * edge_grad_sign(p10, d1_step) > 0.0
                        && edge_grad_sign(p01, d1_step) * edge_grad_sign(p11, d1_step) > 0.0;
                    if mono {
                        let u00 = (p00[2] * p00[2])
                            / (p00[0] * p00[0] + p00[1] * p00[1] + p00[2] * p00[2]);
                        let u01 = (p01[2] * p01[2])
                            / (p01[0] * p01[0] + p01[1] * p01[1] + p01[2] * p01[2]);
                        let u10 = (p10[2] * p10[2])
                            / (p10[0] * p10[0] + p10[1] * p10[1] + p10[2] * p10[2]);
                        let u11 = (p11[2] * p11[2])
                            / (p11[0] * p11[0] + p11[1] * p11[1] + p11[2] * p11[2]);
                        let u_min = u00.min(u01).min(u10.min(u11));
                        let u_max = u00.max(u01).max(u10.max(u11));

                        if !clamped
                            && (u_min > c_edges[0] + MARGIN || u_max < c_edges[n_bins] - MARGIN)
                        {
                            true
                        } else {
                            let mut b_lo = find_bin_hint(&c_edges, u_max, hint);
                            let mut b_hi = find_bin_hint(&c_edges, u_min, b_lo);
                            if clamped {
                                b_lo = b_lo.min(n_bins - 1);
                                b_hi = b_hi.min(n_bins - 1);
                            }
                            hint = b_lo;

                            if b_lo == b_hi && b_lo < n_bins {
                                let lo_ok =
                                    (clamped && b_lo == 0) || u_max + MARGIN < c_edges[b_lo];
                                let hi_ok = (clamped && b_lo == n_bins - 1)
                                    || u_min - MARGIN > c_edges[b_lo + 1];
                                if lo_ok && hi_ok {
                                    entries.push(Contribution {
                                        bin: b_lo,
                                        pixel,
                                        count: ss2,
                                    });
                                    true
                                } else {
                                    false
                                }
                            } else if b_hi == b_lo + 1 && b_hi < n_bins {
                                let lo_ok =
                                    (clamped && b_lo == 0) || u_max + MARGIN < c_edges[b_lo];
                                let hi_ok = (clamped && b_hi == n_bins - 1)
                                    || u_min - MARGIN > c_edges[b_hi + 1];
                                if lo_ok && hi_ok {
                                    let c = c_edges[b_hi];
                                    let mut cnt_lo = 0usize;
                                    let mut cnt_hi = 0usize;
                                    let mut rx = p00[0] + 0.5 * (s1[0] + s2[0]);
                                    let mut ry = p00[1] + 0.5 * (s1[1] + s2[1]);
                                    let mut rz = p00[2] + 0.5 * (s1[2] + s2[2]);
                                    let mut exact_overflow = false;
                                    for i in 0..n_ss {
                                        let mut x = rx;
                                        let mut y = ry;
                                        let mut z = rz;
                                        for j in 0..n_ss {
                                            let r2 = x * x + y * y + z * z;
                                            let diff = z * z - c * r2;
                                            let tol = MARGIN * r2;
                                            if diff > tol {
                                                cnt_lo += 1;
                                            } else if diff < -tol {
                                                cnt_hi += 1;
                                            } else {
                                                let d1 = rf + (i as f64 + 0.5) * inv_ss;
                                                let d2 = ((col * n_ss + j) as f64 + 0.5) * inv_ss;
                                                match sample_bin(d1, d2) {
                                                    Some(b) if b == b_lo => cnt_lo += 1,
                                                    Some(b) if b == b_hi => cnt_hi += 1,
                                                    _ => {
                                                        exact_overflow = true;
                                                        break;
                                                    }
                                                }
                                            }
                                            x += s2[0];
                                            y += s2[1];
                                            z += s2[2];
                                        }
                                        if exact_overflow {
                                            break;
                                        }
                                        rx += s1[0];
                                        ry += s1[1];
                                        rz += s1[2];
                                    }
                                    if exact_overflow {
                                        false
                                    } else {
                                        if cnt_lo > 0 {
                                            entries.push(Contribution {
                                                bin: b_lo,
                                                pixel,
                                                count: cnt_lo,
                                            });
                                        }
                                        if cnt_hi > 0 {
                                            entries.push(Contribution {
                                                bin: b_hi,
                                                pixel,
                                                count: cnt_hi,
                                            });
                                        }
                                        true
                                    }
                                } else {
                                    false
                                }
                            } else if b_hi > b_lo + 1 && b_hi < n_bins && b_hi - b_lo < 16 {
                                let lo_ok =
                                    (clamped && b_lo == 0) || u_max + MARGIN < c_edges[b_lo];
                                let hi_ok = (clamped && b_hi == n_bins - 1)
                                    || u_min - MARGIN > c_edges[b_hi + 1];
                                if lo_ok && hi_ok {
                                    let span = b_hi - b_lo + 1;
                                    let mut local = [0usize; 16];
                                    let mut rx = p00[0] + 0.5 * (s1[0] + s2[0]);
                                    let mut ry = p00[1] + 0.5 * (s1[1] + s2[1]);
                                    let mut rz = p00[2] + 0.5 * (s1[2] + s2[2]);
                                    let mut exact_overflow = false;
                                    for i in 0..n_ss {
                                        let mut x = rx;
                                        let mut y = ry;
                                        let mut z = rz;
                                        for j in 0..n_ss {
                                            let r2 = x * x + y * y + z * z;
                                            let u = (z * z) / r2;
                                            let mut b = b_lo;
                                            while b < b_hi && u <= c_edges[b + 1] {
                                                b += 1;
                                            }
                                            let near_lo =
                                                b > b_lo && (c_edges[b] - u).abs() <= MARGIN;
                                            let near_hi =
                                                b < b_hi && (u - c_edges[b + 1]).abs() <= MARGIN;
                                            if !near_lo && !near_hi {
                                                local[b - b_lo] += 1;
                                            } else {
                                                let d1 = rf + (i as f64 + 0.5) * inv_ss;
                                                let d2 = ((col * n_ss + j) as f64 + 0.5) * inv_ss;
                                                match sample_bin(d1, d2) {
                                                    Some(eb) if eb >= b_lo && eb <= b_hi => {
                                                        local[eb - b_lo] += 1;
                                                    }
                                                    _ => {
                                                        exact_overflow = true;
                                                        break;
                                                    }
                                                }
                                            }
                                            x += s2[0];
                                            y += s2[1];
                                            z += s2[2];
                                        }
                                        if exact_overflow {
                                            break;
                                        }
                                        rx += s1[0];
                                        ry += s1[1];
                                        rz += s1[2];
                                    }
                                    if exact_overflow {
                                        false
                                    } else {
                                        for (offset, &count) in local[..span].iter().enumerate() {
                                            if count > 0 {
                                                entries.push(Contribution {
                                                    bin: b_lo + offset,
                                                    pixel,
                                                    count,
                                                });
                                            }
                                        }
                                        true
                                    }
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                if !fast_done {
                    counts.clear();
                    for i in 0..n_ss {
                        let d1 = rf + (i as f64 + 0.5) * inv_ss;
                        for j in 0..n_ss {
                            let d2 = ((col * n_ss + j) as f64 + 0.5) * inv_ss;
                            let Some(bin) = sample_bin(d1, d2) else {
                                continue;
                            };
                            match counts.iter_mut().find(|(b, _)| *b == bin) {
                                Some((_, count)) => *count += 1,
                                None => counts.push((bin, 1)),
                            }
                        }
                    }
                    entries.extend(counts.iter().map(|&(bin, count)| Contribution {
                        bin,
                        pixel,
                        count,
                    }));
                }

                p00 = p01;
                p10 = p11;
            }
            entries.into_boxed_slice()
        })
        .collect();

    let mut row_ptr = vec![0usize; n_bins + 1];
    for entries in &rows {
        for entry in entries.iter() {
            row_ptr[entry.bin + 1] += 1;
        }
    }
    for bin in 0..n_bins {
        row_ptr[bin + 1] += row_ptr[bin];
    }
    let mut col_idx = vec![0usize; row_ptr[n_bins]];
    let mut values = vec![0.0; row_ptr[n_bins]];
    let mut next = row_ptr[..n_bins].to_vec();
    let denom = ss2 as f64;
    for entries in rows {
        for entry in entries.iter() {
            let slot = next[entry.bin];
            col_idx[slot] = entry.pixel;
            values[slot] = entry.count as f64 / denom;
            next[entry.bin] += 1;
        }
    }
    (row_ptr, col_idx, values)
}

/// Build a supersampled 1D CSR LUT directly (no HashMap in the public type).
///
/// A weight is the fraction of the pixel's `n_ss²` subpixels that fall into the
/// bin. Bins (rows) are not normalized, so the raw sums of several detectors
/// can be added.
///
/// The `radial` array contains pixel center values in the selected radial unit.
/// The `unit` parameter specifies the radial unit for proper subpixel conversion.
///
/// # Example
///
/// ```
/// use dioptrin::geometry::Geometry;
/// use dioptrin::integration::RadialUnit;
/// use dioptrin::poni::Poni;
///
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 1.0;
///
/// let geometry = Geometry::new(10, 10, poni);
/// let radial = geometry.calc_tth_all();
/// let lut = dioptrin::lut::build_lut1d_ss_csr(&geometry, &radial, 100, 2, RadialUnit::TthRad);
/// let intensity = vec![1.0; 100];
/// let norm = vec![1.0; 100];
/// let (_bins, y) = dioptrin::lut::integrate1d(&intensity, &lut, None, &norm);
/// assert_eq!(y.len(), 100);
/// ```
pub fn build_lut1d_ss_csr(
    geometry: &Geometry,
    radial: &[f64],
    n_points: usize,
    n_ss: usize,
    unit: RadialUnit,
) -> Lut1d {
    let (hist_min, _hist_max, hist_step) =
        histogram::ss_tth_min_max_step(geometry, radial, n_points, n_ss, unit);

    let transform = geometry.subpixel_transform();
    let conversion = RadialConversion::new(unit, geometry.poni.wavelength);
    let (row_ptr, col_idx, values) = build_counted_1d_csr(
        geometry,
        n_ss,
        n_points,
        hist_min,
        hist_step,
        true,
        &conversion,
        |d1, d2| {
            let radial = conversion.convert(transform.tth(d1, d2));
            Some(histogram::get_bin_clamped(
                radial, hist_min, hist_step, n_points,
            ))
        },
    );

    Lut1d {
        row_ptr,
        col_idx,
        values: Some(values),
        hist_min,
        hist_step,
    }
}

/// Build a supersampled 1D CSR LUT with explicit radial range.
///
/// Subpixels outside `[radial_min, radial_max]` are dropped (not assigned to
/// the first or last bin), so a pixel partly outside the range keeps only its in-range fraction.
pub fn build_lut1d_ss_csr_with_range(
    geometry: &Geometry,
    _radial: &[f64],
    n_points: usize,
    n_ss: usize,
    unit: RadialUnit,
    radial_min: f64,
    radial_max: f64,
) -> Lut1d {
    let (hist_min, hist_max, hist_step) =
        histogram::min_max_step_from_range(radial_min, radial_max, n_points);

    let transform = geometry.subpixel_transform();
    let conversion = RadialConversion::new(unit, geometry.poni.wavelength);
    let (row_ptr, col_idx, values) = build_counted_1d_csr(
        geometry,
        n_ss,
        n_points,
        hist_min,
        hist_step,
        false,
        &conversion,
        |d1, d2| {
            let radial = conversion.convert(transform.tth(d1, d2));
            histogram::get_bin_in_range(radial, hist_min, hist_max, hist_step, n_points)
        },
    );

    Lut1d {
        row_ptr,
        col_idx,
        values: Some(values),
        hist_min,
        hist_step,
    }
}

/// Partition of `(-π, π)` into monotonic open azimuthal wedges of width `<= 1.5` rad,
/// each mapping to a single constant azimuthal bin index.
struct AzimuthalSegments {
    /// Strictly increasing edges: `edges[0] == -π`, `edges[m] == π`.
    edges: Vec<f64>,
    /// Precomputed `(cos(edge), sin(edge))` for each edge in `edges`.
    cs: Vec<(f64, f64)>,
    /// Output azimuthal bin index for each segment `0..m`.
    bins: Vec<usize>,
}

impl AzimuthalSegments {
    fn from_breakpoints<F>(raw_breaks: Vec<f64>, bin_of_angle: F) -> Self
    where
        F: Fn(f64) -> usize,
    {
        const PI: f64 = std::f64::consts::PI;
        const MAX_WEDGE: f64 = 1.5; // Ensures any 2 adjacent wedges span <= 3.0 < π.

        let mut edges = Vec::with_capacity(raw_breaks.len() + 8);
        let mut bins = Vec::with_capacity(raw_breaks.len() + 8);
        edges.push(-PI);

        let mut prev = -PI;
        for brk in raw_breaks.into_iter().chain(std::iter::once(PI)) {
            if brk <= prev + 1e-12 {
                continue;
            }
            let bin = bin_of_angle(0.5 * (prev + brk));
            let span = brk - prev;
            if span > MAX_WEDGE {
                let parts = (span / MAX_WEDGE).ceil() as usize;
                let step = span / parts as f64;
                for k in 1..parts {
                    edges.push(prev + k as f64 * step);
                    bins.push(bin);
                }
            }
            edges.push(brk);
            bins.push(bin);
            prev = brk;
        }
        if let Some(last) = edges.last_mut() {
            *last = PI;
        }

        let cs = edges
            .iter()
            .enumerate()
            .map(|(i, &a)| {
                if i == 0 || i + 1 == edges.len() {
                    (-1.0, 0.0)
                } else {
                    (a.cos(), a.sin())
                }
            })
            .collect();

        Self { edges, cs, bins }
    }

    fn clamped(azimuthal_min: f64, azimuthal_step: f64, n_azimuthal: usize) -> Option<Self> {
        const PI: f64 = std::f64::consts::PI;
        if n_azimuthal == 0 || !(azimuthal_step > 0.0 && azimuthal_step.is_finite()) {
            return None;
        }
        let mut breaks = Vec::with_capacity(n_azimuthal);
        for k in 1..n_azimuthal {
            let a = azimuthal_min + k as f64 * azimuthal_step;
            if a > -PI && a < PI {
                breaks.push(a);
            }
        }
        Some(Self::from_breakpoints(breaks, |a| {
            histogram::get_bin_clamped(a, azimuthal_min, azimuthal_step, n_azimuthal)
        }))
    }

    fn wrapped(min_azimuthal: f64, azimuthal_step: f64, n_azimuthal: usize) -> Option<Self> {
        const PI: f64 = std::f64::consts::PI;
        if n_azimuthal == 0 || !(azimuthal_step > 0.0 && azimuthal_step.is_finite()) {
            return None;
        }
        let est = (2.0 * PI) / azimuthal_step;
        if !est.is_finite() || est > 100_000.0 {
            return None;
        }
        let k_start = ((-PI - min_azimuthal) / azimuthal_step).floor() as isize;
        let k_end = ((PI - min_azimuthal) / azimuthal_step).ceil() as isize;
        let mut breaks = Vec::with_capacity((k_end - k_start).max(0) as usize);
        for k in k_start..=k_end {
            let a = min_azimuthal + k as f64 * azimuthal_step;
            if a > -PI && a < PI {
                breaks.push(a);
            }
        }
        Some(Self::from_breakpoints(breaks, |a| {
            histogram::get_bin_wrapped(a, min_azimuthal, azimuthal_step, n_azimuthal)
        }))
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.bins.len()
    }

    /// Returns `true` iff point `p` lies strictly inside the open wedge
    /// `(edges[lo], edges[hi])` with at least `MARGIN` angular clearance.
    #[inline(always)]
    fn point_in_wedge(&self, p: [f64; 3], lo: usize, hi: usize, margin: f64) -> bool {
        let tol = margin * (p[0].abs() + p[1].abs());
        let (c_lo, s_lo) = self.cs[lo];
        let (c_hi, s_hi) = self.cs[hi];
        (p[0] * c_lo - p[1] * s_lo > tol) && (p[0] * c_hi - p[1] * s_hi < -tol)
    }

    #[inline(always)]
    fn quad_in_wedge(&self, quad: [[f64; 3]; 4], lo: usize, hi: usize, margin: f64) -> bool {
        self.point_in_wedge(quad[0], lo, hi, margin)
            && self.point_in_wedge(quad[1], lo, hi, margin)
            && self.point_in_wedge(quad[2], lo, hi, margin)
            && self.point_in_wedge(quad[3], lo, hi, margin)
    }

    #[inline(always)]
    fn find_segment(&self, chi: f64) -> usize {
        let pos = self.edges.partition_point(|&e| e <= chi);
        pos.saturating_sub(1).min(self.bins.len() - 1)
    }
}

/// Fast 2D supersampled CSR builder using affine corner monotonicity, `cos²(2θ)`
/// radial thresholds, and convex half-plane azimuthal wedges for flat detectors,
/// falling back to exact per-subpixel evaluation near boundaries or for distorted
/// detectors. Produces bit-for-bit identical output to [`build_counted_csr`].
#[allow(clippy::too_many_arguments)]
fn build_counted_2d_csr<F>(
    geometry: &Geometry,
    n_ss: usize,
    n_radial: usize,
    n_azimuthal: usize,
    radial_min: f64,
    radial_step: f64,
    conversion: &RadialConversion,
    az_segs: Option<AzimuthalSegments>,
    sample_bin: F,
) -> (Vec<usize>, Vec<usize>, Vec<f64>)
where
    F: Fn(f64, f64) -> Option<usize> + Sync,
{
    let n_bins = n_radial * n_azimuthal;
    let transform = geometry.subpixel_transform();
    let (Some((p0, d1_step, d2_step)), Some(az_segs)) = (transform.affine_plane(), az_segs) else {
        return build_counted_csr(geometry, n_ss, n_bins, true, sample_bin);
    };
    if n_radial == 0 || n_azimuthal == 0 || az_segs.len() == 0 {
        return build_counted_csr(geometry, n_ss, n_bins, true, sample_bin);
    }

    const MARGIN: f64 = 1e-9;
    let c_edges = radial_edge_cos2(radial_min, radial_step, n_radial, conversion);
    let ss2 = n_ss * n_ss;
    let inv_ss = 1.0 / n_ss as f64;
    let s1 = [
        d1_step[0] * inv_ss,
        d1_step[1] * inv_ss,
        d1_step[2] * inv_ss,
    ];
    let s2 = [
        d2_step[0] * inv_ss,
        d2_step[1] * inv_ss,
        d2_step[2] * inv_ss,
    ];
    let dim2 = geometry.dim2();
    let n_seg = az_segs.len();

    let rows: Vec<Box<[Contribution]>> = (0..geometry.dim1())
        .into_par_iter()
        .map(|row| {
            let mut entries = Vec::with_capacity(dim2 * 2);
            let mut counts: Vec<(usize, usize)> = Vec::new();
            let rf = row as f64;
            let mut p00 = [
                p0[0] + rf * d1_step[0],
                p0[1] + rf * d1_step[1],
                p0[2] + rf * d1_step[2],
            ];
            let mut p10 = [
                p00[0] + d1_step[0],
                p00[1] + d1_step[1],
                p00[2] + d1_step[2],
            ];
            let mut r_hint = 0usize;
            let mut a_hint = 0usize;

            for col in 0..dim2 {
                let p01 = [
                    p00[0] + d2_step[0],
                    p00[1] + d2_step[1],
                    p00[2] + d2_step[2],
                ];
                let p11 = [
                    p10[0] + d2_step[0],
                    p10[1] + d2_step[1],
                    p10[2] + d2_step[2],
                ];
                let pixel = row * dim2 + col;

                let fast_done = if p00[2] > 0.0 && p01[2] > 0.0 && p10[2] > 0.0 && p11[2] > 0.0 {
                    // 1. Classify pixel azimuthally using zero-atan2 hint first.
                    // Result: None (fallback) or Some((ab0, ab1, split_cs))
                    // where `ab0 == ab1` means 1 azimuthal bin.
                    let quad = [p00, p01, p10, p11];
                    let az_class = if az_segs.quad_in_wedge(quad, a_hint, a_hint + 1, MARGIN) {
                        let b = az_segs.bins[a_hint];
                        Some((b, b, (0.0, 0.0)))
                    } else if a_hint + 1 < n_seg
                        && az_segs.quad_in_wedge(quad, a_hint, a_hint + 2, MARGIN)
                    {
                        Some((
                            az_segs.bins[a_hint],
                            az_segs.bins[a_hint + 1],
                            az_segs.cs[a_hint + 1],
                        ))
                    } else if a_hint > 0
                        && az_segs.quad_in_wedge(quad, a_hint - 1, a_hint + 1, MARGIN)
                    {
                        let prev = a_hint - 1;
                        a_hint = prev;
                        Some((
                            az_segs.bins[prev],
                            az_segs.bins[prev + 1],
                            az_segs.cs[prev + 1],
                        ))
                    } else {
                        let s = az_segs.find_segment(p00[0].atan2(p00[1]));
                        a_hint = s;
                        if az_segs.quad_in_wedge(quad, s, s + 1, MARGIN) {
                            let b = az_segs.bins[s];
                            Some((b, b, (0.0, 0.0)))
                        } else if s + 1 < n_seg && az_segs.quad_in_wedge(quad, s, s + 2, MARGIN) {
                            Some((az_segs.bins[s], az_segs.bins[s + 1], az_segs.cs[s + 1]))
                        } else if s > 0 && az_segs.quad_in_wedge(quad, s - 1, s + 1, MARGIN) {
                            Some((az_segs.bins[s - 1], az_segs.bins[s], az_segs.cs[s]))
                        } else {
                            None
                        }
                    };

                    if let Some((ab0, ab1, (cos_m, sin_m))) = az_class {
                        let mono = edge_grad_sign(p00, d2_step) * edge_grad_sign(p01, d2_step)
                            > 0.0
                            && edge_grad_sign(p10, d2_step) * edge_grad_sign(p11, d2_step) > 0.0
                            && edge_grad_sign(p00, d1_step) * edge_grad_sign(p10, d1_step) > 0.0
                            && edge_grad_sign(p01, d1_step) * edge_grad_sign(p11, d1_step) > 0.0;
                        if mono {
                            let u00 = (p00[2] * p00[2])
                                / (p00[0] * p00[0] + p00[1] * p00[1] + p00[2] * p00[2]);
                            let u01 = (p01[2] * p01[2])
                                / (p01[0] * p01[0] + p01[1] * p01[1] + p01[2] * p01[2]);
                            let u10 = (p10[2] * p10[2])
                                / (p10[0] * p10[0] + p10[1] * p10[1] + p10[2] * p10[2]);
                            let u11 = (p11[2] * p11[2])
                                / (p11[0] * p11[0] + p11[1] * p11[1] + p11[2] * p11[2]);
                            let u_min = u00.min(u01).min(u10.min(u11));
                            let u_max = u00.max(u01).max(u10.max(u11));

                            let rb_lo = find_bin_hint(&c_edges, u_max, r_hint).min(n_radial - 1);
                            let rb_hi = find_bin_hint(&c_edges, u_min, rb_lo).min(n_radial - 1);
                            r_hint = rb_lo;

                            let lo_ok = rb_lo == 0 || u_max + MARGIN < c_edges[rb_lo];
                            let hi_ok =
                                rb_hi == n_radial - 1 || u_min - MARGIN > c_edges[rb_hi + 1];

                            if lo_ok && hi_ok && rb_hi - rb_lo < 16 {
                                let single_r = rb_lo == rb_hi;
                                let two_r = rb_hi == rb_lo + 1;
                                let single_a = ab0 == ab1;
                                if single_r && single_a {
                                    entries.push(Contribution {
                                        bin: ab0 * n_radial + rb_lo,
                                        pixel,
                                        count: ss2,
                                    });
                                    true
                                } else if two_r && single_a {
                                    // 2 radial × 1 azimuthal
                                    let c = c_edges[rb_hi];
                                    let bin_lo = ab0 * n_radial + rb_lo;
                                    let bin_hi = ab0 * n_radial + rb_hi;
                                    let mut cnt_lo = 0usize;
                                    let mut cnt_hi = 0usize;
                                    let mut rx = p00[0] + 0.5 * (s1[0] + s2[0]);
                                    let mut ry = p00[1] + 0.5 * (s1[1] + s2[1]);
                                    let mut rz = p00[2] + 0.5 * (s1[2] + s2[2]);
                                    let mut overflow = false;
                                    for i in 0..n_ss {
                                        let (mut x, mut y, mut z) = (rx, ry, rz);
                                        for j in 0..n_ss {
                                            let r2 = x * x + y * y + z * z;
                                            let diff = z * z - c * r2;
                                            let tol = MARGIN * r2;
                                            if diff > tol {
                                                cnt_lo += 1;
                                            } else if diff < -tol {
                                                cnt_hi += 1;
                                            } else {
                                                let d1 = rf + (i as f64 + 0.5) * inv_ss;
                                                let d2 = ((col * n_ss + j) as f64 + 0.5) * inv_ss;
                                                match sample_bin(d1, d2) {
                                                    Some(b) if b == bin_lo => cnt_lo += 1,
                                                    Some(b) if b == bin_hi => cnt_hi += 1,
                                                    _ => {
                                                        overflow = true;
                                                        break;
                                                    }
                                                }
                                            }
                                            x += s2[0];
                                            y += s2[1];
                                            z += s2[2];
                                        }
                                        if overflow {
                                            break;
                                        }
                                        rx += s1[0];
                                        ry += s1[1];
                                        rz += s1[2];
                                    }
                                    if overflow {
                                        false
                                    } else {
                                        if cnt_lo > 0 {
                                            entries.push(Contribution {
                                                bin: bin_lo,
                                                pixel,
                                                count: cnt_lo,
                                            });
                                        }
                                        if cnt_hi > 0 {
                                            entries.push(Contribution {
                                                bin: bin_hi,
                                                pixel,
                                                count: cnt_hi,
                                            });
                                        }
                                        true
                                    }
                                } else if single_r && !single_a {
                                    // 1 radial × 2 azimuthal
                                    let bin_a0 = ab0 * n_radial + rb_lo;
                                    let bin_a1 = ab1 * n_radial + rb_lo;
                                    let mut cnt_a0 = 0usize;
                                    let mut cnt_a1 = 0usize;
                                    let mut rx = p00[0] + 0.5 * (s1[0] + s2[0]);
                                    let mut ry = p00[1] + 0.5 * (s1[1] + s2[1]);
                                    let mut overflow = false;
                                    for i in 0..n_ss {
                                        let (mut x, mut y) = (rx, ry);
                                        for j in 0..n_ss {
                                            let diff = x * cos_m - y * sin_m;
                                            let tol = MARGIN * (x.abs() + y.abs());
                                            if diff < -tol {
                                                cnt_a0 += 1;
                                            } else if diff > tol {
                                                cnt_a1 += 1;
                                            } else {
                                                let d1 = rf + (i as f64 + 0.5) * inv_ss;
                                                let d2 = ((col * n_ss + j) as f64 + 0.5) * inv_ss;
                                                match sample_bin(d1, d2) {
                                                    Some(b) if b == bin_a0 => cnt_a0 += 1,
                                                    Some(b) if b == bin_a1 => cnt_a1 += 1,
                                                    _ => {
                                                        overflow = true;
                                                        break;
                                                    }
                                                }
                                            }
                                            x += s2[0];
                                            y += s2[1];
                                        }
                                        if overflow {
                                            break;
                                        }
                                        rx += s1[0];
                                        ry += s1[1];
                                    }
                                    if overflow {
                                        false
                                    } else {
                                        if cnt_a0 > 0 {
                                            entries.push(Contribution {
                                                bin: bin_a0,
                                                pixel,
                                                count: cnt_a0,
                                            });
                                        }
                                        if cnt_a1 > 0 {
                                            entries.push(Contribution {
                                                bin: bin_a1,
                                                pixel,
                                                count: cnt_a1,
                                            });
                                        }
                                        true
                                    }
                                } else if two_r && !single_a {
                                    // 2 radial × 2 azimuthal
                                    let c = c_edges[rb_hi];
                                    let bins4 = [
                                        ab0 * n_radial + rb_lo,
                                        ab0 * n_radial + rb_hi,
                                        ab1 * n_radial + rb_lo,
                                        ab1 * n_radial + rb_hi,
                                    ];
                                    let mut cnt = [0usize; 4];
                                    let mut rx = p00[0] + 0.5 * (s1[0] + s2[0]);
                                    let mut ry = p00[1] + 0.5 * (s1[1] + s2[1]);
                                    let mut rz = p00[2] + 0.5 * (s1[2] + s2[2]);
                                    let mut overflow = false;
                                    for i in 0..n_ss {
                                        let (mut x, mut y, mut z) = (rx, ry, rz);
                                        for j in 0..n_ss {
                                            let r2 = x * x + y * y + z * z;
                                            let diff_r = z * z - c * r2;
                                            let tol_r = MARGIN * r2;
                                            let diff_a = x * cos_m - y * sin_m;
                                            let tol_a = MARGIN * (x.abs() + y.abs());
                                            if diff_r.abs() > tol_r && diff_a.abs() > tol_a {
                                                let r_idx = usize::from(diff_r < 0.0);
                                                let a_idx = usize::from(diff_a > 0.0);
                                                cnt[a_idx * 2 + r_idx] += 1;
                                            } else {
                                                let d1 = rf + (i as f64 + 0.5) * inv_ss;
                                                let d2 = ((col * n_ss + j) as f64 + 0.5) * inv_ss;
                                                match sample_bin(d1, d2) {
                                                    Some(b) if b == bins4[0] => cnt[0] += 1,
                                                    Some(b) if b == bins4[1] => cnt[1] += 1,
                                                    Some(b) if b == bins4[2] => cnt[2] += 1,
                                                    Some(b) if b == bins4[3] => cnt[3] += 1,
                                                    _ => {
                                                        overflow = true;
                                                        break;
                                                    }
                                                }
                                            }
                                            x += s2[0];
                                            y += s2[1];
                                            z += s2[2];
                                        }
                                        if overflow {
                                            break;
                                        }
                                        rx += s1[0];
                                        ry += s1[1];
                                        rz += s1[2];
                                    }
                                    if overflow {
                                        false
                                    } else {
                                        for k in 0..4 {
                                            if cnt[k] > 0 {
                                                entries.push(Contribution {
                                                    bin: bins4[k],
                                                    pixel,
                                                    count: cnt[k],
                                                });
                                            }
                                        }
                                        true
                                    }
                                } else if single_a {
                                    // (3..=16) radial × 1 azimuthal
                                    let span = rb_hi - rb_lo + 1;
                                    let base_bin = ab0 * n_radial + rb_lo;
                                    let mut local = [0usize; 16];
                                    let mut rx = p00[0] + 0.5 * (s1[0] + s2[0]);
                                    let mut ry = p00[1] + 0.5 * (s1[1] + s2[1]);
                                    let mut rz = p00[2] + 0.5 * (s1[2] + s2[2]);
                                    let mut overflow = false;
                                    for i in 0..n_ss {
                                        let (mut x, mut y, mut z) = (rx, ry, rz);
                                        for j in 0..n_ss {
                                            let r2 = x * x + y * y + z * z;
                                            let u = (z * z) / r2;
                                            let mut b = rb_lo;
                                            while b < rb_hi && u <= c_edges[b + 1] {
                                                b += 1;
                                            }
                                            let near_lo =
                                                b > rb_lo && (c_edges[b] - u).abs() <= MARGIN;
                                            let near_hi =
                                                b < rb_hi && (u - c_edges[b + 1]).abs() <= MARGIN;
                                            if !near_lo && !near_hi {
                                                local[b - rb_lo] += 1;
                                            } else {
                                                let d1 = rf + (i as f64 + 0.5) * inv_ss;
                                                let d2 = ((col * n_ss + j) as f64 + 0.5) * inv_ss;
                                                match sample_bin(d1, d2) {
                                                    Some(eb)
                                                        if eb >= base_bin
                                                            && eb < base_bin + span =>
                                                    {
                                                        local[eb - base_bin] += 1;
                                                    }
                                                    _ => {
                                                        overflow = true;
                                                        break;
                                                    }
                                                }
                                            }
                                            x += s2[0];
                                            y += s2[1];
                                            z += s2[2];
                                        }
                                        if overflow {
                                            break;
                                        }
                                        rx += s1[0];
                                        ry += s1[1];
                                        rz += s1[2];
                                    }
                                    if overflow {
                                        false
                                    } else {
                                        for (offset, &count) in local[..span].iter().enumerate() {
                                            if count > 0 {
                                                entries.push(Contribution {
                                                    bin: base_bin + offset,
                                                    pixel,
                                                    count,
                                                });
                                            }
                                        }
                                        true
                                    }
                                } else {
                                    // (3..=16) radial × 2 azimuthal
                                    let span = rb_hi - rb_lo + 1;
                                    let base0 = ab0 * n_radial + rb_lo;
                                    let base1 = ab1 * n_radial + rb_lo;
                                    let mut local = [0usize; 32];
                                    let mut rx = p00[0] + 0.5 * (s1[0] + s2[0]);
                                    let mut ry = p00[1] + 0.5 * (s1[1] + s2[1]);
                                    let mut rz = p00[2] + 0.5 * (s1[2] + s2[2]);
                                    let mut overflow = false;
                                    for i in 0..n_ss {
                                        let (mut x, mut y, mut z) = (rx, ry, rz);
                                        for j in 0..n_ss {
                                            let r2 = x * x + y * y + z * z;
                                            let u = (z * z) / r2;
                                            let mut b = rb_lo;
                                            while b < rb_hi && u <= c_edges[b + 1] {
                                                b += 1;
                                            }
                                            let near_lo =
                                                b > rb_lo && (c_edges[b] - u).abs() <= MARGIN;
                                            let near_hi =
                                                b < rb_hi && (u - c_edges[b + 1]).abs() <= MARGIN;
                                            let diff_a = x * cos_m - y * sin_m;
                                            let tol_a = MARGIN * (x.abs() + y.abs());
                                            if !near_lo && !near_hi && diff_a.abs() > tol_a {
                                                let a_idx = usize::from(diff_a > 0.0);
                                                local[a_idx * 16 + (b - rb_lo)] += 1;
                                            } else {
                                                let d1 = rf + (i as f64 + 0.5) * inv_ss;
                                                let d2 = ((col * n_ss + j) as f64 + 0.5) * inv_ss;
                                                match sample_bin(d1, d2) {
                                                    Some(eb)
                                                        if eb >= base0 && eb < base0 + span =>
                                                    {
                                                        local[eb - base0] += 1;
                                                    }
                                                    Some(eb)
                                                        if eb >= base1 && eb < base1 + span =>
                                                    {
                                                        local[16 + (eb - base1)] += 1;
                                                    }
                                                    _ => {
                                                        overflow = true;
                                                        break;
                                                    }
                                                }
                                            }
                                            x += s2[0];
                                            y += s2[1];
                                            z += s2[2];
                                        }
                                        if overflow {
                                            break;
                                        }
                                        rx += s1[0];
                                        ry += s1[1];
                                        rz += s1[2];
                                    }
                                    if overflow {
                                        false
                                    } else {
                                        for (offset, &count) in local[..span].iter().enumerate() {
                                            if count > 0 {
                                                entries.push(Contribution {
                                                    bin: base0 + offset,
                                                    pixel,
                                                    count,
                                                });
                                            }
                                        }
                                        for (offset, &count) in
                                            local[16..16 + span].iter().enumerate()
                                        {
                                            if count > 0 {
                                                entries.push(Contribution {
                                                    bin: base1 + offset,
                                                    pixel,
                                                    count,
                                                });
                                            }
                                        }
                                        true
                                    }
                                }
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                if !fast_done {
                    counts.clear();
                    for i in 0..n_ss {
                        let d1 = rf + (i as f64 + 0.5) * inv_ss;
                        for j in 0..n_ss {
                            let d2 = ((col * n_ss + j) as f64 + 0.5) * inv_ss;
                            let Some(bin) = sample_bin(d1, d2) else {
                                continue;
                            };
                            match counts.iter_mut().find(|(b, _)| *b == bin) {
                                Some((_, count)) => *count += 1,
                                None => counts.push((bin, 1)),
                            }
                        }
                    }
                    entries.extend(counts.iter().map(|&(bin, count)| Contribution {
                        bin,
                        pixel,
                        count,
                    }));
                }

                p00 = p01;
                p10 = p11;
            }
            entries.into_boxed_slice()
        })
        .collect();

    use std::sync::atomic::{AtomicUsize, Ordering};

    let row_counts: Vec<AtomicUsize> = (0..n_bins).map(|_| AtomicUsize::new(0)).collect();
    let totals_atomic: Vec<AtomicUsize> = (0..n_bins).map(|_| AtomicUsize::new(0)).collect();
    rows.par_iter().for_each(|entries| {
        for entry in entries.iter() {
            row_counts[entry.bin].fetch_add(1, Ordering::Relaxed);
            totals_atomic[entry.bin].fetch_add(entry.count, Ordering::Relaxed);
        }
    });

    let mut row_ptr = vec![0usize; n_bins + 1];
    for bin in 0..n_bins {
        row_ptr[bin + 1] = row_ptr[bin] + row_counts[bin].load(Ordering::Relaxed);
    }
    let nnz = row_ptr[n_bins];
    let totals: Vec<usize> = totals_atomic
        .into_iter()
        .map(AtomicUsize::into_inner)
        .collect();

    let next: Vec<AtomicUsize> = row_ptr[..n_bins]
        .iter()
        .map(|&off| AtomicUsize::new(off))
        .collect();
    let col_idx_atomic: Vec<AtomicUsize> = (0..nnz).map(|_| AtomicUsize::new(0)).collect();
    let counts_atomic: Vec<AtomicUsize> = (0..nnz).map(|_| AtomicUsize::new(0)).collect();

    rows.into_par_iter().for_each(|entries| {
        for entry in entries.iter() {
            let slot = next[entry.bin].fetch_add(1, Ordering::Relaxed);
            col_idx_atomic[slot].store(entry.pixel, Ordering::Relaxed);
            counts_atomic[slot].store(entry.count, Ordering::Relaxed);
        }
    });

    let mut col_idx: Vec<usize> = col_idx_atomic
        .into_iter()
        .map(AtomicUsize::into_inner)
        .collect();
    let mut values: Vec<f64> = counts_atomic
        .into_iter()
        .map(|c| c.into_inner() as f64)
        .collect();

    const ROW_BLOCK: usize = 1024;
    let mut blocks = Vec::with_capacity(n_bins.div_ceil(ROW_BLOCK));
    let (mut cols_rest, mut vals_rest) = (col_idx.as_mut_slice(), values.as_mut_slice());
    for start in (0..n_bins).step_by(ROW_BLOCK) {
        let end = (start + ROW_BLOCK).min(n_bins);
        let len = row_ptr[end] - row_ptr[start];
        let (cols, cols_tail) = std::mem::take(&mut cols_rest).split_at_mut(len);
        let (vals, vals_tail) = std::mem::take(&mut vals_rest).split_at_mut(len);
        cols_rest = cols_tail;
        vals_rest = vals_tail;
        blocks.push((start..end, cols, vals));
    }
    blocks.into_par_iter().for_each(|(block, cols, vals)| {
        let base = row_ptr[block.start];
        let mut scratch: Vec<(usize, f64)> = Vec::new();
        for bin in block {
            let (s, e) = (row_ptr[bin] - base, row_ptr[bin + 1] - base);
            if s == e {
                continue;
            }
            let (c_slice, v_slice) = (&mut cols[s..e], &mut vals[s..e]);
            if !c_slice.is_sorted() {
                scratch.clear();
                scratch.extend(c_slice.iter().copied().zip(v_slice.iter().copied()));
                scratch.sort_by_key(|&(pixel, _)| pixel);
                for ((c, v), &(pixel, val)) in
                    c_slice.iter_mut().zip(v_slice.iter_mut()).zip(&scratch)
                {
                    *c = pixel;
                    *v = val;
                }
            }
            let total = totals[bin];
            if total != 0 {
                let denom = total as f64;
                for v in v_slice.iter_mut() {
                    *v /= denom;
                }
            }
        }
    });

    (row_ptr, col_idx, values)
}

/// Build a supersampled 2D CSR LUT directly (no HashMap in the public type).
///
/// The `radial` array contains pixel center values in the selected radial unit.
/// The `unit` parameter specifies the radial unit for proper subpixel conversion.
///
/// # Example
///
/// ```
/// use dioptrin::geometry::Geometry;
/// use dioptrin::integration::RadialUnit;
/// use dioptrin::poni::Poni;
///
/// let mut poni = Poni::new();
/// poni.pixel1 = 1.0;
/// poni.pixel2 = 1.0;
/// poni.distance = 1.0;
///
/// let geometry = Geometry::new(10, 10, poni);
/// let radial = geometry.calc_tth_all();
/// let azimuthal = geometry.calc_azimuth_all();
/// let lut2d = dioptrin::lut::build_lut2d_ss_csr(
///     &geometry, &radial, &azimuthal, 32, 36, 2, RadialUnit::TthRad
/// );
/// let intensity = vec![1.0; 100];
/// let norm = vec![1.0; 100];
/// let y = dioptrin::lut::integrate2d(&intensity, &lut2d, None, &norm);
/// assert_eq!(y.len(), 32 * 36);
/// ```
pub fn build_lut2d_ss_csr(
    geometry: &Geometry,
    radial: &[f64],
    azimuthal: &[f64],
    n_radial: usize,
    n_azimuthal: usize,
    n_ss: usize,
    unit: RadialUnit,
) -> Lut2d {
    let (radial_min, _radial_max, radial_step) =
        histogram::ss_tth_min_max_step(geometry, radial, n_radial, n_ss, unit);
    let (azimuthal_min, _azimuthal_max, azimuthal_step) =
        histogram::supersampled_azimuthal_min_max_step(geometry, azimuthal, n_azimuthal, n_ss);

    let transform = geometry.subpixel_transform();
    let conversion = RadialConversion::new(unit, geometry.poni.wavelength);
    let az_segs = AzimuthalSegments::clamped(azimuthal_min, azimuthal_step, n_azimuthal);
    let (row_ptr, col_idx, values) = build_counted_2d_csr(
        geometry,
        n_ss,
        n_radial,
        n_azimuthal,
        radial_min,
        radial_step,
        &conversion,
        az_segs,
        |d1, d2| {
            let (tth, azimuth) = transform.tth_and_azimuth(d1, d2);
            let rb = histogram::get_bin_clamped(
                conversion.convert(tth),
                radial_min,
                radial_step,
                n_radial,
            );
            // The grid ends at the largest pixel-centre or hull azimuth, so a
            // subpixel that rounds (or lies) past it belongs in the last bin,
            // not (by wrapping) in bin 0.
            let ab =
                histogram::get_bin_clamped(azimuth, azimuthal_min, azimuthal_step, n_azimuthal);
            Some(ab * n_radial + rb)
        },
    );

    Lut2d {
        row_ptr,
        col_idx,
        values: Some(values),
        n_tth: n_radial,
        n_azimuthal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Geometry;
    use crate::poni::Poni;

    #[test]
    fn automatic_azimuthal_range_keeps_the_maximum_in_the_last_bin() {
        // On this real calibration (Pilatus 1M shape) the azimuth spans the
        // -pi/pi cut, and the subpixel with the largest azimuth rounds to bin
        // index n_azimuthal. It belongs to the last azimuthal bin, not bin 0.
        let (dim1, dim2) = (1043, 981);
        let geometry = Geometry::new(dim1, dim2, Poni::load("test_data/v1.poni").unwrap());
        let radial = geometry.calc_tth_all();
        let azimuthal = geometry.calc_azimuth_all();
        let (n_radial, n_azimuthal, n_ss) = (50, 36, 1);
        let lut = build_lut2d_ss_csr(
            &geometry,
            &radial,
            &azimuthal,
            n_radial,
            n_azimuthal,
            n_ss,
            RadialUnit::TthRad,
        );

        // With one subpixel per pixel, every pixel has exactly one table entry.
        let (r_min, _, r_step) =
            histogram::ss_tth_min_max_step(&geometry, &radial, n_radial, n_ss, RadialUnit::TthRad);
        let (a_min, _, a_step) = histogram::supersampled_azimuthal_min_max_step(
            &geometry,
            &azimuthal,
            n_azimuthal,
            n_ss,
        );
        let transform = geometry.subpixel_transform();
        let mut expected = vec![usize::MAX; dim1 * dim2];
        let mut rounds_past_the_end = 0;
        for row in 0..dim1 {
            for col in 0..dim2 {
                let (tth, azimuth) = transform.tth_and_azimuth(row as f64 + 0.5, col as f64 + 0.5);
                if histogram::get_bin(azimuth, a_min, a_step) >= n_azimuthal {
                    rounds_past_the_end += 1;
                }
                let rb = histogram::get_bin_clamped(tth, r_min, r_step, n_radial);
                let ab = histogram::get_bin_clamped(azimuth, a_min, a_step, n_azimuthal);
                expected[row * dim2 + col] = ab * n_radial + rb;
            }
        }
        assert!(
            rounds_past_the_end > 0,
            "geometry no longer exercises the case"
        );

        let mut actual = vec![usize::MAX; dim1 * dim2];
        for bin in 0..n_radial * n_azimuthal {
            for &pixel in &lut.col_idx[lut.row_ptr[bin]..lut.row_ptr[bin + 1]] {
                assert_eq!(actual[pixel], usize::MAX, "pixel {pixel} in two bins");
                actual[pixel] = bin;
            }
        }
        for (pixel, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a, e, "pixel {pixel}: bin {a}, expected {e}");
        }
    }

    fn create_test_geometry() -> Geometry {
        let mut poni = Poni::new();
        poni.pixel1 = 0.0001; // 100 µm
        poni.pixel2 = 0.0001;
        poni.distance = 0.1; // 10 cm
        poni.wavelength = 1e-10; // 1 Angstrom
        Geometry::new(10, 10, poni)
    }

    #[test]
    fn test_build_lut1d_ss_csr_tth_rad() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let lut = build_lut1d_ss_csr(&geometry, &tth, 5, 2, RadialUnit::TthRad);

        assert_eq!(lut.n_points(), 5);
        assert!(lut.hist_min >= 0.0);
        assert!(lut.hist_step > 0.0);
        assert!(lut.values.is_some());
    }

    #[test]
    fn test_build_lut1d_ss_csr_tth_deg() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let radial_deg = RadialUnit::TthDeg.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut1d_ss_csr(&geometry, &radial_deg, 5, 2, RadialUnit::TthDeg);

        assert_eq!(lut.n_points(), 5);
        assert!(lut.hist_min >= 0.0);
    }

    #[test]
    fn test_build_lut1d_ss_csr_q_angstrom() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let radial_q = RadialUnit::QAngstrom.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut1d_ss_csr(&geometry, &radial_q, 5, 2, RadialUnit::QAngstrom);

        assert_eq!(lut.n_points(), 5);
        assert!(lut.hist_min >= 0.0, "q should be non-negative");
    }

    #[test]
    fn test_build_lut1d_ss_csr_q_nm() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let radial_q = RadialUnit::QNm.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut1d_ss_csr(&geometry, &radial_q, 5, 2, RadialUnit::QNm);

        assert_eq!(lut.n_points(), 5);
        assert!(lut.hist_min >= 0.0, "q should be non-negative");
    }

    #[test]
    fn test_build_lut2d_ss_csr_with_q_unit() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let azimuthal = geometry.calc_azimuth_all();
        let radial_q = RadialUnit::QAngstrom.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut2d_ss_csr(
            &geometry,
            &radial_q,
            &azimuthal,
            5,
            4,
            2,
            RadialUnit::QAngstrom,
        );

        assert_eq!(lut.n_tth, 5);
        assert_eq!(lut.n_azimuthal, 4);
        assert_eq!(lut.row_ptr.len(), 5 * 4 + 1);
        assert!(lut.values.is_some());
    }

    #[test]
    fn test_build_lut2d_ss_csr_with_azimuthal_range_q_unit() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let radial_q = RadialUnit::QNm.convert_from_tth(&tth, geometry.poni.wavelength);

        let lut = build_lut2d_ss_csr_with_azimuthal_range(
            &geometry,
            &radial_q,
            5,
            4,
            2,
            0.0,
            std::f64::consts::PI / 2.0,
            RadialUnit::QNm,
        );

        assert_eq!(lut.n_tth, 5);
        assert_eq!(lut.n_azimuthal, 4);
    }

    #[test]
    fn test_supersampled_different_units_different_bins() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();

        // Build LUT in radians
        let lut_rad = build_lut1d_ss_csr(&geometry, &tth, 5, 2, RadialUnit::TthRad);

        // Build LUT in degrees
        let radial_deg = RadialUnit::TthDeg.convert_from_tth(&tth, geometry.poni.wavelength);
        let lut_deg = build_lut1d_ss_csr(&geometry, &radial_deg, 5, 2, RadialUnit::TthDeg);

        // The bin centers should differ by factor of 180/π
        let ratio = lut_deg.hist_min / lut_rad.hist_min;
        let expected_ratio = 180.0 / std::f64::consts::PI;
        assert!(
            (ratio - expected_ratio).abs() / expected_ratio < 0.01,
            "Degree bins should be 180/π times radian bins, got ratio {}",
            ratio
        );
    }

    #[test]
    fn test_supersampled_q_angstrom_vs_q_nm_ratio() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();

        // Build LUT in q (Å⁻¹)
        let radial_qa = RadialUnit::QAngstrom.convert_from_tth(&tth, geometry.poni.wavelength);
        let lut_qa = build_lut1d_ss_csr(&geometry, &radial_qa, 5, 2, RadialUnit::QAngstrom);

        // Build LUT in q (nm⁻¹)
        let radial_qn = RadialUnit::QNm.convert_from_tth(&tth, geometry.poni.wavelength);
        let lut_qn = build_lut1d_ss_csr(&geometry, &radial_qn, 5, 2, RadialUnit::QNm);

        // q in nm⁻¹ should be 10x q in Å⁻¹
        let ratio = lut_qn.hist_min / lut_qa.hist_min;
        assert!(
            (ratio - 10.0).abs() < 0.01,
            "q_nm bins should be 10x q_angstrom bins, got ratio {}",
            ratio
        );
    }

    #[test]
    fn test_supersampled_integration_with_different_units() {
        use crate::lut::integrate1d;

        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let intensity: Vec<f64> = (0..geometry.size()).map(|i| (i + 1) as f64).collect();

        // Build and integrate in radians
        let lut_rad = build_lut1d_ss_csr(&geometry, &tth, 5, 2, RadialUnit::TthRad);
        let norm = vec![1.0; geometry.size()];
        let (bins_rad, int_rad) = integrate1d(&intensity, &lut_rad, None, &norm);

        // Build and integrate in degrees
        let radial_deg = RadialUnit::TthDeg.convert_from_tth(&tth, geometry.poni.wavelength);
        let lut_deg = build_lut1d_ss_csr(&geometry, &radial_deg, 5, 2, RadialUnit::TthDeg);
        let (bins_deg, int_deg) = integrate1d(&intensity, &lut_deg, None, &norm);

        // Bin centers should differ by 180/π
        for (r, d) in bins_rad.iter().zip(bins_deg.iter()) {
            let ratio = d / r;
            let expected = 180.0 / std::f64::consts::PI;
            assert!(
                (ratio - expected).abs() / expected < 0.01,
                "Bin centers should differ by 180/π"
            );
        }

        // Intensities should be the same since it's the same data
        for (ir, id) in int_rad.iter().zip(int_deg.iter()) {
            assert!(
                (ir - id).abs() < 1e-10,
                "Intensities should match: {} vs {}",
                ir,
                id
            );
        }
    }

    fn pixel_totals(lut: &Lut1d, n_pixels: usize) -> Vec<f64> {
        let mut totals = vec![0.0; n_pixels];
        for (&pixel, &w) in lut.col_idx.iter().zip(lut.values.as_ref().unwrap()) {
            totals[pixel] += w;
        }
        totals
    }

    #[test]
    fn weights_1d_are_fractions_of_the_pixel() {
        // 16 subpixels per pixel: a pixel fully inside the range has weight one.
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let lut = build_lut1d_ss_csr(&geometry, &tth, 7, 4, RadialUnit::TthRad);
        for (pixel, total) in pixel_totals(&lut, geometry.size()).iter().enumerate() {
            assert_eq!(*total, 1.0, "pixel {pixel}");
        }
    }

    #[test]
    fn explicit_range_drops_subpixels_outside_it() {
        let geometry = create_test_geometry();
        let tth = geometry.calc_tth_all();
        let (n_ss, n_points) = (4, 6);
        let (lo, hi) = (0.004, 0.009);
        let lut = build_lut1d_ss_csr_with_range(
            &geometry,
            &tth,
            n_points,
            n_ss,
            RadialUnit::TthRad,
            lo,
            hi,
        );
        let transform = geometry.subpixel_transform();
        let mut expected = vec![0.0; geometry.size()];
        let mut expected_bins = vec![0usize; n_points];
        for (pixel, total) in expected.iter_mut().enumerate() {
            let (row, col) = (pixel / geometry.dim2(), pixel % geometry.dim2());
            for i in 0..n_ss {
                for j in 0..n_ss {
                    let d1 = row as f64 + (i as f64 + 0.5) / n_ss as f64;
                    let d2 = ((col * n_ss + j) as f64 + 0.5) / n_ss as f64;
                    let r = transform.tth(d1, d2);
                    if r >= lo && r <= hi {
                        *total += 1.0 / (n_ss * n_ss) as f64;
                        expected_bins[(((r - lo) / (hi - lo)) * n_points as f64) as usize] += 1;
                    }
                }
            }
        }
        let totals = pixel_totals(&lut, geometry.size());
        assert_eq!(totals, expected);
        // The range cuts the detector: pixels fully inside, fully outside and split.
        assert!(totals.contains(&1.0));
        assert!(totals.contains(&0.0));
        assert!(totals.iter().any(|&t| t > 0.0 && t < 1.0));
        // In particular the edge bins hold only their own subpixels.
        let values = lut.values.as_ref().unwrap();
        for bin in [0, n_points - 1] {
            let sum: f64 = values[lut.row_ptr[bin]..lut.row_ptr[bin + 1]].iter().sum();
            assert_eq!(
                sum,
                expected_bins[bin] as f64 / (n_ss * n_ss) as f64,
                "bin {bin}"
            );
        }
    }

    #[test]
    fn fast_1d_builder_matches_exact_across_edge_cases() {
        let make_geo = |dim1: usize,
                        dim2: usize,
                        poni1: f64,
                        poni2: f64,
                        distance: f64,
                        rot1: f64,
                        rot2: f64,
                        rot3: f64| {
            let mut poni = Poni::new();
            poni.pixel1 = 1e-4;
            poni.pixel2 = 1.2e-4;
            poni.distance = distance;
            poni.poni1 = poni1;
            poni.poni2 = poni2;
            poni.rot1 = rot1;
            poni.rot2 = rot2;
            poni.rot3 = rot3;
            poni.wavelength = 1e-10;
            Geometry::new(dim1, dim2, poni)
        };

        // Cover: no tilt, strong tilt with rot3 != 0, beam inside / on pixel corner /
        // on pixel edge / off detector, and a steep geometry with z <= 0 pixels.
        let geometries = [
            (
                "untilted_inside",
                make_geo(12, 13, 5.5e-4, 6.6e-4, 0.05, 0.0, 0.0, 0.0),
            ),
            (
                "beam_on_corner",
                make_geo(12, 13, 6.0e-4, 7.2e-4, 0.05, 0.0, 0.0, 0.0),
            ),
            (
                "beam_on_edge",
                make_geo(12, 13, 6.0e-4, 6.6e-4, 0.05, 0.0, 0.0, 0.0),
            ),
            (
                "beam_off_det",
                make_geo(12, 13, -3.0e-4, -2.0e-4, 0.05, 0.0, 0.0, 0.0),
            ),
            (
                "tilted_rot3",
                make_geo(12, 13, 5.2e-4, 6.1e-4, 0.04, 0.35, -0.28, 0.42),
            ),
            (
                "z_non_positive",
                make_geo(12, 13, 5.0e-4, 6.0e-4, 0.0004, 1.3, 0.2, -0.3),
            ),
        ];

        let units = [
            RadialUnit::TthRad,
            RadialUnit::TthDeg,
            RadialUnit::QAngstrom,
            RadialUnit::QNm,
        ];

        // Verify the z <= 0 case actually exercises negative/zero z.
        let z_geo = &geometries.last().unwrap().1;
        let z_plane = z_geo.subpixel_transform().affine_plane().unwrap();
        let has_non_pos_z = (0..=z_geo.dim1()).any(|r| {
            (0..=z_geo.dim2())
                .any(|c| z_plane.0[2] + r as f64 * z_plane.1[2] + c as f64 * z_plane.2[2] <= 0.0)
        });
        assert!(has_non_pos_z, "z_non_positive fixture must include z <= 0");

        for (gname, geo) in &geometries {
            let tth = geo.calc_tth_all();
            let transform = geo.subpixel_transform();
            for &unit in &units {
                let radial = unit.convert_from_tth(&tth, geo.poni.wavelength);
                let conversion = RadialConversion::new(unit, geo.poni.wavelength);
                // n_points = 1, 2, 15, and 260 (hits single-bin, two-bin, span <= 16, and span >= 17).
                for &n_points in &[1usize, 2, 15, 260] {
                    for &n_ss in &[1usize, 2, 3, 8] {
                        // Clamped mode
                        let (hmin, _hmax, hstep) =
                            histogram::ss_tth_min_max_step(geo, &radial, n_points, n_ss, unit);
                        let sample_clamped = |d1: f64, d2: f64| {
                            let r = conversion.convert(transform.tth(d1, d2));
                            Some(histogram::get_bin_clamped(r, hmin, hstep, n_points))
                        };
                        let fast = build_counted_1d_csr(
                            geo,
                            n_ss,
                            n_points,
                            hmin,
                            hstep,
                            true,
                            &conversion,
                            sample_clamped,
                        );
                        let exact = build_counted_csr(geo, n_ss, n_points, false, sample_clamped);
                        assert_eq!(
                            fast, exact,
                            "clamped mismatch: {gname} {unit:?} n_points={n_points} n_ss={n_ss}"
                        );

                        // Explicit range cutting through the detector + negative hist_min
                        if n_ss <= 3 {
                            let span = (hstep * n_points as f64).max(1e-6);
                            let ranges = [
                                (hmin + 0.2 * span, hmin + 0.8 * span),
                                (-0.1 * span, hmin + 0.6 * span),
                            ];
                            for (rmin, rmax) in ranges {
                                let (emin, emax, estep) =
                                    histogram::min_max_step_from_range(rmin, rmax, n_points);
                                let sample_range = |d1: f64, d2: f64| {
                                    let r = conversion.convert(transform.tth(d1, d2));
                                    histogram::get_bin_in_range(r, emin, emax, estep, n_points)
                                };
                                let fast_r = build_counted_1d_csr(
                                    geo,
                                    n_ss,
                                    n_points,
                                    emin,
                                    estep,
                                    false,
                                    &conversion,
                                    sample_range,
                                );
                                let exact_r =
                                    build_counted_csr(geo, n_ss, n_points, false, sample_range);
                                assert_eq!(
                                    fast_r, exact_r,
                                    "range mismatch: {gname} {unit:?} n_points={n_points} n_ss={n_ss} ({rmin}, {rmax})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fast_2d_builder_matches_exact_across_edge_cases() {
        let make_geo = |dim1: usize,
                        dim2: usize,
                        poni1: f64,
                        poni2: f64,
                        distance: f64,
                        rot1: f64,
                        rot2: f64,
                        rot3: f64| {
            let mut poni = Poni::new();
            poni.pixel1 = 1e-4;
            poni.pixel2 = 1.2e-4;
            poni.distance = distance;
            poni.poni1 = poni1;
            poni.poni2 = poni2;
            poni.rot1 = rot1;
            poni.rot2 = rot2;
            poni.rot3 = rot3;
            poni.wavelength = 1e-10;
            Geometry::new(dim1, dim2, poni)
        };

        let geometries = [
            (
                "untilted_inside",
                make_geo(12, 13, 5.5e-4, 6.6e-4, 0.05, 0.0, 0.0, 0.0),
            ),
            (
                "beam_on_corner",
                make_geo(12, 13, 6.0e-4, 7.2e-4, 0.05, 0.0, 0.0, 0.0),
            ),
            (
                "beam_on_edge",
                make_geo(12, 13, 6.0e-4, 6.6e-4, 0.05, 0.0, 0.0, 0.0),
            ),
            (
                "beam_off_det",
                make_geo(12, 13, -3.0e-4, -2.0e-4, 0.05, 0.0, 0.0, 0.0),
            ),
            (
                "tilted_rot3",
                make_geo(12, 13, 5.2e-4, 6.1e-4, 0.04, 0.35, -0.28, 0.42),
            ),
            (
                "z_non_positive",
                make_geo(12, 13, 5.0e-4, 6.0e-4, 0.0004, 1.3, 0.2, -0.3),
            ),
        ];

        let units = [
            RadialUnit::TthRad,
            RadialUnit::TthDeg,
            RadialUnit::QAngstrom,
            RadialUnit::QNm,
        ];

        for (gname, geo) in &geometries {
            let tth = geo.calc_tth_all();
            let az = geo.calc_azimuth_all();
            let transform = geo.subpixel_transform();
            for &unit in &units {
                let radial = unit.convert_from_tth(&tth, geo.poni.wavelength);
                let conversion = RadialConversion::new(unit, geo.poni.wavelength);
                for &(n_radial, n_azimuthal) in &[(1usize, 1usize), (2, 2), (11, 13), (40, 36)] {
                    for &n_ss in &[1usize, 2, 3, 5] {
                        // Clamped 2D mode
                        let (rmin, _, rstep) =
                            histogram::ss_tth_min_max_step(geo, &radial, n_radial, n_ss, unit);
                        let (amin, _, astep) = histogram::supersampled_azimuthal_min_max_step(
                            geo,
                            &az,
                            n_azimuthal,
                            n_ss,
                        );
                        let sample_clamped = |d1: f64, d2: f64| {
                            let (t, a) = transform.tth_and_azimuth(d1, d2);
                            let rb = histogram::get_bin_clamped(
                                conversion.convert(t),
                                rmin,
                                rstep,
                                n_radial,
                            );
                            let ab = histogram::get_bin_clamped(a, amin, astep, n_azimuthal);
                            Some(ab * n_radial + rb)
                        };
                        let fast = build_counted_2d_csr(
                            geo,
                            n_ss,
                            n_radial,
                            n_azimuthal,
                            rmin,
                            rstep,
                            &conversion,
                            AzimuthalSegments::clamped(amin, astep, n_azimuthal),
                            sample_clamped,
                        );
                        let exact = build_counted_csr(
                            geo,
                            n_ss,
                            n_radial * n_azimuthal,
                            true,
                            sample_clamped,
                        );
                        assert_eq!(
                            fast, exact,
                            "2d clamped mismatch: {gname} {unit:?} {n_radial}x{n_azimuthal} n_ss={n_ss}"
                        );

                        // Wrapped 2D mode (both full circle and shifted/partial step)
                        let az_start = -std::f64::consts::PI + 0.17;
                        let az_step = std::f64::consts::TAU / n_azimuthal as f64;
                        let min_az = az_start - 0.5 * az_step;
                        let sample_wrapped = |d1: f64, d2: f64| {
                            let (t, a) = transform.tth_and_azimuth(d1, d2);
                            let rb = histogram::get_bin_clamped(
                                conversion.convert(t),
                                rmin,
                                rstep,
                                n_radial,
                            );
                            let ab = histogram::get_bin_wrapped(a, min_az, az_step, n_azimuthal);
                            Some(ab * n_radial + rb)
                        };
                        let fast_w = build_counted_2d_csr(
                            geo,
                            n_ss,
                            n_radial,
                            n_azimuthal,
                            rmin,
                            rstep,
                            &conversion,
                            AzimuthalSegments::wrapped(min_az, az_step, n_azimuthal),
                            sample_wrapped,
                        );
                        let exact_w = build_counted_csr(
                            geo,
                            n_ss,
                            n_radial * n_azimuthal,
                            true,
                            sample_wrapped,
                        );
                        assert_eq!(
                            fast_w, exact_w,
                            "2d wrapped mismatch: {gname} {unit:?} {n_radial}x{n_azimuthal} n_ss={n_ss}"
                        );
                    }
                }
            }
        }
    }
}

/// Build a supersampled 2D CSR LUT using an explicit azimuthal binning definition.
///
/// `azimuthal_start` is the center of the first azimuthal bin; the effective start edge is
/// `azimuthal_start - azimuthal_step / 2`. Binning wraps periodically so angles outside the
/// total range fold back into `0..n_azimuthal`.
#[allow(clippy::too_many_arguments)]
pub fn build_lut2d_ss_csr_with_azimuthal_range(
    geometry: &Geometry,
    radial: &[f64],
    n_radial: usize,
    n_azimuthal: usize,
    n_ss: usize,
    azimuthal_start: f64,
    azimuthal_step: f64,
    unit: RadialUnit,
) -> Lut2d {
    assert!(azimuthal_step > 0.0, "azimuthal_step must be positive");
    let (radial_min, _radial_max, radial_step) =
        histogram::ss_tth_min_max_step(geometry, radial, n_radial, n_ss, unit);
    let min_azimuthal = azimuthal_start - 0.5 * azimuthal_step;

    let transform = geometry.subpixel_transform();
    let conversion = RadialConversion::new(unit, geometry.poni.wavelength);
    let az_segs = AzimuthalSegments::wrapped(min_azimuthal, azimuthal_step, n_azimuthal);
    let (row_ptr, col_idx, values) = build_counted_2d_csr(
        geometry,
        n_ss,
        n_radial,
        n_azimuthal,
        radial_min,
        radial_step,
        &conversion,
        az_segs,
        |d1, d2| {
            let (tth, azimuth) = transform.tth_and_azimuth(d1, d2);
            let rb = histogram::get_bin_clamped(
                conversion.convert(tth),
                radial_min,
                radial_step,
                n_radial,
            );
            let ab =
                histogram::get_bin_wrapped(azimuth, min_azimuthal, azimuthal_step, n_azimuthal);
            Some(ab * n_radial + rb)
        },
    );

    Lut2d {
        row_ptr,
        col_idx,
        values: Some(values),
        n_tth: n_radial,
        n_azimuthal,
    }
}

#[cfg(test)]
#[path = "supersampled_parity_tests.rs"]
mod parity_tests;
