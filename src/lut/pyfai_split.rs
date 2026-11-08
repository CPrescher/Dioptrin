//! Opt-in approximation of pyFAI 2025.12.1 fullSplit2D_engine's histogram rule.
//! This reproduces its line-area rasterization and vertex-clipping convention,
//! not geometric polygon intersections. Singular beam pixels are handled by the
//! caller; negative numerical raster areas are discarded before normalization.
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
use crate::integration::IntegrationGrid2d;
use std::f64::consts::{PI, TAU};

/// Input order matches pyFAI: top-left, bottom-left, bottom-right, top-right.
/// Returning None requests the caller's nonnegative fractional-area fallback.
pub(super) fn split(
    mut poly: [(f64, f64); 4],
    grid: &IntegrationGrid2d,
) -> Option<Vec<(usize, f64)>> {
    // Recenter at the canonical discontinuity, independently of sector width.
    // pyFAI's reference test is a signed diagonal-area heuristic, rather than a
    // winding test. Keep it here for compatibility, after the singularity guard.
    let (a0, a1) = grid.azimuthal_range();
    // Preserve the reference's canonical discontinuity for ordinary sectors;
    // do not recenter at a sector edge. Positive intervals extending beyond π
    // use chiDiscAtZero. Translate other full-turn intervals consistently.
    let canonical_start = if a0 >= -PI && a1 <= PI {
        -PI
    } else if a0 >= 0.0 && a1 <= TAU {
        0.0
    } else {
        a0
    };
    let offset = canonical_start + PI;
    for p in &mut poly {
        p.1 = (p.1 - offset + PI).rem_euclid(TAU) - PI;
    }
    let [a, b, c, d] = poly;
    let area = 0.5 * (c.0 - a.0) * (d.1 - b.1) - (c.1 - a.1) * (d.0 - b.0);
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
    let (r0, r1) = grid.radial_range();
    if poly.iter().all(|p| p.0 < r0)
        || poly.iter().all(|p| p.0 > r1)
        || poly.iter().all(|p| p.1 < a0)
        || poly.iter().all(|p| p.1 > a1)
    {
        return Some(Vec::new());
    }
    let dr = grid.radial_step();
    let da = grid.azimuthal_step();
    for p in &mut poly {
        p.0 = (p.0.clamp(r0, r1) - r0) / dr;
        p.1 = (p.1.clamp(a0, a1) - a0) / da;
    }
    let ir = poly
        .iter()
        .map(|p| p.0)
        .fold(f64::INFINITY, f64::min)
        .floor() as usize;
    let ia = poly
        .iter()
        .map(|p| p.1)
        .fold(f64::INFINITY, f64::min)
        .floor() as usize;
    for p in &mut poly {
        p.0 -= ir as f64;
        p.1 -= ia as f64;
    }
    let nr = poly.iter().map(|p| p.0).fold(0.0, f64::max).ceil() as usize;
    let na = poly.iter().map(|p| p.1).fold(0.0, f64::max).ceil() as usize;
    if nr == 0 || na == 0 {
        return None;
    }
    let mut buffer = vec![0.0_f32; (nr + 1) * (na + 1)];
    for i in 0..4 {
        line(&mut buffer, na + 1, poly[i], poly[(i + 1) % 4]);
    }
    let mut entries = Vec::new();
    let mut total = 0.0;
    for r in 0..nr {
        for a in 0..na {
            let value = buffer[r * (na + 1) + a] as f64;
            if !value.is_finite() {
                return None;
            }
            if value > 0.0 && ir + r < grid.n_radial && ia + a < grid.n_azimuthal {
                entries.push(((ia + a) * grid.n_radial + ir + r, value));
                total += value;
            }
        }
    }
    if total <= 0.0 {
        return None;
    }
    for (_, w) in &mut entries {
        *w /= total;
    }
    Some(entries)
}

fn line(buffer: &mut [f32], stride: usize, start: (f64, f64), stop: (f64, f64)) {
    if start.0 == stop.0 {
        return;
    }
    let slope = ((stop.1 - start.1) / (stop.0 - start.0)) as f32;
    let intercept = (stop.1 - slope as f64 * stop.0) as f32;
    let lo = start.0.min(stop.0);
    let hi = start.0.max(stop.0);
    let sign = if stop.0 > start.0 { 1.0 } else { -1.0 };
    // Split the line at integer radial columns. The reference distributes the
    // integrated signed area through azimuthal rows in strips of this width;
    // that differs from clipping the actual line/polygon against each row.
    for r in lo.floor() as usize..hi.ceil() as usize {
        let left = lo.max(r as f64);
        let right = hi.min((r + 1) as f64);
        let area = (sign
            * (right - left)
            * (0.5 * slope as f64 * (right + left) + intercept as f64)) as f32;
        let width = (right - left) as f32;
        let mut remaining = area.abs();
        for a in 0..stride {
            if remaining <= 0.0 {
                break;
            }
            let increment = width.min(remaining);
            buffer[r * stride + a] += increment.copysign(area);
            remaining -= increment;
        }
    }
}
