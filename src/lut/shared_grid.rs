//! Fractional pixel contributions on explicit shared edges. Legacy builders
//! retain their historical row normalization and implicit detector-local grids.
use super::Lut2d;
use super::pixel_split::{clip_polygon_rect_scratch, polygon_area};
use crate::geometry::Geometry;
use crate::integration::{IntegrationGrid2d, IntegrationMethod, PixelSplitMode, RadialUnit};
use rayon::prelude::*;
use std::f64::consts::TAU;

pub(crate) fn build(
    geometry: &Geometry,
    radial: &[f64],
    azimuthal: &[f64],
    grid: &IntegrationGrid2d,
    method: IntegrationMethod,
    unit: RadialUnit,
) -> Lut2d {
    let corners = if matches!(
        method,
        IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull
    ) {
        let (d1, d2) = geometry.corner_grid();
        let (tth, azimuthal) = geometry.calc_tth_and_azimuth(&d1, &d2);
        Some((
            unit.convert_from_tth(&tth, geometry.poni.wavelength),
            azimuthal,
        ))
    } else {
        None
    };
    let transform = geometry.subpixel_transform();
    let convert = |tth: f64| match unit {
        RadialUnit::TthRad => tth,
        RadialUnit::TthDeg => tth.to_degrees(),
        RadialUnit::QAngstrom => {
            4.0 * std::f64::consts::PI / (geometry.poni.wavelength * 1e10) * (tth * 0.5).sin()
        }
        RadialUnit::QNm => {
            4.0 * std::f64::consts::PI / (geometry.poni.wavelength * 1e9) * (tth * 0.5).sin()
        }
    };
    let pixels: Vec<Vec<(usize, f64)>> = (0..radial.len())
        .into_par_iter()
        .map(|pixel| {
            let mut entries = Vec::new();
            match method {
                IntegrationMethod::Counting => {
                    if let Some(bin) = grid.bin(radial[pixel], azimuthal[pixel]) {
                        entries.push((bin, 1.0));
                    }
                }
                IntegrationMethod::Supersampled { n_ss } => {
                    entries = sample_pixel(
                        geometry,
                        &transform,
                        pixel,
                        grid,
                        n_ss,
                        &[(0.5, 0.5)],
                        &convert,
                    );
                }
                IntegrationMethod::PixelSplit | IntegrationMethod::PyfaiFull => {
                    let (r, a) = corners.as_ref().unwrap();
                    let width = geometry.dim2 + 1;
                    let base = pixel / geometry.dim2 * width + pixel % geometry.dim2;
                    let indices = [base, base + 1, base + width + 1, base + width];
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
                            &convert,
                        );
                    }
                    if grid.pixel_split_mode == PixelSplitMode::PyfaiHistogram {
                        // Reverse to pyFAI's corner order, preserving raw angles.
                        let poly =
                            [base, base + width, base + width + 1, base + 1].map(|i| (r[i], a[i]));
                        if let Some(entries) = super::pyfai_split::split(poly, grid) {
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
                IntegrationMethod::Histogram => unreachable!("validated before LUT construction"),
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

fn sample_pixel<F: Fn(f64) -> f64>(
    geometry: &Geometry,
    transform: &crate::geometry::SubpixelTransform,
    pixel: usize,
    grid: &IntegrationGrid2d,
    n_ss: usize,
    offsets: &[(f64, f64)],
    convert: &F,
) -> Vec<(usize, f64)> {
    let row = pixel / geometry.dim2;
    let col = pixel % geometry.dim2;
    let weight = 1.0 / ((n_ss * n_ss) as f64 * offsets.len() as f64);
    let mut entries = Vec::new();
    for i in 0..n_ss {
        for j in 0..n_ss {
            for &(off1, off2) in offsets {
                let d1 = row as f64 + (i as f64 + off1) / n_ss as f64;
                let d2 = ((col * n_ss + j) as f64 + off2) / n_ss as f64;
                let (tth, azimuthal) = transform.tth_and_azimuth(d1, d2);
                if let Some(bin) = grid.bin(convert(tth), azimuthal) {
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
