use rayon::prelude::*;
use std::collections::HashMap;

use crate::geometry::Geometry;
use crate::math::histogram;
use crate::lut::common::subgrid;

pub fn lut_2d_ss(
    geometry: &Geometry,
    tth: &Vec<f64>,
    azi: &Vec<f64>,
    n_tth: usize,
    n_azi: usize,
    n_ss: usize,
) -> Vec<Vec<HashMap<usize, f64>>> {
    let mut lut_2d: Vec<Vec<HashMap<usize, f64>>> = vec![vec![HashMap::new(); n_azi]; n_tth];

    let (tth_min, _max, tth_step) = histogram::ss_tth_min_max_step(&geometry, &tth, n_tth, n_ss);
    let (azi_min, _max, azi_step) =
        histogram::supersampled_chi_min_max_step(&geometry, &azi, n_azi, n_ss);

    for row in 0..geometry.dim1 {
        let (d1, d2, actual_index) = subgrid(row, geometry.dim2, n_ss);
        let (tth, azi) = geometry.calc_tth_and_chi(&d1, &d2);
        let tth_indices: Vec<usize> = tth
            .par_iter()
            .map(|tth| histogram::get_bin(*tth, tth_min, tth_step))
            .collect();
        let azi_indices: Vec<usize> = azi
            .par_iter()
            .map(|azi| histogram::get_bin(*azi, azi_min, azi_step))
            .collect();

        for (i, (tth_index, azi_index)) in tth_indices.iter().zip(&azi_indices).enumerate() {
            let azi_capped = azi_index % n_azi;
            let count = lut_2d[*tth_index][azi_capped]
                .entry(actual_index[i])
                .or_insert(0.0);
            *count += 1.0;
        }
    }

    let lut_len = lut_2d
        .par_iter()
        .map(|v| {
            v.par_iter()
                .map(|hash| hash.values().sum::<f64>())
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    lut_2d
        .par_iter()
        .enumerate()
        .map(|(i, v)| {
            v.par_iter()
                .enumerate()
                .map(|(j, hash)| {
                    hash.par_iter()
                        .map(|(k, v)| (*k, v / lut_len[i][j]))
                        .collect()
                })
                .collect::<Vec<HashMap<usize, f64>>>()
        })
        .collect()
}

pub fn integrate2d_ss(intensity: &Vec<f64>, lut: &Vec<Vec<HashMap<usize, f64>>>) -> Vec<f64> {
    let n_points_tth = lut.len();
    let n_points_azi = lut[0].len();

    let indices: Vec<(usize, usize)> = (0..n_points_tth * n_points_azi)
        .map(|i| (i % n_points_tth, i / n_points_tth))
        .collect();

    let cake_intensity: Vec<f64> = indices
        .par_iter()
        .map(|(tth_idx, azi_idx)| {
            lut[*tth_idx][*azi_idx]
                .iter()
                .map(|(j, v)| intensity[*j] * v)
                .sum::<f64>()
        })
        .collect();
    cake_intensity
}

