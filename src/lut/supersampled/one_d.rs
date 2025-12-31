use rayon::prelude::*;
use std::collections::HashMap;

use crate::geometry::Geometry;
use crate::math::histogram;
use crate::lut::common::subgrid;

pub fn lut_ss(
    geometry: &Geometry,
    tth: &Vec<f64>,
    n_points: usize,
    n_ss: usize,
) -> Vec<HashMap<usize, f64>> {
    let mut lut: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n_points];

    let (min, _max, step) = histogram::ss_tth_min_max_step(&geometry, &tth, n_points, n_ss);

    for row in 0..geometry.dim1 {
        let (d1, d2, actual_index) = subgrid(row, geometry.dim2, n_ss);
        let tth = geometry.calc_tth(&d1, &d2);
        let tth_bin_indices: Vec<usize> = tth
            .par_iter()
            .map(|tth| histogram::get_bin(*tth, min, step))
            .collect();

        for (i, bin_index) in tth_bin_indices.iter().enumerate() {
            let count = lut[*bin_index].entry(actual_index[i]).or_insert(0.0);
            *count += 1.0;
        }
    }

    let lut_len = lut
        .par_iter()
        .map(|hash| hash.values().sum::<f64>())
        .collect::<Vec<f64>>();

    lut.par_iter()
        .enumerate()
        .map(|(i, hash)| {
            hash.par_iter()
                .map(|(k, v)| (*k, v / lut_len[i]))
                .collect()
        })
        .collect()
}

pub fn integrate_ss(
    intensity: &Vec<f64>,
    lut: &Vec<HashMap<usize, f64>>,
    hist_min: f64,
    hist_step: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_points = lut.len();
    let mut pattern_intensity: Vec<f64> = vec![0.0; n_points];

    pattern_intensity
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, int)| {
            *int += lut[i].iter().map(|(j, v)| intensity[*j] * *v).sum::<f64>();
        });

    let pattern_tth = histogram::bins(n_points, hist_min, hist_step);

    (pattern_tth, pattern_intensity)
}

