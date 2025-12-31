use std::collections::HashMap;

use crate::math::histogram;

pub fn lut_ps(tth: &Vec<f64>, n_points: usize) -> Vec<HashMap<usize, f64>> {
    let (min, _max, step) = histogram::min_max_step(tth, n_points);
    let mut bins: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n_points];

    for (i, &t) in tth.iter().enumerate() {
        let pos = (t - min) / step;
        if pos < 0.0 {
            continue;
        }
        let b0 = pos.floor() as isize;
        let frac = pos - (b0 as f64);

        if b0 >= 0 && (b0 as usize) < n_points {
            let w0 = 1.0 - frac;
            if w0 > 0.0 {
                let entry = bins[b0 as usize].entry(i).or_insert(0.0);
                *entry += w0;
            }
        }
        let b1 = b0 + 1;
        if (b1 as usize) < n_points {
            let w1 = frac;
            if w1 > 0.0 {
                let entry = bins[b1 as usize].entry(i).or_insert(0.0);
                *entry += w1;
            }
        }
    }

    // normalize per bin
    for m in &mut bins {
        let s: f64 = m.values().sum();
        if s > 0.0 {
            for v in m.values_mut() {
                *v /= s;
            }
        }
    }
    bins
}

