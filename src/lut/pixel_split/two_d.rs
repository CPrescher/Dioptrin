use std::collections::HashMap;

use crate::math::histogram;

pub fn lut_2d_ps(
    tth: &Vec<f64>,
    azi: &Vec<f64>,
    n_tth: usize,
    n_azi: usize,
) -> Vec<Vec<HashMap<usize, f64>>> {
    let (min_tth, _max_tth, step_tth) = histogram::min_max_step(tth, n_tth);
    let (min_azi, _max_azi, step_azi) = histogram::min_max_step(azi, n_azi);

    let mut bins: Vec<Vec<HashMap<usize, f64>>> = vec![vec![HashMap::new(); n_azi]; n_tth];

    for i in 0..tth.len() {
        let ut = (tth[i] - min_tth) / step_tth;
        let ua = (azi[i] - min_azi) / step_azi;
        if ut < 0.0 || ua < 0.0 {
            continue;
        }
        let it = ut.floor() as isize;
        let ia = ua.floor() as isize;
        let dt = ut - (it as f64);
        let da = ua - (ia as f64);
        let candidates = [
            (it, ia, (1.0 - dt) * (1.0 - da)),
            (it + 1, ia, dt * (1.0 - da)),
            (it, ia + 1, (1.0 - dt) * da),
            (it + 1, ia + 1, dt * da),
        ];
        for (bt, ba, w) in candidates {
            if w <= 0.0 {
                continue;
            }
            if bt >= 0 && ba >= 0 {
                let bt = bt as usize;
                let ba = (ba as usize) % n_azi;
                if bt < n_tth {
                    let entry = bins[bt][ba].entry(i).or_insert(0.0);
                    *entry += w;
                }
            }
        }
    }

    // normalize per bin
    for row in &mut bins {
        for m in row {
            let s: f64 = m.values().sum();
            if s > 0.0 {
                for v in m.values_mut() {
                    *v /= s;
                }
            }
        }
    }
    bins
}

