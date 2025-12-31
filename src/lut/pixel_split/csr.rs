use crate::lut::supersampled::csr::CsrLutSs;
use crate::lut::supersampled::csr::CsrLut2dSs;
use crate::math::histogram;

pub fn lut_ps_csr(tth: &Vec<f64>, n_points: usize) -> (CsrLutSs, f64, f64) {
    let (min, _max, step) = histogram::min_max_step(tth, n_points);

    // Count contributions per bin
    let mut counts = vec![0usize; n_points];
    for &t in tth.iter() {
        let pos = (t - min) / step;
        if pos < 0.0 {
            continue;
        }
        let b0 = pos.floor() as isize;
        if b0 >= 0 && (b0 as usize) < n_points {
            counts[b0 as usize] += 1;
        }
        let b1 = b0 + 1;
        if (b1 as usize) < n_points {
            counts[b1 as usize] += 1;
        }
    }

    // Prefix sum
    let mut row_ptr = vec![0usize; n_points + 1];
    for i in 0..n_points {
        row_ptr[i + 1] = row_ptr[i] + counts[i];
    }
    let total = *row_ptr.last().unwrap();
    let mut col_idx = vec![0usize; total];
    let mut values = vec![0.0f64; total];
    let mut next = row_ptr.clone();

    // Second pass: fill indices and raw weights
    for (i, &t) in tth.iter().enumerate() {
        let pos = (t - min) / step;
        if pos < 0.0 {
            continue;
        }
        let b0 = pos.floor() as isize;
        let frac = pos - (b0 as f64);

        if b0 >= 0 && (b0 as usize) < n_points {
            let row = b0 as usize;
            let pos_w = next[row];
            col_idx[pos_w] = i;
            values[pos_w] = 1.0 - frac;
            next[row] += 1;
        }
        let b1 = b0 + 1;
        if (b1 as usize) < n_points {
            let row = b1 as usize;
            let pos_w = next[row];
            col_idx[pos_w] = i;
            values[pos_w] = frac;
            next[row] += 1;
        }
    }

    // Normalize per row
    for i in 0..n_points {
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        let s: f64 = values[start..end].iter().sum();
        if s > 0.0 {
            for v in &mut values[start..end] {
                *v /= s;
            }
        }
    }

    (
        CsrLutSs {
            row_ptr,
            col_idx,
            values,
        },
        min,
        step,
    )
}

pub fn lut2d_ps_csr(
    tth: &Vec<f64>,
    azi: &Vec<f64>,
    n_tth: usize,
    n_azi: usize,
) -> CsrLut2dSs {
    let (min_tth, _max_tth, step_tth) = histogram::min_max_step(tth, n_tth);
    let (min_azi, _max_azi, step_azi) = histogram::min_max_step(azi, n_azi);

    let rows = n_tth * n_azi;
    let mut counts = vec![0usize; rows];

    // Count contributions per (tth,azi) row
    for i in 0..tth.len() {
        let ut = (tth[i] - min_tth) / step_tth;
        let ua = (azi[i] - min_azi) / step_azi;
        if ut < 0.0 || ua < 0.0 {
            continue;
        }
        let it = ut.floor() as isize;
        let ia = ua.floor() as isize;
        let candidates = [(it, ia), (it + 1, ia), (it, ia + 1), (it + 1, ia + 1)];
        for (bt, ba) in candidates {
            if bt >= 0 && ba >= 0 {
                let bt = bt as usize;
                let ba = (ba as usize) % n_azi;
                if bt < n_tth {
                    counts[ba * n_tth + bt] += 1;
                }
            }
        }
    }

    // Prefix sum
    let mut row_ptr = vec![0usize; rows + 1];
    for r in 0..rows {
        row_ptr[r + 1] = row_ptr[r] + counts[r];
    }
    let total = *row_ptr.last().unwrap();
    let mut col_idx = vec![0usize; total];
    let mut values = vec![0.0f64; total];
    let mut next = row_ptr.clone();

    // Fill indices and raw bilinear weights
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
                    let row = ba * n_tth + bt;
                    let pos_w = next[row];
                    col_idx[pos_w] = i;
                    values[pos_w] = w;
                    next[row] += 1;
                }
            }
        }
    }

    // Normalize each row to sum to 1
    for r in 0..rows {
        let start = row_ptr[r];
        let end = row_ptr[r + 1];
        let s: f64 = values[start..end].iter().sum();
        if s > 0.0 {
            for v in &mut values[start..end] {
                *v /= s;
            }
        }
    }

    CsrLut2dSs {
        row_ptr,
        col_idx,
        values,
        n_tth,
        n_azi,
    }
}
