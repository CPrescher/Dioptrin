//! Pixel-splitting CSR LUT builders (1D/2D) producing weighted `Lut1d`/`Lut2d`.
//! Integrate with `crate::lut::integrate_lut1d` / `integrate_lut2d`.
use crate::math::histogram;
use crate::lut::counting_csr::{Lut1d, Lut2d};

/// Pixel-split CSR builders (exact fractional/bilinear weights) using unified LUT types.
///
/// Example (1D)
/// ```
/// use dioptrin::lut::{build_lut1d_ps_csr, integrate_lut1d};
/// let tth = vec![0.1, 0.2, 0.3, 0.4];
/// let intensity = vec![10.0, 20.0, 30.0, 40.0];
/// let lut = build_lut1d_ps_csr(&tth, 3);
/// let (_bins, y) = integrate_lut1d(&intensity, &lut);
/// assert_eq!(y.len(), 3);
/// ```
pub fn build_lut1d_ps_csr(tth: &Vec<f64>, n_points: usize) -> Lut1d {
    let (min, _max, step) = histogram::min_max_step(tth, n_points);

    // First pass: count per row (each pixel contributes up to two rows)
    let mut counts = vec![0usize; n_points];
    for &t in tth.iter() {
        let pos = (t - min) / step;
        if pos < 0.0 { continue; }
        let b0 = pos.floor() as isize;
        if b0 >= 0 && (b0 as usize) < n_points { counts[b0 as usize] += 1; }
        let b1 = b0 + 1;
        if (b1 as usize) < n_points { counts[b1 as usize] += 1; }
    }

    // Prefix sum
    let mut row_ptr = vec![0usize; n_points + 1];
    for i in 0..n_points { row_ptr[i + 1] = row_ptr[i] + counts[i]; }
    let total = *row_ptr.last().unwrap();
    let mut col_idx = vec![0usize; total];
    let mut values = vec![0.0f64; total];
    let mut next = row_ptr.clone();

    // Second pass: fill indices and raw weights
    for (i, &t) in tth.iter().enumerate() {
        let pos = (t - min) / step;
        if pos < 0.0 { continue; }
        let b0 = pos.floor() as isize;
        let frac = pos - (b0 as f64);

        if b0 >= 0 && (b0 as usize) < n_points {
            let row = b0 as usize;
            let p = next[row];
            col_idx[p] = i;
            values[p] = 1.0 - frac;
            next[row] += 1;
        }
        let b1 = b0 + 1;
        if (b1 as usize) < n_points {
            let row = b1 as usize;
            let p = next[row];
            col_idx[p] = i;
            values[p] = frac;
            next[row] += 1;
        }
    }

    // Normalize per row
    for i in 0..n_points {
        let (s, e) = (row_ptr[i], row_ptr[i + 1]);
        let ssum: f64 = values[s..e].iter().sum();
        if ssum > 0.0 { for v in &mut values[s..e] { *v /= ssum; } }
    }

    Lut1d { row_ptr, col_idx, values: Some(values), hist_min: min, hist_step: step }
}

/// Build a 2D pixel-split LUT with bilinear weights over `tth`/`azi` bins.
///
/// Example (2D)
/// ```
/// use dioptrin::lut::{build_lut2d_ps_csr, integrate_lut2d};
/// let tth = vec![0.1, 0.2, 0.3, 0.4];
/// let chi = vec![0.0, 0.1, 0.2, 0.3];
/// let intensity = vec![1.0, 2.0, 3.0, 4.0];
/// let lut2d = build_lut2d_ps_csr(&tth, &chi, 2, 3);
/// let y = integrate_lut2d(&intensity, &lut2d);
/// assert_eq!(y.len(), 2*3);
/// ```
pub fn build_lut2d_ps_csr(
    tth: &Vec<f64>,
    azi: &Vec<f64>,
    n_tth: usize,
    n_azi: usize,
) -> Lut2d {
    let (min_tth, _max_tth, step_tth) = histogram::min_max_step(tth, n_tth);
    let (min_azi, _max_azi, step_azi) = histogram::min_max_step(azi, n_azi);

    let rows = n_tth * n_azi;
    let mut counts = vec![0usize; rows];

    for i in 0..tth.len() {
        let ut = (tth[i] - min_tth) / step_tth;
        let ua = (azi[i] - min_azi) / step_azi;
        if ut < 0.0 || ua < 0.0 { continue; }
        let it = ut.floor() as isize;
        let ia = ua.floor() as isize;
        let candidates = [(it, ia), (it + 1, ia), (it, ia + 1), (it + 1, ia + 1)];
        for (bt, ba) in candidates {
            if bt >= 0 && ba >= 0 {
                let bt = bt as usize;
                let ba = (ba as usize) % n_azi;
                if bt < n_tth { counts[ba * n_tth + bt] += 1; }
            }
        }
    }

    let mut row_ptr = vec![0usize; rows + 1];
    for r in 0..rows { row_ptr[r + 1] = row_ptr[r] + counts[r]; }
    let total = *row_ptr.last().unwrap();
    let mut col_idx = vec![0usize; total];
    let mut values = vec![0.0f64; total];
    let mut next = row_ptr.clone();

    for i in 0..tth.len() {
        let ut = (tth[i] - min_tth) / step_tth;
        let ua = (azi[i] - min_azi) / step_azi;
        if ut < 0.0 || ua < 0.0 { continue; }
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
            if w <= 0.0 { continue; }
            if bt >= 0 && ba >= 0 {
                let bt = bt as usize;
                let ba = (ba as usize) % n_azi;
                if bt < n_tth {
                    let row = ba * n_tth + bt;
                    let p = next[row];
                    col_idx[p] = i;
                    values[p] = w;
                    next[row] += 1;
                }
            }
        }
    }

    // Normalize per row
    for r in 0..rows {
        let (s, e) = (row_ptr[r], row_ptr[r + 1]);
        let ssum: f64 = values[s..e].iter().sum();
        if ssum > 0.0 { for v in &mut values[s..e] { *v /= ssum; } }
    }

    Lut2d { row_ptr, col_idx, values: Some(values), n_tth, n_azi }
}
