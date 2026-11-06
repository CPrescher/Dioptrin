//! Geometry-only CSC execution for masks that change between frames.
//! Float64 coefficients are transposed without rounding or renormalization.
use rayon::prelude::*;

use super::prepared::{Columns, madd};
use crate::frame::Pixel;

pub(super) struct Csc {
    ptr: Vec<u32>,
    bins: Vec<u32>,
    weights: Option<Vec<f64>>,
    nbins: usize,
}

struct Sums {
    signal: Vec<f64>,
    norm: Vec<f64>,
    variance: Vec<f64>,
    count: Vec<f64>,
}

impl Csc {
    pub(super) fn new(
        rows: &[usize],
        columns: &Columns,
        weights: Option<&[f64]>,
        npix: usize,
    ) -> Option<Self> {
        let nbins = rows.len().checked_sub(1)?;
        let nnz = *rows.last()?;
        if u32::try_from(nnz).is_err()
            || u32::try_from(nbins).is_err()
            || u32::try_from(npix).is_err()
        {
            return None;
        }
        if let Some(w) = weights {
            assert_eq!(w.len(), nnz);
        }
        let mut ptr = vec![0u32; npix.checked_add(1)?];
        for k in 0..nnz {
            ptr[columns.get(k) + 1] += 1;
        }
        for p in 0..npix {
            ptr[p + 1] += ptr[p];
        }
        let mut next = ptr[..npix].to_vec();
        let mut bins = vec![0; nnz];
        let mut transposed_weights = weights.map(|_| vec![0.; nnz]);
        for b in 0..nbins {
            for k in rows[b]..rows[b + 1] {
                let p = columns.get(k);
                let at = next[p] as usize;
                next[p] += 1;
                bins[at] = b as u32;
                if let Some(w) = transposed_weights.as_mut() {
                    w[at] = weights.unwrap()[k];
                }
            }
        }
        Some(Self {
            ptr,
            bins,
            weights: transposed_weights,
            nbins,
        })
    }

    fn accumulate<const VAR: bool, const COUNT: bool, P: Pixel>(
        &self,
        signal: &[P],
        mask: Option<&[u8]>,
        norm: &[f64],
        start: usize,
        end: usize,
    ) -> Sums {
        let size = self.nbins * 4;
        let mut sums = Sums {
            signal: vec![0.; size],
            norm: vec![0.; size],
            variance: vec![0.; if VAR { size } else { 0 }],
            count: vec![0.; if COUNT { size } else { 0 }],
        };
        // SAFETY: construction checks compact bounds and builds bins from valid
        // CSR row numbers. Integrator validates frame/mask dimensions; norm is
        // geometry-sized. Each invocation owns all destination histograms.
        unsafe {
            for p in start..end {
                if mask.is_some_and(|m| *m.get_unchecked(p) != 0) {
                    continue;
                }
                let x = signal.get_unchecked(p).to_f64();
                let n = *norm.get_unchecked(p);
                let lane = (p % 4) * self.nbins;
                for k in
                    *self.ptr.get_unchecked(p) as usize..*self.ptr.get_unchecked(p + 1) as usize
                {
                    let b = *self.bins.get_unchecked(k) as usize + lane;
                    let w = self
                        .weights
                        .as_ref()
                        .map_or(1., |weights| *weights.get_unchecked(k));
                    let s = sums.signal.get_unchecked_mut(b);
                    *s = madd(w, x, *s);
                    let s = sums.norm.get_unchecked_mut(b);
                    *s = madd(w, n, *s);
                    if VAR {
                        let s = sums.variance.get_unchecked_mut(b);
                        *s = madd(w * w, x.max(0.), *s);
                    }
                    if COUNT {
                        *sums.count.get_unchecked_mut(b) += w;
                    }
                }
            }
        }
        sums
    }

    pub(super) fn results<const VAR: bool, const COUNT: bool, P: Pixel>(
        &self,
        signal: &[P],
        mask: Option<&[u8]>,
        norm: &[f64],
    ) -> Vec<[f64; 4]> {
        assert_eq!(signal.len() + 1, self.ptr.len());
        assert_eq!(signal.len(), norm.len());
        if let Some(m) = mask {
            assert_eq!(signal.len(), m.len());
        }
        // A batch is already parallel over frames. For a direct single-frame
        // call, split pixels over the pool with private histograms. Bound their
        // memory for large 2D outputs. No Rayon work is queued during OnceLock
        // table initialization, so concurrent first use cannot block the pool.
        let bytes_per_hist = self
            .nbins
            .saturating_mul(4 * 8 * (2 + usize::from(VAR) + usize::from(COUNT)));
        let parallel = rayon::current_thread_index().is_none()
            && signal.len() >= 262_144
            && bytes_per_hist > 0;
        let chunks = if parallel {
            rayon::current_num_threads()
                .min(signal.len().div_ceil(262_144))
                .min((32 * 1024 * 1024 / bytes_per_hist).max(1))
                .max(1)
        } else {
            1
        };
        let partials: Vec<Sums> = if chunks == 1 {
            vec![self.accumulate::<VAR, COUNT, P>(signal, mask, norm, 0, signal.len())]
        } else {
            (0..chunks)
                .into_par_iter()
                .map(|i| {
                    self.accumulate::<VAR, COUNT, P>(
                        signal,
                        mask,
                        norm,
                        i * signal.len() / chunks,
                        (i + 1) * signal.len() / chunks,
                    )
                })
                .collect()
        };
        let reduce = |v: &[f64], b: usize| {
            (v[b] + v[self.nbins + b]) + (v[2 * self.nbins + b] + v[3 * self.nbins + b])
        };
        let mut rows = vec![[0.; 4]; self.nbins];
        for partial in partials {
            for (b, row) in rows.iter_mut().enumerate() {
                row[0] += reduce(&partial.signal, b);
                row[1] += reduce(&partial.norm, b);
                if VAR {
                    row[2] += reduce(&partial.variance, b);
                }
                if COUNT {
                    row[3] += reduce(&partial.count, b);
                }
            }
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_preserves_float64_weight_bits() {
        let columns = Columns::Compact(vec![0, 1, 0]);
        let weights = [0.1, 0.3, 0.15];
        let csc = Csc::new(&[0, 2, 3], &columns, Some(&weights), 2).unwrap();
        assert_eq!(csc.ptr, vec![0, 2, 3]);
        assert_eq!(csc.bins, vec![0, 1, 0]);
        assert_eq!(
            csc.weights
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            [weights[0], weights[2], weights[1]].map(f64::to_bits)
        );
        #[cfg(target_pointer_width = "64")]
        assert!(
            Csc::new(
                &[0, u32::MAX as usize + 1],
                &Columns::Compact(vec![]),
                None,
                0
            )
            .is_none()
        );
    }

    #[test]
    fn parallel_single_frame_and_batch_histograms_match_independent_raw_sums() {
        let npix = 300_001;
        let bins = 7;
        let mut row_ptr = vec![0];
        let mut columns = Vec::new();
        let mut weights = Vec::new();
        for b in 0..bins {
            for p in (b..npix).step_by(bins) {
                columns.push(p as u32);
                weights.push(0.1 + (p % 5) as f64 * 0.037);
            }
            row_ptr.push(columns.len());
        }
        let mask: Vec<_> = (0..npix).map(|p| u8::from(p % 29 == 0)).collect();
        let signal: Vec<_> = (0..npix)
            .map(|p| {
                if mask[p] != 0 {
                    f64::NAN
                } else {
                    0.1 + (p as f64 * 0.23).sin() * 100.
                }
            })
            .collect();
        let norm: Vec<_> = (0..npix).map(|p| 0.95 + (p % 11) as f64 * 0.001).collect();
        let mut expected = vec![[0.; 4]; bins];
        for (b, row) in expected.iter_mut().enumerate() {
            for k in row_ptr[b]..row_ptr[b + 1] {
                let p = columns[k] as usize;
                if mask[p] != 0 {
                    continue;
                }
                let w = weights[k];
                row[0] += w * signal[p];
                row[1] += w * norm[p];
                row[2] += w * w * signal[p].max(0.);
                row[3] += w;
            }
        }
        let table = Csc::new(&row_ptr, &Columns::Compact(columns), Some(&weights), npix).unwrap();
        let direct = table.results::<true, true, _>(&signal, Some(&mask), &norm);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let batch: Vec<_> = pool.install(|| {
            (0..4)
                .into_par_iter()
                .map(|_| table.results::<true, true, _>(&signal, Some(&mask), &norm))
                .collect()
        });
        for result in std::iter::once(&direct).chain(&batch) {
            for (a, b) in result.iter().flatten().zip(expected.iter().flatten()) {
                assert!((a - b).abs() <= 2e-11 * b.abs().max(1.), "{a} vs {b}");
            }
        }
    }
}
