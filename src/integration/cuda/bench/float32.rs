//! Paired f32 experiment using the CPU's existing sparse layouts.
use super::*;
use crate::lut::prepared::Prepared1d;

enum Indices<'a> {
    Csr {
        rows: &'a [usize],
        columns: &'a [u32],
        fast: bool,
    },
    Csc {
        ptr: &'a [u32],
        bins: std::borrow::Cow<'a, [u32]>,
        active: &'a [u32],
        chunks: usize,
    },
}

pub(super) struct Float32<'a> {
    lut: &'a Prepared1d,
    indices: Indices<'a>,
    weights: Vec<f32>,
    norm: Vec<f32>,
    pixels: usize,
}

impl<'a> Float32<'a> {
    pub(super) fn new(lut: &'a Prepared1d, cpu: &Integrator, norm: &[f64]) -> Result<Self> {
        let mask = cpu.mask().map(|m| m.data.as_slice());
        let (indices, weights, denominator) =
            if cpu.fast_math() && cpu.sparse_layout() == SparseLayout::Csc {
                lut.prepare_normalization(mask, norm, true);
                let csc = lut
                    .fixed_csc()
                    .ok_or("f32 experiment requires compact CSC")?;
                let active = lut.fixed_pixels();
                let (ptr, bins, weights, chunks) = csc.fixed_benchmark_parts(active.len());
                (
                    Indices::Csc {
                        ptr,
                        bins,
                        active,
                        chunks,
                    },
                    weights,
                    lut.fixed_bin_norm(),
                )
            } else {
                let (table, denominator) = lut.fixed_csr_parts(mask, norm);
                let Columns::Compact(columns) = &table.col_idx else {
                    return Err("f32 experiment requires compact CSR".into());
                };
                (
                    Indices::Csr {
                        rows: &table.row_ptr,
                        columns,
                        fast: cpu.fast_math(),
                    },
                    table.values.as_deref(),
                    denominator,
                )
            };
        Ok(Self {
            lut,
            indices,
            weights: weights
                .unwrap_or_default()
                .iter()
                .map(|&w| w as f32)
                .collect(),
            norm: denominator.iter().map(|&n| n as f32).collect(),
            pixels: norm.len(),
        })
    }

    pub(super) fn integrate(&self, data: &[f32], errors: bool) -> (Vec<f64>, Vec<f32>, Vec<f32>) {
        assert_eq!(data.len(), self.pixels);
        let (signal, sigma) = match (errors, !self.weights.is_empty()) {
            (false, false) => self.pattern::<false, false>(data),
            (false, true) => self.pattern::<false, true>(data),
            (true, false) => self.pattern::<true, false>(data),
            (true, true) => self.pattern::<true, true>(data),
        };
        (self.lut.bins(), signal, sigma)
    }

    fn pattern<const VAR: bool, const WEIGHT: bool>(&self, data: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let sums = match self.indices {
            Indices::Csr {
                rows,
                columns,
                fast,
            } => (0..rows.len() - 1)
                .into_par_iter()
                .map(|bin| {
                    if fast {
                        row::<VAR, WEIGHT, true>(
                            rows[bin],
                            rows[bin + 1],
                            columns,
                            &self.weights,
                            data,
                        )
                    } else {
                        row::<VAR, WEIGHT, false>(
                            rows[bin],
                            rows[bin + 1],
                            columns,
                            &self.weights,
                            data,
                        )
                    }
                })
                .collect::<Vec<_>>(),
            Indices::Csc {
                ptr,
                ref bins,
                active,
                chunks,
            } => {
                let n = self.norm.len();
                let mut sums = vec![[0.; 2]; n];
                // Frame workers already run in Rayon, so the f64 CSC kernel
                // processes its fixed pixel partitions on the calling worker.
                for chunk in 0..chunks {
                    let pixels =
                        &active[chunk * active.len() / chunks..(chunk + 1) * active.len() / chunks];
                    let mut part = [vec![0f32; n * 4], vec![0f32; if VAR { n * 4 } else { 0 }]];
                    let [signal, variance] = &mut part;
                    // SAFETY: Csc checks all offsets, bins and pixel indices.
                    // These are its active pixels and this is a full frame.
                    unsafe {
                        for &p in pixels {
                            let p = p as usize;
                            let x = *data.get_unchecked(p);
                            let lane = p % 4 * n;
                            for k in
                                *ptr.get_unchecked(p) as usize..*ptr.get_unchecked(p + 1) as usize
                            {
                                let b = lane + *bins.get_unchecked(k) as usize;
                                let w = if WEIGHT {
                                    *self.weights.get_unchecked(k)
                                } else {
                                    1.
                                };
                                let signal = signal.get_unchecked_mut(b);
                                *signal = madd32(w, x, *signal);
                                if VAR {
                                    let variance = variance.get_unchecked_mut(b);
                                    *variance = madd32(w * w, x.max(0.), *variance);
                                }
                            }
                        }
                    }
                    let reduce = |v: &[f32], b| (v[b] + v[n + b]) + (v[2 * n + b] + v[3 * n + b]);
                    for (b, sum) in sums.iter_mut().enumerate() {
                        sum[0] += reduce(signal, b);
                        if VAR {
                            sum[1] += reduce(variance, b);
                        }
                    }
                }
                sums
            }
        };
        let mut intensity = Vec::with_capacity(sums.len());
        let mut sigma = Vec::with_capacity(if VAR { sums.len() } else { 0 });
        for ([s, v], &n) in sums.into_iter().zip(&self.norm) {
            intensity.push(if n == 0. { 0. } else { s / n });
            if VAR {
                sigma.push(if n == 0. {
                    0.
                } else {
                    (v / (n * n)).max(0.).sqrt()
                });
            }
        }
        (intensity, sigma)
    }
}

#[inline(always)]
fn madd32(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        a * b + c
    }
}

#[inline]
fn row<const VAR: bool, const WEIGHT: bool, const FAST: bool>(
    start: usize,
    end: usize,
    columns: &[u32],
    weights: &[f32],
    data: &[f32],
) -> [f32; 2] {
    let mut s = [0.; 4];
    let mut v = [0.; 4];
    let mut visit = |lane: usize, k| {
        // SAFETY: ranges and indices come from the checked fixed CSR and data
        // is the matching full frame. Rounded weights retain the table length.
        unsafe {
            let x = *data.get_unchecked(*columns.get_unchecked(k) as usize);
            let w = if WEIGHT {
                *weights.get_unchecked(k)
            } else {
                1.
            };
            s[lane] = if FAST && WEIGHT {
                madd32(w, x, s[lane])
            } else {
                s[lane] + w * x
            };
            if VAR {
                v[lane] = if FAST && WEIGHT {
                    madd32(w * w, x.max(0.), v[lane])
                } else {
                    v[lane] + w * w * x.max(0.)
                };
            }
        }
    };
    if FAST {
        let mut k = start;
        while k + 4 <= end {
            for lane in 0..4 {
                visit(lane, k + lane);
            }
            k += 4;
        }
        for lane in 0..end - k {
            visit(lane, k + lane);
        }
        [(s[0] + s[1]) + (s[2] + s[3]), (v[0] + v[1]) + (v[2] + v[3])]
    } else {
        for k in start..end {
            visit(0, k);
        }
        [s[0], v[0]]
    }
}
