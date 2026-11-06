use super::prepared::madd;
/// Experimental fixed-mask CSC: mask-compacted active pixels, four f64 partial
/// histograms, and EXACTLY the production CSR's cached bin normalization.
pub(crate) struct FixedCsc {
    pixels: Vec<u32>,
    ptr: Vec<u32>,
    bins: Vec<u32>,
    weights: Vec<f64>,
    norm: Vec<f64>,
}
impl FixedCsc {
    pub(crate) fn new(
        rows: &[usize],
        cols: &[usize],
        weights: &[f64],
        npix: usize,
        norm: Vec<f64>,
    ) -> Self {
        let mut ptr = vec![0u32; npix + 1];
        for &p in cols {
            ptr[p + 1] += 1;
        }
        for p in 0..npix {
            ptr[p + 1] += ptr[p];
        }
        let mut next = ptr[..npix].to_vec();
        let mut bins = vec![0u32; cols.len()];
        let mut w = vec![0.; cols.len()];
        for b in 0..rows.len() - 1 {
            for k in rows[b]..rows[b + 1] {
                let p = cols[k];
                let at = next[p] as usize;
                next[p] += 1;
                bins[at] = b as u32;
                w[at] = weights[k];
            }
        }
        // Check transpose coefficients and ordering bit-for-bit.
        let mut back = rows[..rows.len() - 1].to_vec();
        for p in 0..npix {
            for k in ptr[p] as usize..ptr[p + 1] as usize {
                let b = bins[k] as usize;
                let at = back[b];
                back[b] += 1;
                assert_eq!(cols[at], p);
                assert_eq!(weights[at].to_bits(), w[k].to_bits());
            }
        }
        let pixels: Vec<_> = (0..npix)
            .filter(|&p| ptr[p] != ptr[p + 1])
            .map(|p| p as u32)
            .collect();
        let mut compact: Vec<_> = pixels.iter().map(|&p| ptr[p as usize]).collect();
        compact.push(*ptr.last().unwrap());
        Self {
            pixels,
            ptr: compact,
            bins,
            weights: w,
            norm,
        }
    }
    pub(crate) fn integrate<const ERR: bool, const LANES: usize>(
        &self,
        image: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.norm.len();
        let mut sums = vec![0.; LANES * n];
        let mut variances = vec![0.; if ERR { LANES * n } else { 0 }];
        // SAFETY: construction validates CSR/CSC identity; caller validates
        // image dimensions. Histograms are private to this frame invocation.
        unsafe {
            for a in 0..self.pixels.len() {
                let p = *self.pixels.get_unchecked(a) as usize;
                let x = *image.get_unchecked(p);
                let lane = (p % LANES) * n;
                for k in
                    *self.ptr.get_unchecked(a) as usize..*self.ptr.get_unchecked(a + 1) as usize
                {
                    let b = *self.bins.get_unchecked(k) as usize + lane;
                    let w = *self.weights.get_unchecked(k);
                    let s = sums.get_unchecked_mut(b);
                    *s = madd(w, x, *s);
                    if ERR {
                        let v = variances.get_unchecked_mut(b);
                        *v = madd(w * w, x.max(0.), *v);
                    }
                }
            }
        }
        let reduce = |v: &[f64], b: usize| {
            if LANES == 1 {
                v[b]
            } else if LANES == 2 {
                v[b] + v[n + b]
            } else {
                (v[b] + v[n + b]) + (v[2 * n + b] + v[3 * n + b])
            }
        };
        let mut y = vec![0.; n];
        let mut e = vec![0.; if ERR { n } else { 0 }];
        for b in 0..n {
            if self.norm[b] != 0. {
                y[b] = reduce(&sums, b) / self.norm[b];
                if ERR {
                    e[b] = (reduce(&variances, b) / (self.norm[b] * self.norm[b]))
                        .max(0.)
                        .sqrt();
                }
            }
        }
        (y, e)
    }
}
