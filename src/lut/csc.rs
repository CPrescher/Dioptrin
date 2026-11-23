//! Geometry-only CSC execution for masks that change between frames.
//! Float64 coefficients are transposed without rounding or renormalization.
use rayon::prelude::*;

use super::prepared::{Columns, madd};
use crate::frame::Pixel;

/// Compact bin index storage: `u16` whenever `nbins <= 65,535` (halves index
/// bandwidth for all standard 1D integrations), falling back to `u32`.
pub(crate) enum BinIndices {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl BinIndices {
    #[inline(always)]
    fn get(&self, k: usize) -> usize {
        match self {
            Self::U16(v) => v[k] as usize,
            Self::U32(v) => v[k] as usize,
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, k: usize) -> usize {
        match self {
            Self::U16(v) => unsafe { *v.get_unchecked(k) as usize },
            Self::U32(v) => unsafe { *v.get_unchecked(k) as usize },
        }
    }

    #[cfg(test)]
    fn to_vec_u32(&self) -> Vec<u32> {
        match self {
            Self::U16(v) => v.iter().map(|&b| u32::from(b)).collect(),
            Self::U32(v) => v.clone(),
        }
    }
}

pub(crate) struct Csc {
    ptr: Vec<u32>,
    bins: BinIndices,
    weights: Option<Vec<f64>>,
    nbins: usize,
}

pub(crate) struct Sums {
    pub(crate) signal: Vec<f64>,
    norm: Vec<f64>,
    pub(crate) variance: Vec<f64>,
    count: Vec<f64>,
}

/// Bin-oriented resident-frame execution in exactly the fixed CSC reduction
/// order. Rows retain the original chunk and pixel-modulo-four lanes.
pub(crate) struct FixedRows {
    ptr: Vec<u32>,
    pixels: Vec<u32>,
    weights: Option<Vec<f64>>,
    nbins: usize,
    chunks: usize,
    npix: usize,
}

impl FixedRows {
    pub(crate) fn new(csc: &Csc, pixels: &[u32]) -> Option<Self> {
        // Keep streamed/block plans lightweight. This optional resident cache is
        // lazy and bounded; sparse tables beyond this budget retain CSC.
        let chunks = csc.fixed_chunks(pixels.len());
        if csc.nbins < 64 || chunks < 2 {
            return None;
        }
        let entries: usize = pixels
            .iter()
            .map(|&p| (csc.ptr[p as usize + 1] - csc.ptr[p as usize]) as usize)
            .sum();
        let rows = csc.nbins.checked_mul(chunks)?.checked_mul(4)?;
        let bytes = entries
            .checked_mul(if csc.weights.is_some() { 12 } else { 4 })?
            .checked_add(rows.checked_add(1)?.checked_mul(4)?)?;
        if bytes > 128 * 1024 * 1024 {
            return None;
        }
        let mut ptr = vec![0u32; rows + 1];
        for chunk in 0..chunks {
            for &p in &pixels[chunk * pixels.len() / chunks..(chunk + 1) * pixels.len() / chunks] {
                for k in csc.ptr[p as usize] as usize..csc.ptr[p as usize + 1] as usize {
                    ptr[(csc.bins.get(k) * chunks + chunk) * 4 + p as usize % 4 + 1] += 1;
                }
            }
        }
        for i in 0..rows {
            ptr[i + 1] += ptr[i];
        }
        let mut next = ptr[..rows].to_vec();
        let mut columns = vec![0u32; entries];
        let mut weights = csc.weights.as_ref().map(|_| vec![0.; entries]);
        for chunk in 0..chunks {
            for &p in &pixels[chunk * pixels.len() / chunks..(chunk + 1) * pixels.len() / chunks] {
                for k in csc.ptr[p as usize] as usize..csc.ptr[p as usize + 1] as usize {
                    let row = (csc.bins.get(k) * chunks + chunk) * 4 + p as usize % 4;
                    let at = next[row] as usize;
                    next[row] += 1;
                    columns[at] = p;
                    if let Some(w) = weights.as_mut() {
                        w[at] = csc.weights.as_ref().unwrap()[k];
                    }
                }
            }
        }
        Some(Self {
            ptr,
            pixels: columns,
            weights,
            nbins: csc.nbins,
            chunks,
            npix: csc.ptr.len() - 1,
        })
    }
    pub(crate) fn integrate<const VAR: bool>(
        &self,
        data: &[f64],
        norm: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(data.len(), self.npix);
        assert_eq!(norm.len(), self.nbins);
        let rows: Vec<_> = (0..self.nbins)
            .into_par_iter()
            .map(|bin| {
                let mut signal = 0.;
                let mut variance = 0.;
                for chunk in 0..self.chunks {
                    let mut s = [0.; 4];
                    let mut v = [0.; 4];
                    // SAFETY: construction copies validated CSC columns/weights
                    // and stable chunk partitions; Fixed1dPlan checks data length.
                    unsafe {
                        for lane in 0..4 {
                            let row = (bin * self.chunks + chunk) * 4 + lane;
                            let mut sig = 0.;
                            let mut var = 0.;
                            for k in self.ptr[row] as usize..self.ptr[row + 1] as usize {
                                let p = *self.pixels.get_unchecked(k) as usize;
                                let x = *data.get_unchecked(p);
                                let w = self
                                    .weights
                                    .as_ref()
                                    .map_or(1., |weights| *weights.get_unchecked(k));
                                sig = madd(w, x, sig);
                                if VAR {
                                    var = madd(w * w, x.max(0.), var);
                                }
                            }
                            s[lane] = sig;
                            v[lane] = var;
                        }
                    }
                    signal += (s[0] + s[1]) + (s[2] + s[3]);
                    if VAR {
                        variance += (v[0] + v[1]) + (v[2] + v[3]);
                    }
                }
                let n = norm[bin];
                (
                    if n == 0. { 0. } else { signal / n },
                    if VAR && n != 0. {
                        (variance / (n * n)).max(0.).sqrt()
                    } else {
                        0.
                    },
                )
            })
            .collect();
        let intensity = rows.iter().map(|r| r.0).collect();
        let errors = if VAR {
            rows.iter().map(|r| r.1).collect()
        } else {
            vec![]
        };
        (intensity, errors)
    }
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
        let mut transposed_weights = weights.map(|_| vec![0.; nnz]);
        let bins = if u16::try_from(nbins).is_ok() {
            let mut b16 = vec![0u16; nnz];
            for b in 0..nbins {
                for k in rows[b]..rows[b + 1] {
                    let p = columns.get(k);
                    let at = next[p] as usize;
                    next[p] += 1;
                    b16[at] = b as u16;
                    if let Some(w) = transposed_weights.as_mut() {
                        w[at] = weights.unwrap()[k];
                    }
                }
            }
            BinIndices::U16(b16)
        } else {
            let mut b32 = vec![0u32; nnz];
            for b in 0..nbins {
                for k in rows[b]..rows[b + 1] {
                    let p = columns.get(k);
                    let at = next[p] as usize;
                    next[p] += 1;
                    b32[at] = b as u32;
                    if let Some(w) = transposed_weights.as_mut() {
                        w[at] = weights.unwrap()[k];
                    }
                }
            }
            BinIndices::U32(b32)
        };
        Some(Self {
            ptr,
            bins,
            weights: transposed_weights,
            nbins,
        })
    }

    /// Prepared once per fixed mask. Keep original geometry/weight storage;
    /// only this compact pixel selection and bin normalization depend on mask.
    pub(super) fn active_pixels(&self, mask: Option<&[u8]>) -> Vec<u32> {
        if let Some(mask) = mask {
            assert_eq!(mask.len() + 1, self.ptr.len());
        }
        (0..self.ptr.len() - 1)
            .filter(|&p| self.ptr[p] != self.ptr[p + 1] && !mask.is_some_and(|m| m[p] != 0))
            .map(|p| p as u32)
            .collect()
    }

    /// Fixed-mask 1D execution: no per-frame mask checks or normalization sums.
    /// Coefficients are shared with the dynamic path; every call owns its sums.
    pub(super) fn fixed_results<const VAR: bool, const COUNT: bool, P: Pixel>(
        &self,
        signal: &[P],
        pixels: &[u32],
        normalization: &[f64],
    ) -> Vec<[f64; 4]> {
        assert_eq!(signal.len() + 1, self.ptr.len());
        assert_eq!(normalization.len(), self.nbins);
        // Partition by geometry, not the active execution pool or requested
        // output fields. Direct, file-batch and array-batch APIs then preserve
        // the same reduction order across worker counts and pixel types.
        let chunks = self.fixed_chunks(pixels.len());
        let accumulate = |i| {
            self.fixed_accumulate::<VAR, COUNT, P>(
                signal,
                &pixels[i * pixels.len() / chunks..(i + 1) * pixels.len() / chunks],
            )
        };
        let n = self.nbins;
        let reduce = |v: &[f64], b: usize| (v[b] + v[n + b]) + (v[2 * n + b] + v[3 * n + b]);
        let mut rows: Vec<_> = normalization
            .iter()
            .map(|&norm| [0., norm, 0., 0.])
            .collect();
        if rayon::current_thread_index().is_none() && chunks > 1 {
            let partials: Vec<Sums> = (0..chunks).into_par_iter().map(accumulate).collect();
            for sums in partials {
                for (b, row) in rows.iter_mut().enumerate() {
                    row[0] += reduce(&sums.signal, b);
                    if VAR {
                        row[2] += reduce(&sums.variance, b);
                    }
                    if COUNT {
                        row[3] += reduce(&sums.count, b);
                    }
                }
            }
        } else {
            // Reuse one Sums buffer across chunks when running sequentially on a worker.
            let mut sums = self.block_sums(VAR);
            if COUNT {
                sums.count = vec![0.; 4 * n];
            }
            for i in 0..chunks {
                if i > 0 {
                    sums.signal.fill(0.);
                    if VAR {
                        sums.variance.fill(0.);
                    }
                    if COUNT {
                        sums.count.fill(0.);
                    }
                }
                self.add_fixed_slice::<VAR, COUNT, P>(
                    signal,
                    0,
                    &pixels[i * pixels.len() / chunks..(i + 1) * pixels.len() / chunks],
                    &mut sums,
                );
                for (b, row) in rows.iter_mut().enumerate() {
                    row[0] += reduce(&sums.signal, b);
                    if VAR {
                        row[2] += reduce(&sums.variance, b);
                    }
                    if COUNT {
                        row[3] += reduce(&sums.count, b);
                    }
                }
            }
        }
        rows
    }

    /// 4-frame SIMD-tiled fixed-mask 1D integration. Reads the CSC table once
    /// per 4 frames and accumulates `[f64; 4]` per bin/lane in the exact same
    /// chunk and lane reduction order as 1-frame `fixed_results` (bit-identical).
    #[cfg(feature = "python")]
    pub(super) fn fixed_results_batch4<const VAR: bool>(
        &self,
        signals: [&[f64]; 4],
        pixels: &[u32],
        normalization: &[f64],
    ) -> [(Vec<f64>, Vec<f64>); 4] {
        for s in signals {
            assert_eq!(s.len() + 1, self.ptr.len());
        }
        assert_eq!(normalization.len(), self.nbins);
        let n = self.nbins;
        let chunks = self.fixed_chunks(pixels.len());
        let mut sig_acc = vec![[0.0f64; 4]; n];
        let mut var_acc = if VAR {
            vec![[0.0f64; 4]; n]
        } else {
            Vec::new()
        };
        let mut chunk_sig = vec![[0.0f64; 4]; 4 * n];
        let mut chunk_var = if VAR {
            vec![[0.0f64; 4]; 4 * n]
        } else {
            Vec::new()
        };

        macro_rules! run_chunk {
            ($bins:expr, $weights:expr, $weighted:expr, $pix_slice:expr) => {{
                let ptr = self.ptr.as_ptr();
                let bins = $bins.as_ptr();
                let weights = $weights;
                let s0 = signals[0].as_ptr();
                let s1 = signals[1].as_ptr();
                let s2 = signals[2].as_ptr();
                let s3 = signals[3].as_ptr();
                let cs = chunk_sig.as_mut_ptr();
                let cv = chunk_var.as_mut_ptr();
                for &pixel in $pix_slice {
                    let p = pixel as usize;
                    let x0 = *s0.add(p);
                    let x1 = *s1.add(p);
                    let x2 = *s2.add(p);
                    let x3 = *s3.add(p);
                    let xp0 = if VAR { x0.max(0.0) } else { 0.0 };
                    let xp1 = if VAR { x1.max(0.0) } else { 0.0 };
                    let xp2 = if VAR { x2.max(0.0) } else { 0.0 };
                    let xp3 = if VAR { x3.max(0.0) } else { 0.0 };
                    let lane = (p & 3) * n;
                    let k0 = *ptr.add(p) as usize;
                    let k1 = *ptr.add(p + 1) as usize;
                    for k in k0..k1 {
                        let b = *bins.add(k) as usize + lane;
                        let dst_s = &mut *cs.add(b);
                        if $weighted {
                            let w = *weights.add(k);
                            dst_s[0] = madd(w, x0, dst_s[0]);
                            dst_s[1] = madd(w, x1, dst_s[1]);
                            dst_s[2] = madd(w, x2, dst_s[2]);
                            dst_s[3] = madd(w, x3, dst_s[3]);
                            if VAR {
                                let w2 = w * w;
                                let dst_v = &mut *cv.add(b);
                                dst_v[0] = madd(w2, xp0, dst_v[0]);
                                dst_v[1] = madd(w2, xp1, dst_v[1]);
                                dst_v[2] = madd(w2, xp2, dst_v[2]);
                                dst_v[3] = madd(w2, xp3, dst_v[3]);
                            }
                        } else {
                            dst_s[0] += x0;
                            dst_s[1] += x1;
                            dst_s[2] += x2;
                            dst_s[3] += x3;
                            if VAR {
                                let dst_v = &mut *cv.add(b);
                                dst_v[0] += xp0;
                                dst_v[1] += xp1;
                                dst_v[2] += xp2;
                                dst_v[3] += xp3;
                            }
                        }
                    }
                }
            }};
        }

        for i in 0..chunks {
            if i > 0 {
                chunk_sig.fill([0.0; 4]);
                if VAR {
                    chunk_var.fill([0.0; 4]);
                }
            }
            let pix_slice = &pixels[i * pixels.len() / chunks..(i + 1) * pixels.len() / chunks];
            // SAFETY: CSC construction checks all bin indices and pixel pointers;
            // all 4 signal slices are checked above to match `ptr.len() - 1`.
            unsafe {
                match (&self.bins, &self.weights) {
                    (BinIndices::U16(b), Some(w)) => run_chunk!(b, w.as_ptr(), true, pix_slice),
                    (BinIndices::U16(b), None) => {
                        run_chunk!(b, std::ptr::null::<f64>(), false, pix_slice)
                    }
                    (BinIndices::U32(b), Some(w)) => run_chunk!(b, w.as_ptr(), true, pix_slice),
                    (BinIndices::U32(b), None) => {
                        run_chunk!(b, std::ptr::null::<f64>(), false, pix_slice)
                    }
                }
            }
            for b in 0..n {
                let a0 = chunk_sig[b];
                let a1 = chunk_sig[n + b];
                let a2 = chunk_sig[2 * n + b];
                let a3 = chunk_sig[3 * n + b];
                for f in 0..4 {
                    sig_acc[b][f] += (a0[f] + a1[f]) + (a2[f] + a3[f]);
                }
                if VAR {
                    let v0 = chunk_var[b];
                    let v1 = chunk_var[n + b];
                    let v2 = chunk_var[2 * n + b];
                    let v3 = chunk_var[3 * n + b];
                    for f in 0..4 {
                        var_acc[b][f] += (v0[f] + v1[f]) + (v2[f] + v3[f]);
                    }
                }
            }
        }

        std::array::from_fn(|f| {
            let mut intensity = Vec::with_capacity(n);
            let mut errors = Vec::with_capacity(if VAR { n } else { 0 });
            for b in 0..n {
                let norm = normalization[b];
                intensity.push(if norm == 0.0 {
                    0.0
                } else {
                    sig_acc[b][f] / norm
                });
                if VAR {
                    let v = if norm == 0.0 {
                        0.0
                    } else {
                        var_acc[b][f] / (norm * norm)
                    };
                    errors.push(v.max(0.0).sqrt());
                }
            }
            (intensity, errors)
        })
    }

    fn fixed_chunks(&self, pixels: usize) -> usize {
        let bytes_per_hist = self.nbins.saturating_mul(4 * 8 * 3);
        (32 * 1024 * 1024_usize)
            .checked_div(bytes_per_hist)
            .map_or(1, |limit| pixels.div_ceil(262_144).min(limit.max(1)).max(1))
    }

    pub(crate) fn block_sums(&self, errors: bool) -> Sums {
        Sums {
            signal: vec![0.; 4 * self.nbins],
            norm: vec![],
            variance: vec![0.; if errors { 4 * self.nbins } else { 0 }],
            count: vec![],
        }
    }

    pub(crate) fn dynamic_sums(&self, errors: bool) -> Sums {
        let mut sums = self.block_sums(errors);
        sums.norm = vec![0.; 4 * self.nbins];
        sums
    }

    /// Reuse private histograms; geometry/weights are independent of each mask.
    /// A supplied variance is already propagated from the original counts.
    pub(crate) fn dynamic_frame(
        &self,
        signal: &[f64],
        variance: Option<&[f64]>,
        mask: Option<&[u8]>,
        norm: &[f64],
        sums: &mut Sums,
    ) -> (Vec<f64>, Option<Vec<f64>>) {
        assert_eq!(signal.len() + 1, self.ptr.len());
        assert_eq!(norm.len(), signal.len());
        assert!(mask.is_none_or(|v| v.len() == signal.len()));
        assert!(variance.is_none_or(|v| v.len() == signal.len()));
        sums.signal.fill(0.);
        sums.norm.fill(0.);
        sums.variance.fill(0.);
        let errors = !sums.variance.is_empty();
        // SAFETY: compact indices originate from checked CSR construction;
        // the complete input dimensions are checked above. Sums are private.
        unsafe {
            for p in 0..signal.len() {
                if mask.is_some_and(|m| *m.get_unchecked(p) != 0) {
                    continue;
                }
                let x = *signal.get_unchecked(p);
                let n = *norm.get_unchecked(p);
                let v = if errors {
                    variance.map_or_else(|| x.max(0.), |v| *v.get_unchecked(p))
                } else {
                    0.
                };
                let lane = (p % 4) * self.nbins;
                for k in
                    *self.ptr.get_unchecked(p) as usize..*self.ptr.get_unchecked(p + 1) as usize
                {
                    let b = self.bins.get_unchecked(k) + lane;
                    let w = self.weights.as_ref().map_or(1., |w| *w.get_unchecked(k));
                    let s = sums.signal.get_unchecked_mut(b);
                    *s = madd(w, x, *s);
                    let nsum = sums.norm.get_unchecked_mut(b);
                    *nsum = madd(w, n, *nsum);
                    if errors {
                        let var = sums.variance.get_unchecked_mut(b);
                        *var = madd(w * w, v, *var);
                    }
                }
            }
        }
        let n = self.nbins;
        let reduce = |s: &[f64], b| (s[b] + s[n + b]) + (s[2 * n + b] + s[3 * n + b]);
        let mut intensity = Vec::with_capacity(n);
        let mut sigma = errors.then(|| Vec::with_capacity(n));
        for b in 0..n {
            let norm = reduce(&sums.norm, b);
            intensity.push(if norm == 0. {
                0.
            } else {
                reduce(&sums.signal, b) / norm
            });
            if let Some(sigma) = &mut sigma {
                let var = if norm == 0. {
                    0.
                } else {
                    reduce(&sums.variance, b) / (norm * norm)
                };
                sigma.push(var.max(0.).sqrt());
            }
        }
        (intensity, sigma)
    }

    pub(crate) fn add_fixed_block<P: Pixel>(
        &self,
        signal: &[P],
        offset: usize,
        pixels: &[u32],
        sums: &mut Sums,
    ) {
        if sums.variance.is_empty() {
            self.add_fixed_slice::<false, false, P>(signal, offset, pixels, sums);
        } else {
            self.add_fixed_slice::<true, false, P>(signal, offset, pixels, sums);
        }
    }

    fn fixed_accumulate<const VAR: bool, const COUNT: bool, P: Pixel>(
        &self,
        signal: &[P],
        pixels: &[u32],
    ) -> Sums {
        let mut sums = self.block_sums(VAR);
        if COUNT {
            sums.count = vec![0.; 4 * self.nbins];
        }
        self.add_fixed_slice::<VAR, COUNT, P>(signal, 0, pixels, &mut sums);
        sums
    }

    // Caller selects only active pixels in [offset, offset + signal.len()).
    fn add_fixed_slice<const VAR: bool, const COUNT: bool, P: Pixel>(
        &self,
        signal: &[P],
        offset: usize,
        pixels: &[u32],
        sums: &mut Sums,
    ) {
        let n = self.nbins;
        // SAFETY: Csc construction checks compact bounds and active_pixels
        // supplies valid pixels. Signal dimensions are checked above. All
        // destination histograms are private to this frame invocation.
        macro_rules! run_slice {
            ($bins:expr, $weights:expr, $weighted:expr) => {{
                let ptr = self.ptr.as_ptr();
                let bins = $bins.as_ptr();
                let weights = $weights;
                let sig_ptr = sums.signal.as_mut_ptr();
                let var_ptr = sums.variance.as_mut_ptr();
                let cnt_ptr = sums.count.as_mut_ptr();
                for &pixel in pixels {
                    let p = pixel as usize;
                    let x = signal.get_unchecked(p - offset).to_f64();
                    let xp = if VAR { x.max(0.) } else { 0. };
                    let lane = (p & 3) * n;
                    let k0 = *ptr.add(p) as usize;
                    let k1 = *ptr.add(p + 1) as usize;
                    for k in k0..k1 {
                        let b = *bins.add(k) as usize + lane;
                        if $weighted {
                            let w = *weights.add(k);
                            let s = &mut *sig_ptr.add(b);
                            *s = madd(w, x, *s);
                            if VAR {
                                let v = &mut *var_ptr.add(b);
                                *v = madd(w * w, xp, *v);
                            }
                            if COUNT {
                                *cnt_ptr.add(b) += w;
                            }
                        } else {
                            *sig_ptr.add(b) += x;
                            if VAR {
                                *var_ptr.add(b) += xp;
                            }
                            if COUNT {
                                *cnt_ptr.add(b) += 1.0;
                            }
                        }
                    }
                }
            }};
        }
        unsafe {
            match (&self.bins, &self.weights) {
                (BinIndices::U16(b), Some(w)) => run_slice!(b, w.as_ptr(), true),
                (BinIndices::U16(b), None) => run_slice!(b, std::ptr::null::<f64>(), false),
                (BinIndices::U32(b), Some(w)) => run_slice!(b, w.as_ptr(), true),
                (BinIndices::U32(b), None) => run_slice!(b, std::ptr::null::<f64>(), false),
            }
        }
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
                    let b = self.bins.get_unchecked(k) + lane;
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
    fn resident_rows_preserve_chunk_lane_bits_masks_errors_and_thread_counts() {
        let npix = 540_013;
        let nbins = 128;
        let mut ptr = vec![0];
        let mut cols = Vec::new();
        let mut weights = Vec::new();
        for b in 0..nbins {
            for phase in [b, (b + nbins - 1) % nbins] {
                for p in (phase..npix).step_by(nbins) {
                    cols.push(p as u32);
                    weights.push((p % 41 + 1) as f64 / 43.);
                    if p % 131 == 0 {
                        cols.push(p as u32);
                        weights.push(0.125);
                    }
                }
            }
            ptr.push(cols.len());
        }
        let cols = Columns::Compact(cols);
        let mut normalization = vec![1.3; nbins];
        normalization[0] = 0.;
        normalization[1] = 1e-280;
        for weighted in [false, true] {
            let csc = Csc::new(&ptr, &cols, weighted.then_some(weights.as_slice()), npix).unwrap();
            for masked in [false, true] {
                let mask: Vec<_> = (0..npix).map(|p| u8::from(p % 17 == 0)).collect();
                let selected = csc.active_pixels(masked.then_some(mask.as_slice()));
                let rows = FixedRows::new(&csc, &selected).unwrap();
                assert!(rows.chunks > 1);
                for factor in [1., -3.7] {
                    let mut data: Vec<_> = (0..npix)
                        .map(|p| ((p * 997) % 113) as f64 * factor - 5.3)
                        .collect();
                    if masked {
                        data[0] = f64::NAN;
                    }
                    let reference =
                        csc.fixed_results::<true, false, _>(&data, &selected, &normalization);
                    let expected_y: Vec<_> = reference
                        .iter()
                        .map(|r| if r[1] == 0. { 0. } else { r[0] / r[1] })
                        .collect();
                    let expected_e: Vec<_> = reference
                        .iter()
                        .map(|r| {
                            if r[1] == 0. {
                                0.
                            } else {
                                (r[2] / (r[1] * r[1])).max(0.).sqrt()
                            }
                        })
                        .collect();
                    let check = || {
                        let (y, e) = rows.integrate::<true>(&data, &normalization);
                        assert_eq!(
                            y.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            expected_y.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                        );
                        assert_eq!(
                            e.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            expected_e.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                        );
                        let (y, e) = rows.integrate::<false>(&data, &normalization);
                        assert!(e.is_empty());
                        assert_eq!(
                            y.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            expected_y.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                        );
                    };
                    check();
                    if masked && weighted && factor == 1. {
                        for threads in [1, 4, 8] {
                            rayon::ThreadPoolBuilder::new()
                                .num_threads(threads)
                                .build()
                                .unwrap()
                                .install(check);
                        }
                    }
                }
            }
        }
        let small = Csc::new(&[0, 1], &Columns::Compact(vec![0]), None, 1).unwrap();
        assert!(FixedRows::new(&small, &[0]).is_none());
    }

    #[test]
    fn transpose_preserves_float64_weight_bits() {
        let columns = Columns::Compact(vec![0, 1, 0]);
        let weights = [0.1, 0.3, 0.15];
        let csc = Csc::new(&[0, 2, 3], &columns, Some(&weights), 2).unwrap();
        assert_eq!(csc.ptr, vec![0, 2, 3]);
        assert_eq!(csc.bins.to_vec_u32(), vec![0, 1, 0]);
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

#[cfg(test)]
mod fixed_partition_tests {
    use super::*;
    #[test]
    fn fixed_reduction_is_identical_across_direct_and_frame_worker_pools() {
        let npix = 3 * 262_144 + 17;
        let mut rows = vec![0];
        let mut cols = Vec::new();
        let mut weights = Vec::new();
        for b in 0..7 {
            for p in (b..npix).step_by(7) {
                cols.push(p as u32);
                weights.push((p % 13 + 1) as f64 / 17.);
            }
            rows.push(cols.len());
        }
        let table = Csc::new(&rows, &Columns::Compact(cols), Some(&weights), npix).unwrap();
        let mask: Vec<_> = (0..npix).map(|p| u8::from(p % 29 == 0)).collect();
        let pixels = table.active_pixels(Some(&mask));
        let signal: Vec<_> = (0..npix).map(|p| (p % 997) as f64 / 31.).collect();
        let norm = vec![17.; 7];
        let direct = table.fixed_results::<true, true, _>(&signal, &pixels, &norm);
        for threads in [1, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let inside =
                pool.install(|| table.fixed_results::<true, true, _>(&signal, &pixels, &norm));
            assert_eq!(direct, inside);
        }
    }
}
