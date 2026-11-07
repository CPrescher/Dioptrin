//! Internal execution tables. Public LUT builders retain their usize-based API.
//!
//! Pixel order, f64 weights and each arithmetic policy's reduction order are
//! preserved. Only column storage and frame-independent normalization change.
use std::sync::OnceLock;

use super::csc::Csc;
use crate::frame::Pixel;
use crate::integration::{MaskMode, SparseLayout};
use rayon::prelude::*;

use super::{IntermediateResult1d, IntermediateResult2d, Lut1d, Lut2d};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Columns {
    Compact(Vec<u32>),
    Wide(Vec<usize>),
}

impl Columns {
    fn new(indices: Vec<usize>) -> Self {
        // Check before converting; never truncate an unusually large detector.
        if indices.iter().all(|&j| u32::try_from(j).is_ok()) {
            Self::Compact(indices.iter().map(|&j| j as u32).collect())
        } else {
            Self::Wide(indices)
        }
    }

    pub(crate) fn get(&self, k: usize) -> usize {
        match self {
            Self::Compact(v) => v[k] as usize,
            Self::Wide(v) => v[k],
        }
    }

    #[cfg(test)]
    pub(crate) fn as_ptr(&self) -> *const () {
        match self {
            Self::Compact(v) => v.as_ptr().cast(),
            Self::Wide(v) => v.as_ptr().cast(),
        }
    }
}

trait PixelIndex: Copy + Sync {
    fn pixel(self) -> usize;
}
impl PixelIndex for u32 {
    #[inline(always)]
    fn pixel(self) -> usize {
        self as usize
    }
}
impl PixelIndex for usize {
    #[inline(always)]
    fn pixel(self) -> usize {
        self
    }
}

pub(crate) struct PreparedCsr {
    pub(crate) row_ptr: Vec<usize>,
    pub(crate) col_idx: Columns,
    pub(crate) values: Option<Vec<f64>>,
    strict_norm: OnceLock<Vec<f64>>,
    fast_norm: OnceLock<Vec<f64>>,
    mode: MaskMode,
    fixed: OnceLock<Option<Box<PreparedCsr>>>,
    csc: OnceLock<Option<Csc>>,
    fixed_pixels: OnceLock<Vec<u32>>,
    fixed_csc_enabled: bool,
    prefer_csc: bool,
}

pub(crate) struct Prepared1d {
    csr: PreparedCsr,
    hist_min: f64,
    hist_step: f64,
}
pub(crate) struct Prepared2d {
    csr: PreparedCsr,
    pub(crate) n_tth: usize,
    pub(crate) n_azimuthal: usize,
}

impl std::ops::Deref for Prepared1d {
    type Target = PreparedCsr;
    fn deref(&self) -> &PreparedCsr {
        &self.csr
    }
}
impl std::ops::Deref for Prepared2d {
    type Target = PreparedCsr;
    fn deref(&self) -> &PreparedCsr {
        &self.csr
    }
}

impl PreparedCsr {
    fn new(row_ptr: Vec<usize>, col_idx: Vec<usize>, values: Option<Vec<f64>>) -> Self {
        Self {
            row_ptr,
            col_idx: Columns::new(col_idx),
            values,
            strict_norm: OnceLock::new(),
            fast_norm: OnceLock::new(),
            mode: MaskMode::Fixed,
            fixed: OnceLock::new(),
            csc: OnceLock::new(),
            fixed_pixels: OnceLock::new(),
            fixed_csc_enabled: false,
            prefer_csc: true,
        }
    }

    pub(crate) fn clear_normalization(&mut self) {
        self.strict_norm.take();
        self.fast_norm.take();
        if let Some(Some(fixed)) = self.fixed.get_mut() {
            fixed.clear_normalization();
        }
    }

    pub(crate) fn invalidate_mask(&mut self) {
        self.fixed.take();
        self.fixed_pixels.take();
        self.clear_normalization();
        // CSC contains geometry only and remains valid for a different mask.
    }

    pub(crate) fn set_mask_mode(&mut self, mode: MaskMode) {
        if self.mode != mode {
            // A policy switch changes execution, not geometry, mask or
            // corrections. Keep both prepared layouts and denominators valid;
            // their actual input setters invalidate the relevant state.
            self.mode = mode;
        }
    }

    fn fixed_table(&self, mask: Option<&[u8]>) -> &Self {
        let Some(mask) = mask else {
            return self;
        };
        self.fixed
            .get_or_init(|| {
                if mask.iter().all(|&x| x == 0) {
                    return None;
                }
                let mut row_ptr = Vec::with_capacity(self.row_ptr.len());
                let mut columns = Vec::with_capacity(self.col_idx_len());
                let mut weights = self.values.as_ref().map(|w| Vec::with_capacity(w.len()));
                row_ptr.push(0);
                for row in self.row_ptr.windows(2) {
                    for k in row[0]..row[1] {
                        let pixel = self.col_idx.get(k);
                        if mask[pixel] == 0 {
                            columns.push(pixel);
                            if let Some(w) = weights.as_mut() {
                                w.push(self.values.as_ref().unwrap()[k]);
                            }
                        }
                    }
                    row_ptr.push(columns.len());
                }
                Some(Box::new(Self::new(row_ptr, columns, weights)))
            })
            .as_deref()
            .unwrap_or(self)
    }

    fn col_idx_len(&self) -> usize {
        *self.row_ptr.last().unwrap_or(&0)
    }

    fn dynamic_csc(&self, npix: usize) -> Option<&Csc> {
        self.csc
            .get_or_init(|| Csc::new(&self.row_ptr, &self.col_idx, self.values.as_deref(), npix))
            .as_ref()
    }

    fn normalization<const FAST: bool>(&self, mask: Option<&[u8]>, norm: &[f64]) -> &[f64] {
        let cache = if FAST {
            &self.fast_norm
        } else {
            &self.strict_norm
        };
        cache.get_or_init(|| {
            // Serial initialization cannot block a Rayon worker waiting for work
            // queued behind another worker initializing this same OnceLock.
            macro_rules! build {
                ($cols:expr) => {
                    (0..self.row_ptr.len() - 1)
                        .map(|i| {
                            self.sum::<_, f64, FAST, false, true, false, false>(
                                i,
                                $cols,
                                mask,
                                &[],
                                norm,
                            )[1]
                        })
                        .collect()
                };
            }
            match &self.col_idx {
                Columns::Compact(c) => build!(c),
                Columns::Wide(c) => build!(c),
            }
        })
    }

    pub(crate) fn prepare_normalization(&self, mask: Option<&[u8]>, norm: &[f64], fast: bool) {
        match self.mode {
            MaskMode::Fixed => {
                if fast
                    && self.fixed_csc_enabled
                    && self.prefer_csc
                    && let Some(csc) = self.dynamic_csc(norm.len())
                {
                    self.fixed_pixels.get_or_init(|| csc.active_pixels(mask));
                    self.normalization::<true>(mask, norm);
                    return;
                }
                let table = self.fixed_table(mask);
                if fast {
                    table.normalization::<true>(None, norm);
                } else {
                    table.normalization::<false>(None, norm);
                }
            }
            MaskMode::Dynamic if fast && self.prefer_csc => {
                self.dynamic_csc(norm.len());
            }
            MaskMode::Dynamic => {}
        }
    }

    #[inline(always)]
    fn sum<
        I: PixelIndex,
        P: Pixel,
        const FAST: bool,
        const SIGNAL: bool,
        const NORM: bool,
        const VAR: bool,
        const COUNT: bool,
    >(
        &self,
        i: usize,
        cols: &[I],
        mask: Option<&[u8]>,
        signal: &[P],
        norm: &[f64],
    ) -> [f64; 4] {
        let start = self.row_ptr[i];
        let end = self.row_ptr[i + 1];
        match (self.values.as_deref(), mask) {
            (Some(w), Some(m)) => row::<I, P, FAST, SIGNAL, NORM, VAR, COUNT, true, true>(
                start, end, cols, w, m, signal, norm,
            ),
            (Some(w), None) => row::<I, P, FAST, SIGNAL, NORM, VAR, COUNT, true, false>(
                start,
                end,
                cols,
                w,
                &[],
                signal,
                norm,
            ),
            (None, Some(m)) => row::<I, P, FAST, SIGNAL, NORM, VAR, COUNT, false, true>(
                start,
                end,
                cols,
                &[],
                m,
                signal,
                norm,
            ),
            (None, None) => row::<I, P, FAST, SIGNAL, NORM, VAR, COUNT, false, false>(
                start,
                end,
                cols,
                &[],
                &[],
                signal,
                norm,
            ),
        }
    }

    fn cached_results<const FAST: bool, const VAR: bool, const COUNT: bool, P: Pixel>(
        &self,
        signal: &[P],
        mask: Option<&[u8]>,
        norm: &[f64],
    ) -> Vec<[f64; 4]> {
        let cached = self.normalization::<FAST>(mask, norm);
        macro_rules! run {
            ($cols:expr) => {
                (0..self.row_ptr.len() - 1)
                    .into_par_iter()
                    .map(|i| {
                        let mut sums = self.sum::<_, P, FAST, true, false, VAR, COUNT>(
                            i,
                            $cols,
                            mask,
                            signal,
                            &[],
                        );
                        sums[1] = cached[i];
                        sums
                    })
                    .collect()
            };
        }
        match &self.col_idx {
            Columns::Compact(c) => run!(c),
            Columns::Wide(c) => run!(c),
        }
    }

    fn results<const FAST: bool, const VAR: bool, const COUNT: bool, P: Pixel>(
        &self,
        signal: &[P],
        mask: Option<&[u8]>,
        norm: &[f64],
    ) -> Vec<[f64; 4]> {
        if self.mode == MaskMode::Fixed {
            if FAST
                && self.fixed_csc_enabled
                && self.prefer_csc
                && let Some(csc) = self.dynamic_csc(norm.len())
            {
                let pixels = self.fixed_pixels.get_or_init(|| csc.active_pixels(mask));
                let cached_norm = self.normalization::<true>(mask, norm);
                return csc.fixed_results::<VAR, COUNT, P>(signal, pixels, cached_norm);
            }
            return self
                .fixed_table(mask)
                .cached_results::<FAST, VAR, COUNT, P>(signal, None, norm);
        }
        if FAST
            && self.prefer_csc
            && let Some(csc) = self.dynamic_csc(norm.len())
        {
            return csc.results::<VAR, COUNT, P>(signal, mask, norm);
        }
        // Strict reduction order (or a table too large for compact CSC).
        macro_rules! run {
            ($cols:expr) => {
                (0..self.row_ptr.len() - 1)
                    .into_par_iter()
                    .map(|i| {
                        self.sum::<_, P, FAST, true, true, VAR, COUNT>(i, $cols, mask, signal, norm)
                    })
                    .collect()
            };
        }
        match &self.col_idx {
            Columns::Compact(c) => run!(c),
            Columns::Wide(c) => run!(c),
        }
    }

    fn normalized<const FAST: bool, const VAR: bool, P: Pixel>(
        &self,
        s: &[P],
        m: Option<&[u8]>,
        n: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let rows = self.results::<FAST, VAR, false, P>(s, m, n);
        let mut intensity = Vec::with_capacity(rows.len());
        let mut errors = Vec::with_capacity(if VAR { rows.len() } else { 0 });
        for [signal, norm, variance, _] in rows {
            // Keep the original division and square-root order, including zero
            // normalization and nonfinite inputs; do not precompute reciprocals.
            intensity.push(if norm == 0.0 { 0.0 } else { signal / norm });
            if VAR {
                let v = if norm == 0.0 {
                    0.0
                } else {
                    variance / (norm * norm)
                };
                errors.push(v.max(0.0).sqrt());
            }
        }
        (intensity, errors)
    }

    fn raw<const FAST: bool, const VAR: bool>(
        &self,
        s: &[f64],
        m: Option<&[u8]>,
        n: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>, Vec<f64>) {
        let rows = self.results::<FAST, VAR, true, f64>(s, m, n);
        let mut signal = Vec::with_capacity(rows.len());
        let mut norm = Vec::with_capacity(rows.len());
        let mut variance = VAR.then(|| Vec::with_capacity(rows.len()));
        let mut count = Vec::with_capacity(rows.len());
        for [s, n, v, c] in rows {
            signal.push(s);
            norm.push(n);
            count.push(c);
            if let Some(ref mut values) = variance {
                values.push(v);
            }
        }
        (signal, norm, variance, count)
    }
}

impl Prepared1d {
    pub(crate) fn fixed_csc(&self) -> Option<&Csc> {
        self.csr.csc.get()?.as_ref()
    }
    pub(crate) fn fixed_pixels(&self) -> &[u32] {
        self.csr.fixed_pixels.get().expect("prepared fixed pixels")
    }
    pub(crate) fn fixed_bin_norm(&self) -> &[f64] {
        self.csr.fast_norm.get().expect("prepared normalization")
    }

    pub(crate) fn set_sparse_layout(&mut self, layout: SparseLayout) {
        self.csr.prefer_csc = layout == SparseLayout::Csc;
    }
    pub(crate) fn set_mask_mode(&mut self, mode: MaskMode) {
        self.csr.set_mask_mode(mode);
    }
    pub(crate) fn invalidate_mask(&mut self) {
        self.csr.invalidate_mask();
    }

    pub(crate) fn new(l: Lut1d) -> Self {
        let mut csr = PreparedCsr::new(l.row_ptr, l.col_idx, l.values);
        // Fast 1D uses one geometry-only CSC for both mask policies. 2D and
        // strict arithmetic retain their existing CSR implementations.
        csr.fixed_csc_enabled = true;
        Self {
            csr,
            hist_min: l.hist_min,
            hist_step: l.hist_step,
        }
    }
    pub(crate) fn bins(&self) -> Vec<f64> {
        crate::math::histogram::bins(self.row_ptr.len() - 1, self.hist_min, self.hist_step)
    }
    pub(crate) fn invalidate_normalization(&mut self) {
        self.csr.clear_normalization();
    }
    pub(crate) fn integrate<const FAST: bool, const VAR: bool>(
        &self,
        s: &[f64],
        m: Option<&[u8]>,
        n: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        self.integrate_pixels::<FAST, VAR, f64>(s, m, n)
    }
    pub(crate) fn integrate_pixels<const FAST: bool, const VAR: bool, P: Pixel>(
        &self,
        s: &[P],
        m: Option<&[u8]>,
        n: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (y, e) = self.normalized::<FAST, VAR, P>(s, m, n);
        (self.bins(), y, e)
    }
    pub(crate) fn raw1d<const FAST: bool, const VAR: bool>(
        &self,
        s: &[f64],
        m: Option<&[u8]>,
        n: &[f64],
    ) -> IntermediateResult1d {
        let (sum_signal, sum_norm, sum_variance, count) = self.raw::<FAST, VAR>(s, m, n);
        IntermediateResult1d {
            bins: self.bins(),
            sum_signal,
            sum_norm,
            sum_variance,
            count,
        }
    }
}
impl Prepared2d {
    pub(crate) fn set_mask_mode(&mut self, mode: MaskMode) {
        self.csr.set_mask_mode(mode);
    }
    pub(crate) fn invalidate_mask(&mut self) {
        self.csr.invalidate_mask();
    }

    pub(crate) fn new(l: Lut2d) -> Self {
        Self {
            csr: PreparedCsr::new(l.row_ptr, l.col_idx, l.values),
            n_tth: l.n_tth,
            n_azimuthal: l.n_azimuthal,
        }
    }
    pub(crate) fn invalidate_normalization(&mut self) {
        self.csr.clear_normalization();
    }
    pub(crate) fn integrate<const FAST: bool, const VAR: bool>(
        &self,
        s: &[f64],
        m: Option<&[u8]>,
        n: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        self.integrate_pixels::<FAST, VAR, f64>(s, m, n)
    }
    pub(crate) fn integrate_pixels<const FAST: bool, const VAR: bool, P: Pixel>(
        &self,
        s: &[P],
        m: Option<&[u8]>,
        n: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        self.normalized::<FAST, VAR, P>(s, m, n)
    }
    pub(crate) fn raw2d<const FAST: bool, const VAR: bool>(
        &self,
        s: &[f64],
        m: Option<&[u8]>,
        n: &[f64],
    ) -> IntermediateResult2d {
        let (sum_signal, sum_norm, sum_variance, count) = self.raw::<FAST, VAR>(s, m, n);
        IntermediateResult2d {
            sum_signal,
            sum_norm,
            sum_variance,
            count,
            n_radial: self.n_tth,
            n_azimuthal: self.n_azimuthal,
        }
    }
}

#[inline(always)]
pub(super) fn madd(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        a * b + c
    }
}

#[inline(always)]
fn row<
    I: PixelIndex,
    P: Pixel,
    const FAST: bool,
    const SIGNAL: bool,
    const NORM: bool,
    const VAR: bool,
    const COUNT: bool,
    const WEIGHT: bool,
    const MASK: bool,
>(
    start: usize,
    end: usize,
    cols: &[I],
    weights: &[f64],
    mask: &[u8],
    signal: &[P],
    norm: &[f64],
) -> [f64; 4] {
    let mut s = [0.0; 4];
    let mut n = [0.0; 4];
    let mut v = [0.0; 4];
    let mut c = [0.0; 4];
    let mut k = start;
    // SAFETY: these tables are private and constructed only from valid LUTs.
    // Integrator validates image/mask dimensions; const branches avoid reads of
    // absent arrays. Compact conversion checks every index before narrowing.
    unsafe {
        macro_rules! visit {
            ($lane:expr, $at:expr) => {{
                let lane = $lane;
                let j = cols.get_unchecked($at).pixel();
                if !MASK || *mask.get_unchecked(j) == 0 {
                    let w = if WEIGHT {
                        *weights.get_unchecked($at)
                    } else {
                        1.0
                    };
                    if SIGNAL {
                        let x = signal.get_unchecked(j).to_f64();
                        if WEIGHT {
                            s[lane] = if FAST {
                                madd(w, x, s[lane])
                            } else {
                                s[lane] + w * x
                            };
                            if VAR {
                                let variance = x.max(0.0);
                                v[lane] = if FAST {
                                    madd(w * w, variance, v[lane])
                                } else {
                                    v[lane] + w * w * variance
                                };
                            }
                        } else {
                            s[lane] += x;
                            if VAR {
                                v[lane] += x.max(0.0);
                            }
                        }
                    }
                    if NORM {
                        let x = *norm.get_unchecked(j);
                        n[lane] = if WEIGHT {
                            if FAST {
                                madd(w, x, n[lane])
                            } else {
                                n[lane] + w * x
                            }
                        } else {
                            n[lane] + x
                        };
                    }
                    if COUNT {
                        c[lane] += w;
                    }
                }
            }};
        }
        if FAST {
            while k + 4 <= end {
                for lane in 0..4 {
                    visit!(lane, k + lane);
                }
                k += 4;
            }
            for lane in 0..end - k {
                visit!(lane, k + lane);
            }
        } else {
            while k < end {
                visit!(0, k);
                k += 1;
            }
        }
    }
    let reduce = |x: [f64; 4]| {
        if FAST {
            (x[0] + x[1]) + (x[2] + x[3])
        } else {
            x[0]
        }
    };
    [reduce(s), reduce(n), reduce(v), reduce(c)]
}

#[cfg(test)]
#[path = "prepared_tests.rs"]
mod tests;
