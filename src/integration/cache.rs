use crate::lut;

use super::Integrator;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OneDKey {
    Counting { n_points: usize },
    PixelSplit { n_points: usize },
    Supersampled { n_points: usize, n_ss: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TwoDKey {
    Counting {
        n_tth: usize,
        n_chi: usize,
    },
    PixelSplit {
        n_tth: usize,
        n_chi: usize,
    },
    Supersampled {
        n_tth: usize,
        n_chi: usize,
        n_ss: usize,
    },
    CountingChi {
        n_tth: usize,
        n_chi: usize,
        chi: ChiRangeKey,
    },
    PixelSplitChi {
        n_tth: usize,
        n_chi: usize,
        chi: ChiRangeKey,
    },
    SupersampledChi {
        n_tth: usize,
        n_chi: usize,
        n_ss: usize,
        chi: ChiRangeKey,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ChiRangeKey {
    pub(super) start_bits: u64,
    pub(super) step_bits: u64,
}

impl ChiRangeKey {
    pub(super) fn new(start: f64, step: f64) -> Self {
        Self { start_bits: start.to_bits(), step_bits: step.to_bits() }
    }
}

pub(super) struct LutCache<K, L> {
    pub(super) key: Option<K>,
    pub(super) lut: Option<L>,
}

impl<K, L> LutCache<K, L> {
    pub(super) fn new() -> Self {
        Self {
            key: None,
            lut: None,
        }
    }
}

pub(super) type Lut1dCache = LutCache<OneDKey, lut::Lut1d>;
pub(super) type Lut2dCache = LutCache<TwoDKey, lut::Lut2d>;

impl Integrator {
    /// checks whether current lut was calculated for the same number of points (n_points), if not,
    /// it is recalculated
    pub(super) fn ensure_lut1d(&self, key: OneDKey) -> std::cell::Ref<'_, lut::Lut1d> {
        let rebuild = {
            let cache = self.cache_1d.borrow();
            match &cache.key {
                None => true,
                Some(k) => k != &key,
            }
        };
        if rebuild {
            let lut1d = match key {
                OneDKey::Counting { n_points } => lut::build_lut1d_csr(&self.tth, n_points),
                OneDKey::PixelSplit { n_points } => {
                    lut::build_lut1d_ps_csr(&self.geometry, &self.tth, n_points)
                }
                OneDKey::Supersampled { n_points, n_ss } => {
                    lut::build_lut1d_ss_csr(&self.geometry, &self.tth, n_points, n_ss)
                }
            };
            let mut cache = self.cache_1d.borrow_mut();
            cache.key = Some(key);
            cache.lut = Some(lut1d);
        }
        std::cell::Ref::map(self.cache_1d.borrow(), |cache| cache.lut.as_ref().unwrap())
    }

    /// checks whether current lut_2d was calculated for the same number of points in tth and chi, if not,
    /// it is recalculated
    ///
    /// # Arguments
    /// * `n_tth` - number of points in tth for the 2d integration
    /// * `n_chi` - number of points in chi for the 2d integration
    pub(super) fn ensure_lut2d(&self, key: TwoDKey) -> std::cell::Ref<'_, lut::Lut2d> {
        let rebuild = {
            let cache = self.cache_2d.borrow();
            match &cache.key {
                None => true,
                Some(k) => k != &key,
            }
        };
        if rebuild {
            let lut2d = match key {
                TwoDKey::Counting { n_tth, n_chi } => {
                    lut::build_lut2d_csr(&self.tth, &self.chi, n_tth, n_chi)
                }
                TwoDKey::PixelSplit { n_tth, n_chi } => {
                    lut::build_lut2d_ps_csr(&self.geometry, &self.tth, &self.chi, n_tth, n_chi)
                }
                TwoDKey::Supersampled { n_tth, n_chi, n_ss } => lut::build_lut2d_ss_csr(
                    &self.geometry,
                    &self.tth,
                    &self.chi,
                    n_tth,
                    n_chi,
                    n_ss,
                ),
                TwoDKey::CountingChi { n_tth, n_chi, chi } => {
                    lut::build_lut2d_csr_with_chi_range(
                        &self.tth,
                        &self.chi,
                        n_tth,
                        n_chi,
                        f64::from_bits(chi.start_bits),
                        f64::from_bits(chi.step_bits),
                    )
                }
                TwoDKey::PixelSplitChi { n_tth, n_chi, chi } => {
                    lut::build_lut2d_ps_csr_with_chi_range(
                        &self.geometry,
                        &self.tth,
                        &self.chi,
                        n_tth,
                        n_chi,
                        f64::from_bits(chi.start_bits),
                        f64::from_bits(chi.step_bits),
                    )
                }
                TwoDKey::SupersampledChi { n_tth, n_chi, n_ss, chi } => {
                    lut::build_lut2d_ss_csr_with_chi_range(
                        &self.geometry,
                        &self.tth,
                        n_tth,
                        n_chi,
                        n_ss,
                        f64::from_bits(chi.start_bits),
                        f64::from_bits(chi.step_bits),
                    )
                }
            };
            let mut cache = self.cache_2d.borrow_mut();
            cache.key = Some(key);
            cache.lut = Some(lut2d);
        }
        std::cell::Ref::map(self.cache_2d.borrow(), |cache| cache.lut.as_ref().unwrap())
    }
}
