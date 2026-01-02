use crate::lut;

use super::{Integrator, RadialUnit};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OneDKey {
    Counting { n_points: usize, unit: RadialUnit },
    PixelSplit { n_points: usize, unit: RadialUnit },
    Supersampled { n_points: usize, n_ss: usize, unit: RadialUnit },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TwoDKey {
    Counting {
        n_radial: usize,
        n_azimuthal: usize,
        unit: RadialUnit,
    },
    PixelSplit {
        n_radial: usize,
        n_azimuthal: usize,
        unit: RadialUnit,
    },
    Supersampled {
        n_radial: usize,
        n_azimuthal: usize,
        n_ss: usize,
        unit: RadialUnit,
    },
    CountingAzimuthal {
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal: AzimuthalRangeKey,
        unit: RadialUnit,
    },
    PixelSplitAzimuthal {
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal: AzimuthalRangeKey,
        unit: RadialUnit,
    },
    SupersampledAzimuthal {
        n_radial: usize,
        n_azimuthal: usize,
        n_ss: usize,
        azimuthal: AzimuthalRangeKey,
        unit: RadialUnit,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AzimuthalRangeKey {
    pub(super) start_bits: u64,
    pub(super) step_bits: u64,
}

impl AzimuthalRangeKey {
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
    /// checks whether current lut was calculated for the same number of points and unit, if not,
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
                OneDKey::Counting { n_points, unit: _ } => {
                    lut::build_lut1d_csr(&self.radial, n_points)
                }
                OneDKey::PixelSplit { n_points, unit } => {
                    lut::build_lut1d_ps_csr(
                        &self.geometry,
                        &self.radial,
                        n_points,
                        unit,
                    )
                }
                OneDKey::Supersampled { n_points, n_ss, unit } => {
                    lut::build_lut1d_ss_csr(
                        &self.geometry,
                        &self.radial,
                        n_points,
                        n_ss,
                        unit,
                    )
                }
            };
            let mut cache = self.cache_1d.borrow_mut();
            cache.key = Some(key);
            cache.lut = Some(lut1d);
        }
        std::cell::Ref::map(self.cache_1d.borrow(), |cache| cache.lut.as_ref().unwrap())
    }

    /// checks whether current lut_2d was calculated for the same number of points in radial/azimuthal and unit,
    /// if not, it is recalculated
    ///
    /// # Arguments
    /// * `n_radial` - number of points in radial direction for the 2d integration
    /// * `n_azimuthal` - number of points in azimuthal for the 2d integration
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
                TwoDKey::Counting { n_radial, n_azimuthal, unit: _ } => {
                    lut::build_lut2d_csr(&self.radial, &self.azimuth, n_radial, n_azimuthal)
                }
                TwoDKey::PixelSplit { n_radial, n_azimuthal, unit } => {
                    lut::build_lut2d_ps_csr(
                        &self.geometry,
                        &self.radial,
                        &self.azimuth,
                        n_radial,
                        n_azimuthal,
                        unit,
                    )
                }
                TwoDKey::Supersampled { n_radial, n_azimuthal, n_ss, unit } => {
                    lut::build_lut2d_ss_csr(
                        &self.geometry,
                        &self.radial,
                        &self.azimuth,
                        n_radial,
                        n_azimuthal,
                        n_ss,
                        unit,
                    )
                }
                TwoDKey::CountingAzimuthal { n_radial, n_azimuthal, azimuthal, unit: _ } => {
                    lut::build_lut2d_csr_with_azimuthal_range(
                        &self.radial,
                        &self.azimuth,
                        n_radial,
                        n_azimuthal,
                        f64::from_bits(azimuthal.start_bits),
                        f64::from_bits(azimuthal.step_bits),
                    )
                }
                TwoDKey::PixelSplitAzimuthal { n_radial, n_azimuthal, azimuthal, unit } => {
                    lut::build_lut2d_ps_csr_with_azimuthal_range(
                        &self.geometry,
                        &self.radial,
                        &self.azimuth,
                        n_radial,
                        n_azimuthal,
                        f64::from_bits(azimuthal.start_bits),
                        f64::from_bits(azimuthal.step_bits),
                        unit,
                    )
                }
                TwoDKey::SupersampledAzimuthal { n_radial, n_azimuthal, n_ss, azimuthal, unit } => {
                    lut::build_lut2d_ss_csr_with_azimuthal_range(
                        &self.geometry,
                        &self.radial,
                        n_radial,
                        n_azimuthal,
                        n_ss,
                        f64::from_bits(azimuthal.start_bits),
                        f64::from_bits(azimuthal.step_bits),
                        unit,
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
