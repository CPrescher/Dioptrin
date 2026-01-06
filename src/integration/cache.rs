use std::sync::RwLockReadGuard;

use crate::lut;

use super::{Integrator, RadialUnit};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OneDKey {
    Counting {
        n_points: usize,
        unit: RadialUnit,
    },
    PixelSplit {
        n_points: usize,
        unit: RadialUnit,
    },
    Supersampled {
        n_points: usize,
        n_ss: usize,
        unit: RadialUnit,
    },
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
        Self {
            start_bits: start.to_bits(),
            step_bits: step.to_bits(),
        }
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

/// A mapped read guard that provides access to the LUT inside the cache.
pub(super) struct Lut1dGuard<'a> {
    _guard: RwLockReadGuard<'a, Lut1dCache>,
}

impl<'a> std::ops::Deref for Lut1dGuard<'a> {
    type Target = lut::Lut1d;
    fn deref(&self) -> &Self::Target {
        self._guard
            .lut
            .as_ref()
            .expect("Lut1dGuard created without LUT being built")
    }
}

pub(super) struct Lut2dGuard<'a> {
    _guard: RwLockReadGuard<'a, Lut2dCache>,
}

impl<'a> std::ops::Deref for Lut2dGuard<'a> {
    type Target = lut::Lut2d;
    fn deref(&self) -> &Self::Target {
        self._guard
            .lut
            .as_ref()
            .expect("Lut2dGuard created without LUT being built")
    }
}

impl Integrator {
    /// Ensures the 1D LUT is built for the given key, using double-checked locking.
    /// Thread-safe: uses RwLock for interior mutability.
    pub(super) fn ensure_lut1d(&self, key: OneDKey) -> Lut1dGuard<'_> {
        // First check with read lock
        let needs_rebuild = {
            let cache = self.cache_1d.read().expect("1D LUT cache lock poisoned");
            match &cache.key {
                None => true,
                Some(k) => k != &key,
            }
        };

        if needs_rebuild {
            // Get state for building LUT
            let state = self.state.read().expect("geometry state lock poisoned");
            let radial = &state.radial;

            // Build the LUT (this can take time, but we're not holding the cache lock)
            let lut1d = match key {
                OneDKey::Counting { n_points, unit: _ } => lut::build_lut1d_csr(radial, n_points),
                OneDKey::PixelSplit { n_points, unit } => {
                    let mut geom = self.geometry.clone();
                    geom.dim1 = state.dim1;
                    geom.dim2 = state.dim2;
                    lut::build_lut1d_ps_csr(&geom, radial, n_points, unit)
                }
                OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                } => {
                    let mut geom = self.geometry.clone();
                    geom.dim1 = state.dim1;
                    geom.dim2 = state.dim2;
                    lut::build_lut1d_ss_csr(&geom, radial, n_points, n_ss, unit)
                }
            };
            drop(state);

            // Acquire write lock and double-check (another thread may have built it)
            let mut cache = self.cache_1d.write().expect("1D LUT cache lock poisoned");
            if cache.key.as_ref() != Some(&key) {
                cache.key = Some(key);
                cache.lut = Some(lut1d);
            }
        }

        // Return read guard
        Lut1dGuard {
            _guard: self.cache_1d.read().expect("1D LUT cache lock poisoned"),
        }
    }

    /// Ensures the 2D LUT is built for the given key, using double-checked locking.
    /// Thread-safe: uses RwLock for interior mutability.
    pub(super) fn ensure_lut2d(&self, key: TwoDKey) -> Lut2dGuard<'_> {
        // First check with read lock
        let needs_rebuild = {
            let cache = self.cache_2d.read().expect("2D LUT cache lock poisoned");
            match &cache.key {
                None => true,
                Some(k) => k != &key,
            }
        };

        if needs_rebuild {
            // Get state for building LUT
            let state = self.state.read().expect("geometry state lock poisoned");
            let radial = &state.radial;
            let azimuth = &state.azimuth;
            let mut geom = self.geometry.clone();
            geom.dim1 = state.dim1;
            geom.dim2 = state.dim2;

            // Build the LUT
            let lut2d = match key {
                TwoDKey::Counting {
                    n_radial,
                    n_azimuthal,
                    unit: _,
                } => lut::build_lut2d_csr(radial, azimuth, n_radial, n_azimuthal),
                TwoDKey::PixelSplit {
                    n_radial,
                    n_azimuthal,
                    unit,
                } => lut::build_lut2d_ps_csr(&geom, radial, azimuth, n_radial, n_azimuthal, unit),
                TwoDKey::Supersampled {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    unit,
                } => lut::build_lut2d_ss_csr(
                    &geom,
                    radial,
                    azimuth,
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    unit,
                ),
                TwoDKey::CountingAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit: _,
                } => lut::build_lut2d_csr_with_azimuthal_range(
                    radial,
                    azimuth,
                    n_radial,
                    n_azimuthal,
                    f64::from_bits(azimuthal.start_bits),
                    f64::from_bits(azimuthal.step_bits),
                ),
                TwoDKey::PixelSplitAzimuthal {
                    n_radial,
                    n_azimuthal,
                    azimuthal,
                    unit,
                } => lut::build_lut2d_ps_csr_with_azimuthal_range(
                    &geom,
                    radial,
                    azimuth,
                    n_radial,
                    n_azimuthal,
                    f64::from_bits(azimuthal.start_bits),
                    f64::from_bits(azimuthal.step_bits),
                    unit,
                ),
                TwoDKey::SupersampledAzimuthal {
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    azimuthal,
                    unit,
                } => lut::build_lut2d_ss_csr_with_azimuthal_range(
                    &geom,
                    radial,
                    n_radial,
                    n_azimuthal,
                    n_ss,
                    f64::from_bits(azimuthal.start_bits),
                    f64::from_bits(azimuthal.step_bits),
                    unit,
                ),
            };
            drop(state);

            // Acquire write lock and double-check
            let mut cache = self.cache_2d.write().expect("2D LUT cache lock poisoned");
            if cache.key.as_ref() != Some(&key) {
                cache.key = Some(key);
                cache.lut = Some(lut2d);
            }
        }

        // Return read guard
        Lut2dGuard {
            _guard: self.cache_2d.read().expect("2D LUT cache lock poisoned"),
        }
    }
}
