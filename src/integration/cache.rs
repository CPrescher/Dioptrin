use std::sync::Arc;

use crate::lut;
use crate::lut::prepared::{Prepared1d, Prepared2d};

use super::{GeometryState, Integrator, RadialUnit};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RadialRangeKey {
    pub(super) min_bits: u64,
    pub(super) max_bits: u64,
}

impl RadialRangeKey {
    pub(super) fn new(min: f64, max: f64) -> Self {
        Self {
            min_bits: min.to_bits(),
            max_bits: max.to_bits(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OneDKey {
    Counting {
        n_points: usize,
        unit: RadialUnit,
        radial_range: Option<RadialRangeKey>,
    },
    PixelSplit {
        pyfai: bool,
        n_points: usize,
        unit: RadialUnit,
        radial_range: Option<RadialRangeKey>,
    },
    Supersampled {
        n_points: usize,
        n_ss: usize,
        unit: RadialUnit,
        radial_range: Option<RadialRangeKey>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TwoDKey {
    PyfaiFull {
        n_radial: usize,
        n_azimuthal: usize,
        unit: RadialUnit,
        radial_range: Option<RadialRangeKey>,
        azimuthal: Option<AzimuthalRangeKey>,
    },
    SharedGrid {
        grid: super::IntegrationGrid2d,
        unit: RadialUnit,
        sampling: usize,
    },
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

/// One memoized lookup table, tagged with everything it was built from: the
/// image shape and the binning key. A request for another shape or key
/// replaces it; callers that still hold the old table keep it alive through
/// their `Arc`.
pub(super) struct LutCache<K, L> {
    /// `(dim1, dim2)` of the image the table was built for.
    pub(super) shape: (usize, usize),
    pub(super) key: Option<K>,
    pub(super) lut: Option<Arc<L>>,
}

impl<K: PartialEq, L> LutCache<K, L> {
    pub(super) fn new() -> Self {
        Self {
            shape: (0, 0),
            key: None,
            lut: None,
        }
    }

    pub(super) fn clear(&mut self) {
        self.key = None;
        self.lut = None;
    }

    fn get(&self, shape: (usize, usize), key: &K) -> Option<Arc<L>> {
        if self.shape == shape && self.key.as_ref() == Some(key) {
            self.lut.clone()
        } else {
            None
        }
    }

    /// Stores a freshly built table, unless another thread stored one for the
    /// same shape and key first: then that one is returned so all callers share
    /// a single table.
    fn insert(&mut self, shape: (usize, usize), key: K, lut: L) -> Arc<L> {
        if let Some(existing) = self.get(shape, &key) {
            return existing;
        }
        let lut = Arc::new(lut);
        self.shape = shape;
        self.key = Some(key);
        self.lut = Some(Arc::clone(&lut));
        lut
    }

    /// Changes the cached table in place. Settings are changed through
    /// `&mut Integrator`, so no integration call can hold the table; should a
    /// prepared plan or similar still share it, it is dropped and rebuilt on
    /// demand instead.
    pub(super) fn modify(&mut self, change: impl FnOnce(&mut L)) {
        match self.lut.as_mut().map(Arc::get_mut) {
            Some(Some(lut)) => change(lut),
            Some(None) => self.clear(),
            None => {}
        }
    }
}

pub(super) type Lut1dCache = LutCache<OneDKey, Prepared1d>;
pub(super) type Lut2dCache = LutCache<TwoDKey, Prepared2d>;

/// Shared handle to a cached lookup table for one image shape. It owns a
/// reference to the table rather than a lock on the cache, so integrating with
/// it never blocks another thread that needs a table for a different shape.
pub(super) struct Lut1dGuard(Arc<Prepared1d>);

impl std::ops::Deref for Lut1dGuard {
    type Target = Prepared1d;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub(super) struct Lut2dGuard(Arc<Prepared2d>);

impl std::ops::Deref for Lut2dGuard {
    type Target = Prepared2d;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Integrator {
    /// Mask/correction changes preserve sampling tables but invalidate both
    /// arithmetic policies' frame-independent denominators.
    pub(super) fn invalidate_normalization(&self) {
        self.cache_1d
            .write()
            .expect("1D LUT cache lock poisoned")
            .modify(|lut| lut.invalidate_normalization());
        self.cache_2d
            .write()
            .expect("2D LUT cache lock poisoned")
            .modify(|lut| lut.invalidate_normalization());
    }

    pub(super) fn invalidate_mask(&self) {
        self.cache_1d
            .write()
            .expect("1D LUT cache lock poisoned")
            .modify(|lut| lut.invalidate_mask());
        self.cache_2d
            .write()
            .expect("2D LUT cache lock poisoned")
            .modify(|lut| lut.invalidate_mask());
    }

    /// Initialize a common batch geometry/LUT before entering parallel frame work.
    /// Workers that find the cache cold would each build the same geometry and
    /// table; completing it first lets all of them share one.
    pub(crate) fn prepare1d(&self, dim1: usize, dim2: usize, n_points: usize) {
        if let Some(mask) = &self.mask {
            mask.assert_compatible(dim1, dim2);
        }
        let unit = self.radial_unit;
        let radial_range = self.radial_range_key();
        let key = match self.method {
            super::IntegrationMethod::Counting => OneDKey::Counting {
                n_points,
                unit,
                radial_range,
            },
            super::IntegrationMethod::PixelSplit | super::IntegrationMethod::PyfaiFull => {
                OneDKey::PixelSplit {
                    pyfai: matches!(self.method, super::IntegrationMethod::PyfaiFull),
                    n_points,
                    unit,
                    radial_range,
                }
            }
            super::IntegrationMethod::Supersampled { n_ss } => OneDKey::Supersampled {
                n_points,
                n_ss,
                unit,
                radial_range,
            },
            super::IntegrationMethod::Histogram => return,
        };
        drop(self.for_shape(dim1, dim2).ensure_lut1d(key));
    }

    pub(super) fn pixel_split2d_key(
        &self,
        n_radial: usize,
        n_azimuthal: usize,
        azimuthal: Option<AzimuthalRangeKey>,
    ) -> TwoDKey {
        if matches!(self.method, super::IntegrationMethod::PyfaiFull) {
            TwoDKey::PyfaiFull {
                n_radial,
                n_azimuthal,
                azimuthal,
                unit: self.radial_unit,
                radial_range: self.radial_range_key(),
            }
        } else if let Some(azimuthal) = azimuthal {
            TwoDKey::PixelSplitAzimuthal {
                n_radial,
                n_azimuthal,
                azimuthal,
                unit: self.radial_unit,
            }
        } else {
            TwoDKey::PixelSplit {
                n_radial,
                n_azimuthal,
                unit: self.radial_unit,
            }
        }
    }
}

/// An [`Integrator`] together with the geometry snapshot of one image shape.
///
/// Every integration call takes one view for the shape of its image and reads
/// normalization and lookup tables through it, so all of them belong to that
/// shape even while other threads integrate images of other shapes with the
/// same integrator.
pub(super) struct ShapeView<'a> {
    integrator: &'a Integrator,
    pub(super) state: Arc<GeometryState>,
}

impl Integrator {
    pub(super) fn for_shape(&self, dim1: usize, dim2: usize) -> ShapeView<'_> {
        ShapeView {
            integrator: self,
            state: self.geometry_for(dim1, dim2),
        }
    }
}

impl ShapeView<'_> {
    /// Normalization array (pol * sa) for weighted-average integration. An
    /// `Arc` clone: callers share the buffer without copying it.
    pub(super) fn normalization(&self) -> Arc<Vec<f64>> {
        Arc::clone(&self.state.normalization)
    }

    /// Returns the 1D LUT for this view's image shape and the given key,
    /// building it if the cache holds a table for another shape or key.
    ///
    /// Thread-safe for any mix of shapes: the table is built from the geometry
    /// snapshot of exactly this shape and handed out as a shared reference, and
    /// no lock is held while building or integrating. Threads that race on a
    /// missing table may each build it; the first one stored is shared.
    pub(super) fn ensure_lut1d(&self, key: OneDKey) -> Lut1dGuard {
        let ShapeView { integrator, state } = self;
        let shape = (state.dim1, state.dim2);
        let cached = integrator
            .cache_1d
            .read()
            .expect("1D LUT cache lock poisoned")
            .get(shape, &key);
        let lut = if let Some(lut) = cached {
            lut
        } else {
            let radial = &state.radial;

            // Build the LUT (this can take time, but we're not holding the cache lock)
            let lut1d = match key {
                OneDKey::Counting {
                    n_points,
                    unit: _,
                    radial_range,
                } => match radial_range {
                    Some(rr) => lut::build_lut1d_csr_with_range(
                        radial,
                        n_points,
                        f64::from_bits(rr.min_bits),
                        f64::from_bits(rr.max_bits),
                    ),
                    None => lut::build_lut1d_csr(radial, n_points),
                },
                OneDKey::PixelSplit {
                    pyfai,
                    n_points,
                    unit,
                    radial_range,
                } => {
                    let mut geom = integrator.geometry.clone();
                    geom.dim1 = state.dim1;
                    geom.dim2 = state.dim2;
                    if pyfai {
                        lut::build_lut1d_pyfai_csr(
                            &geom,
                            unit,
                            n_points,
                            None,
                            radial_range.map(|rr| {
                                [f64::from_bits(rr.min_bits), f64::from_bits(rr.max_bits)]
                            }),
                        )
                        .expect("valid pyFAI full-splitting grid")
                    } else {
                        match radial_range {
                            Some(rr) => lut::build_lut1d_ps_csr_with_range(
                                &geom,
                                radial,
                                n_points,
                                unit,
                                f64::from_bits(rr.min_bits),
                                f64::from_bits(rr.max_bits),
                            ),
                            None => lut::build_lut1d_ps_csr(&geom, radial, n_points, unit),
                        }
                    }
                }
                OneDKey::Supersampled {
                    n_points,
                    n_ss,
                    unit,
                    radial_range,
                } => {
                    let mut geom = integrator.geometry.clone();
                    geom.dim1 = state.dim1;
                    geom.dim2 = state.dim2;
                    match radial_range {
                        Some(rr) => lut::build_lut1d_ss_csr_with_range(
                            &geom,
                            radial,
                            n_points,
                            n_ss,
                            unit,
                            f64::from_bits(rr.min_bits),
                            f64::from_bits(rr.max_bits),
                        ),
                        None => lut::build_lut1d_ss_csr(&geom, radial, n_points, n_ss, unit),
                    }
                }
            };

            let mut prepared = Prepared1d::new(lut1d);
            prepared.set_mask_mode(integrator.mask_mode);
            prepared.set_sparse_layout(integrator.sparse_layout);
            integrator
                .cache_1d
                .write()
                .expect("1D LUT cache lock poisoned")
                .insert(shape, key, prepared)
        };
        lut.prepare_normalization(
            integrator.mask.as_ref().map(|m| m.data.as_slice()),
            &state.normalization,
            integrator.fast_math,
        );
        Lut1dGuard(lut)
    }

    /// Returns the 2D LUT for this view's image shape and the given key.
    /// Same guarantees as [`Self::ensure_lut1d`].
    pub(super) fn ensure_lut2d(&self, key: TwoDKey) -> Lut2dGuard {
        let ShapeView { integrator, state } = self;
        let shape = (state.dim1, state.dim2);
        let cached = integrator
            .cache_2d
            .read()
            .expect("2D LUT cache lock poisoned")
            .get(shape, &key);
        let lut = if let Some(lut) = cached {
            lut
        } else {
            let radial = &state.radial;
            let azimuth = &state.azimuth;
            let mut geom = integrator.geometry.clone();
            geom.dim1 = state.dim1;
            geom.dim2 = state.dim2;

            // Build the LUT
            let lut2d = match key {
                TwoDKey::PyfaiFull {
                    n_radial,
                    n_azimuthal,
                    unit,
                    radial_range,
                    azimuthal,
                } => {
                    let [lo, hi] = radial_range
                        .map(|r| [f64::from_bits(r.min_bits), f64::from_bits(r.max_bits)])
                        .unwrap_or_else(|| lut::pyfai_radial_range(&geom, unit));
                    let (start, step) = azimuthal
                        .map(|a| (f64::from_bits(a.start_bits), f64::from_bits(a.step_bits)))
                        .unwrap_or((
                            -std::f64::consts::PI + std::f64::consts::PI / n_azimuthal as f64,
                            std::f64::consts::TAU / n_azimuthal as f64,
                        ));
                    let grid = super::IntegrationGrid2d::new(
                        n_radial,
                        n_azimuthal,
                        (lo, hi),
                        (
                            start - 0.5 * step,
                            start + (n_azimuthal as f64 - 0.5) * step,
                        ),
                    )
                    .expect("valid pyFAI cake grid")
                    .with_pixel_split_mode(super::PixelSplitMode::PyfaiHistogram);
                    lut::shared_grid::build(
                        &geom,
                        radial,
                        azimuth,
                        &grid,
                        super::IntegrationMethod::PixelSplit,
                        unit,
                    )
                }
                TwoDKey::SharedGrid {
                    grid,
                    unit,
                    sampling,
                } => {
                    let method = match sampling {
                        0 => super::IntegrationMethod::Counting,
                        1 => super::IntegrationMethod::PixelSplit,
                        n => super::IntegrationMethod::Supersampled { n_ss: n - 1 },
                    };
                    lut::shared_grid::build(&geom, radial, azimuth, &grid, method, unit)
                }
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

            let mut prepared = Prepared2d::new(lut2d);
            prepared.set_mask_mode(integrator.mask_mode);
            integrator
                .cache_2d
                .write()
                .expect("2D LUT cache lock poisoned")
                .insert(shape, key, prepared)
        };
        lut.prepare_normalization(
            integrator.mask.as_ref().map(|m| m.data.as_slice()),
            &state.normalization,
            integrator.fast_math,
        );
        Lut2dGuard(lut)
    }
}
