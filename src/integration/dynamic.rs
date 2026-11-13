//! Immutable 1D integration plans shared by independent frame workers.
use super::{IntegrationMethod, Integrator, MaskMode, SparseLayout};
use crate::lut::{Lut1d, csc::Sums, prepared::Prepared1d};
use std::sync::Arc;

/// A prepared geometry/weight/correction snapshot. All per-frame masks and
/// accumulators are supplied to or allocated by `integrate`, never stored here.
/// Sharing this plan does not prevent changing the source integrator later.
pub struct Dynamic1dPlan {
    lut: Arc<Prepared1d>,
    normalization: Arc<Vec<f64>>,
    fast_math: bool,
    csc: bool,
    shape: (usize, usize),
    n_points: usize,
}

pub struct Dynamic1dResult {
    pub radial: Vec<f64>,
    pub intensity: Vec<f64>,
    pub errors: Option<Vec<f64>>,
    pub azimuthal_std: Option<Vec<f64>>,
}

impl Integrator {
    /// Prepare one geometry-only table for concurrent, explicitly masked frames.
    /// The source's stored mask is intentionally excluded; pass the complete
    /// effective mask for each frame to the plan's `integrate` method.
    pub fn prepare_dynamic1d(
        &self,
        shape: (usize, usize),
        n_points: usize,
    ) -> Result<Dynamic1dPlan, String> {
        if shape.0 == 0 || shape.1 == 0 || shape.0.checked_mul(shape.1).is_none() || n_points == 0 {
            return Err("shape and n_points must be positive and fit in memory".into());
        }
        let mut integrator = Integrator::new(self.geometry.clone())
            .with_method(self.method)
            .with_radial_unit(self.radial_unit)
            .with_fast_math(self.fast_math)
            .with_mask_mode(MaskMode::Dynamic)
            .with_sparse_layout(self.sparse_layout);
        integrator.polarization_factor = self.polarization_factor;
        integrator.radial_range = self.radial_range;
        if matches!(self.method, IntegrationMethod::Histogram) {
            return Err("dynamic plans require counting, pixel_split or supersampled".into());
        }
        // Complete lazy geometry/CSR/CSC initialization ONCE, before any frame
        // workers start. There are no cache writers on the resulting plan.
        integrator.prepare1d(shape.0, shape.1, n_points);
        let normalization = integrator.get_normalization();
        let lut = integrator
            .cache_1d
            .into_inner()
            .map_err(|_| "poisoned lookup cache")?
            .lut
            .ok_or("missing prepared dynamic table")?;
        Ok(Dynamic1dPlan {
            lut: Arc::new(lut),
            normalization,
            fast_math: self.fast_math,
            csc: self.sparse_layout == SparseLayout::Csc,
            shape,
            n_points,
        })
    }
}

impl Dynamic1dPlan {
    /// Freeze a caller-built grid/sector LUT and normalization as a dynamic
    /// CSC plan. No mask is baked into the shared weights or denominators.
    pub fn new(
        shape: (usize, usize),
        lut: Lut1d,
        normalization: Arc<Vec<f64>>,
    ) -> Result<Self, String> {
        let pixels = shape.0.checked_mul(shape.1).ok_or("image shape overflow")?;
        if pixels == 0
            || normalization.len() != pixels
            || normalization.iter().any(|n| !n.is_finite() || *n < 0.)
            || lut.row_ptr.len() < 2
            || lut.row_ptr[0] != 0
            || lut.row_ptr.windows(2).any(|p| p[0] > p[1])
            || lut.row_ptr.last() != Some(&lut.col_idx.len())
            || lut.col_idx.iter().any(|&p| p >= pixels)
            || lut
                .values
                .as_ref()
                .is_some_and(|v| v.len() != lut.col_idx.len() || v.iter().any(|x| !x.is_finite()))
            || !lut.hist_min.is_finite()
            || !lut.hist_step.is_finite()
            || lut.hist_step <= 0.
        {
            return Err(
                "Dynamic plan requires valid dimensions, LUT weights/indices and normalization."
                    .into(),
            );
        }
        let n_points = lut.n_points();
        let mut lut = Prepared1d::new(lut);
        lut.set_mask_mode(MaskMode::Dynamic);
        lut.set_sparse_layout(SparseLayout::Csc);
        lut.prepare_normalization(None, &normalization, true);
        Ok(Self {
            lut: Arc::new(lut),
            normalization,
            shape,
            n_points,
            fast_math: true,
            csc: true,
        })
    }

    /// Private reusable histograms for a frame worker borrowing this plan.
    pub fn worker(&self, errors: bool) -> Dynamic1dWorker<'_> {
        let sums = if self.fast_math && self.csc {
            self.lut
                .dynamic_csc(self.normalization.len())
                .map(|csc| csc.dynamic_sums(errors))
        } else {
            None
        };
        Dynamic1dWorker {
            plan: self,
            sums,
            errors,
        }
    }

    /// Supplied variances replace the Poisson estimate, including for negative
    /// background-subtracted signal. Correction parameters are deterministic.
    pub fn integrate_with_variances(
        &self,
        data: &[f64],
        variance: &[f64],
        mask: Option<&[u8]>,
        azimuthal_std: bool,
    ) -> Result<Dynamic1dResult, String> {
        self.worker(true)
            .integrate(data, mask, Some(variance), azimuthal_std)
    }
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Call inside a worker's Rayon pool to control intra-frame parallelism.
    /// Image and mask storage must remain immutable until this call returns.
    pub fn integrate(
        &self,
        data: &[f64],
        mask: Option<&[u8]>,
        errors: bool,
        azimuthal_std: bool,
    ) -> Dynamic1dResult {
        assert_eq!(data.len(), self.shape.0 * self.shape.1);
        if let Some(mask) = mask {
            assert_eq!(mask.len(), data.len());
        }
        let normalization = &self.normalization;
        let lut = &self.lut;
        if azimuthal_std {
            let variance = errors.then(|| data.iter().map(|&s| s.max(0.0)).collect::<Vec<_>>());
            let result = crate::lut::azimuthal::integrate_with_columns(
                data,
                &lut.row_ptr,
                |k| lut.col_idx.get(k),
                lut.values.as_deref(),
                lut.bins(),
                mask,
                normalization,
                variance.as_deref(),
            );
            return Dynamic1dResult {
                radial: result.radial,
                intensity: result.intensity,
                errors: result.errors,
                azimuthal_std: Some(result.azimuthal_std),
            };
        }
        let (radial, intensity, sigma) = match (self.fast_math, errors) {
            (true, true) => lut.integrate::<true, true>(data, mask, normalization),
            (true, false) => lut.integrate::<true, false>(data, mask, normalization),
            (false, true) => lut.integrate::<false, true>(data, mask, normalization),
            (false, false) => lut.integrate::<false, false>(data, mask, normalization),
        };
        Dynamic1dResult {
            radial,
            intensity,
            errors: errors.then_some(sigma),
            azimuthal_std: None,
        }
    }
}

/// Per-frame sums are private; the LUT and normalization remain shared.
pub struct Dynamic1dWorker<'a> {
    plan: &'a Dynamic1dPlan,
    sums: Option<Sums>,
    errors: bool,
}
impl Dynamic1dWorker<'_> {
    pub fn integrate(
        &mut self,
        data: &[f64],
        mask: Option<&[u8]>,
        variance: Option<&[f64]>,
        azimuthal_std: bool,
    ) -> Result<Dynamic1dResult, String> {
        if data.len() != self.plan.normalization.len()
            || mask.is_some_and(|m| m.len() != data.len())
            || variance.is_some_and(|v| {
                !self.errors || v.len() != data.len() || v.iter().any(|x| !x.is_finite() || *x < 0.)
            })
        {
            return Err("Dynamic frame requires matching shapes and finite nonnegative variances with errors enabled.".into());
        }
        let plan = self.plan;
        if !azimuthal_std && let Some(sums) = &mut self.sums {
            let csc = plan.lut.dynamic_csc(data.len()).expect("prepared CSC");
            let (intensity, errors) =
                csc.dynamic_frame(data, variance, mask, &plan.normalization, sums);
            return Ok(Dynamic1dResult {
                radial: plan.lut.bins(),
                intensity,
                errors,
                azimuthal_std: None,
            });
        }
        let Some(variance) = variance else {
            return Ok(plan.integrate(data, mask, self.errors, azimuthal_std));
        };
        let lut = &plan.lut;
        if !azimuthal_std {
            let (radial, intensity, errors) =
                lut.integrate_with_variances(data, variance, mask, &plan.normalization);
            return Ok(Dynamic1dResult {
                radial,
                intensity,
                errors: Some(errors),
                azimuthal_std: None,
            });
        }
        let result = crate::lut::azimuthal::integrate_with_columns(
            data,
            &lut.row_ptr,
            |k| lut.col_idx.get(k),
            lut.values.as_deref(),
            lut.bins(),
            mask,
            &plan.normalization,
            Some(variance),
        );
        Ok(Dynamic1dResult {
            radial: result.radial,
            intensity: result.intensity,
            errors: result.errors,
            azimuthal_std: azimuthal_std.then_some(result.azimuthal_std),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{geometry::Geometry, mask::Mask, poni::Poni};
    use rayon::prelude::*;
    use std::sync::Arc;

    #[test]
    fn supplied_variance_fallback_preserves_nonfinite_and_nonpositive_contributions() {
        for values in [None, Some(vec![0.25, 0.75, -0.5, 0.25, 1., 0., 0.3])] {
            let lut = crate::lut::Lut1d {
                row_ptr: vec![0, 2, 4, 7, 7],
                col_idx: vec![0, 1, 2, 3, 0, 2, 4],
                values,
                hist_min: 0.,
                hist_step: 1.,
            };
            let norm = Arc::new(vec![2., 0., 1., 4., 0.]);
            let variance = [4., 9., 16., 25., 36.];
            for fast in [false, true] {
                for layout in [SparseLayout::Csr, SparseLayout::Csc] {
                    let mut plan = Dynamic1dPlan::new((1, 5), lut.clone(), norm.clone()).unwrap();
                    plan.fast_math = fast;
                    plan.csc = layout == SparseLayout::Csc;
                    Arc::get_mut(&mut plan.lut)
                        .unwrap()
                        .set_sparse_layout(layout);
                    let mut worker = plan.worker(true);
                    for special in [3., f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                        let data = [special, 7., special, -2., 11.];
                        for mask in [None, Some([1, 0, 1, 0, 0]), Some([1; 5])] {
                            let mask = mask.as_ref().map(|m| m.as_slice());
                            let expected = crate::lut::integrate1d_with_errors(
                                &data, &variance, &lut, mask, &norm,
                            );
                            let actual = worker
                                .integrate(&data, mask, Some(&variance), false)
                                .unwrap();
                            let same = |a: &[f64], b: &[f64]| {
                                for (&a, &b) in a.iter().zip(b) {
                                    assert!(
                                        (a.is_nan() && b.is_nan())
                                            || a == b
                                            || (a.is_finite()
                                                && b.is_finite()
                                                && (a - b).abs() <= 1e-12 * (1. + b.abs())),
                                        "{a} != {b}; fast={fast}, layout={layout:?}, signal={special}"
                                    );
                                }
                            };
                            same(&actual.intensity, &expected.1);
                            same(actual.errors.as_ref().unwrap(), &expected.2);
                            assert!(actual.azimuthal_std.is_none());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reusable_workers_propagate_supplied_variances_and_changing_masks() {
        let source = source(IntegrationMethod::PixelSplit, true);
        let g = source.geometry();
        let radial = source
            .radial_unit
            .convert_from_tth(&g.calc_tth_all(), g.poni.wavelength);
        let lut = crate::lut::build_lut1d_ps_csr(g, &radial, 40, source.radial_unit);
        let plan = Arc::new(source.prepare_dynamic1d((32, 64), 40).unwrap());
        let normalization = plan.normalization.clone();
        let table = plan.lut.row_ptr.as_ptr() as usize;
        (0..8usize).into_par_iter().for_each(|frame| {
            let data: Vec<_> = (0..2048)
                .map(|p| (p % 97) as f64 - 80. - frame as f64)
                .collect();
            let variance: Vec<_> = (0..2048).map(|p| 4. + ((p + frame) % 31) as f64).collect();
            let mut worker = plan.worker(true);
            let hist = worker.sums.as_ref().unwrap().signal.as_ptr();
            for step in 0..5 {
                let mask: Vec<_> = (0..2048)
                    .map(|p| u8::from(step == 4 || (p + step + frame) % 11 == 0))
                    .collect();
                let actual = worker
                    .integrate(&data, Some(&mask), Some(&variance), false)
                    .unwrap();
                let expected = crate::lut::integrate1d_with_errors(
                    &data,
                    &variance,
                    &lut,
                    Some(&mask),
                    &normalization,
                );
                close(&actual.radial, &expected.0);
                close(&actual.intensity, &expected.1);
                close(actual.errors.as_ref().unwrap(), &expected.2);
                assert_eq!(hist, worker.sums.as_ref().unwrap().signal.as_ptr());
                assert_eq!(table, plan.lut.row_ptr.as_ptr() as usize);
                assert!(Arc::ptr_eq(&normalization, &plan.normalization));
                let spread = worker
                    .integrate(&data, Some(&mask), Some(&variance), true)
                    .unwrap();
                close(&spread.intensity, &expected.1);
                close(spread.errors.as_ref().unwrap(), &expected.2);
            }
        });
    }

    #[test]
    fn custom_dynamic_grid_validates_shapes_and_recovers_after_failed_frames() {
        let lut = crate::lut::Lut1d {
            row_ptr: vec![0, 2, 4],
            col_idx: vec![0, 1, 0, 1],
            values: Some(vec![0.25, 0.5, 0.75, 0.5]),
            hist_min: 0.,
            hist_step: 1.,
        };
        let plan = Dynamic1dPlan::new((1, 2), lut.clone(), Arc::new(vec![2., 4.])).unwrap();
        let mut worker = plan.worker(true);
        assert!(
            worker
                .integrate(&[-4., 2.], None, Some(&[-1., 25.]), false)
                .is_err()
        );
        assert!(
            worker
                .integrate(&[-4.], None, Some(&[9., 25.]), false)
                .is_err()
        );
        for mask in [[0, 1], [1, 0], [0, 0], [1, 1]] {
            let result = worker
                .integrate(&[-4., 2.], Some(&mask), Some(&[9., 25.]), false)
                .unwrap();
            let expected = crate::lut::integrate1d_with_errors(
                &[-4., 2.],
                &[9., 25.],
                &lut,
                Some(&mask),
                &[2., 4.],
            );
            close(&result.intensity, &expected.1);
            close(result.errors.as_ref().unwrap(), &expected.2);
        }
        assert!(Dynamic1dPlan::new((1, 3), lut, Arc::new(vec![2., 4.])).is_err());
    }

    fn source(method: IntegrationMethod, fast: bool) -> Integrator {
        let mut poni = Poni::new();
        poni.pixel1 = 1e-4;
        poni.pixel2 = 1e-4;
        poni.distance = 0.1;
        poni.poni1 = 0.0016;
        poni.poni2 = 0.0032;
        poni.wavelength = 1e-10;
        Integrator::new(Geometry::new(32, 64, poni))
            .with_method(method)
            .with_fast_math(fast)
            .with_mask_mode(MaskMode::Dynamic)
            .with_polarization_factor(0.965)
    }
    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (&a, &b) in a.iter().zip(b) {
            assert!(
                (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-10 * (1. + a.abs() + b.abs()),
                "{a} != {b}"
            );
        }
    }
    #[test]
    fn concurrent_masks_share_prepared_tables_and_match_independent_integrators() {
        for method in [
            IntegrationMethod::Counting,
            IntegrationMethod::PixelSplit,
            IntegrationMethod::Supersampled { n_ss: 2 },
        ] {
            for fast in [false, true] {
                let source = source(method, fast);
                let plan = Arc::new(source.prepare_dynamic1d((32, 64), 40).unwrap());
                let table = plan.lut.row_ptr.as_ptr() as usize;
                let norm = plan.normalization.clone();
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(8)
                    .build()
                    .unwrap();
                pool.install(|| {
                    (0..16usize).into_par_iter().for_each(|frame| {
                        let data: Vec<_> = (0..2048)
                            .map(|p| ((p * 13 + frame * 17) % 997) as f64 - 7.)
                            .collect();
                        let mask: Vec<u8> = (0..2048)
                            .map(|p| u8::from(frame == 15 || (p + frame) % 13 == 0))
                            .collect();
                        let mut independent = super::tests::source(method, fast);
                        independent.set_mask(Mask {
                            data: mask.clone(),
                            dim1: 32,
                            dim2: 64,
                        });
                        for errors in [false, true] {
                            for spread in [false, true] {
                                let actual = plan.integrate(&data, Some(&mask), errors, spread);
                                if spread {
                                    let expected = independent
                                        .integrate1d_with_azimuthal_std_from_slice(
                                            &data, 32, 64, 40, errors,
                                        );
                                    close(&actual.radial, &expected.radial);
                                    close(&actual.intensity, &expected.intensity);
                                    close(
                                        actual.azimuthal_std.as_ref().unwrap(),
                                        &expected.azimuthal_std,
                                    );
                                    if errors {
                                        close(
                                            actual.errors.as_ref().unwrap(),
                                            expected.errors.as_ref().unwrap(),
                                        );
                                    }
                                } else if errors {
                                    let (x, y, e) = independent
                                        .integrate1d_with_errors_from_slice(&data, 32, 64, 40);
                                    close(&actual.radial, &x);
                                    close(&actual.intensity, &y);
                                    close(actual.errors.as_ref().unwrap(), &e);
                                } else {
                                    let (x, y) =
                                        independent.integrate1d_from_slice(&data, 32, 64, 40);
                                    close(&actual.radial, &x);
                                    close(&actual.intensity, &y);
                                }
                            }
                        }
                        assert_eq!(table, plan.lut.row_ptr.as_ptr() as usize);
                        assert!(Arc::ptr_eq(&norm, &plan.normalization.clone()));
                    })
                });
            }
        }
    }
    #[test]
    fn plan_is_an_independent_configuration_snapshot() {
        let mut source = source(IntegrationMethod::PixelSplit, true);
        let plan = source.prepare_dynamic1d((32, 64), 40).unwrap();
        let data = vec![100.; 2048];
        let before = plan.integrate(&data, None, true, false);
        source.set_polarization_factor(Some(0.1));
        source.set_mask(Mask {
            data: vec![1; 2048],
            dim1: 32,
            dim2: 64,
        });
        source.set_method(IntegrationMethod::Counting);
        let after = plan.integrate(&data, None, true, false);
        assert_eq!(before.intensity, after.intensity);
        assert_eq!(before.errors, after.errors);
        let changed = source
            .prepare_dynamic1d((32, 64), 40)
            .unwrap()
            .integrate(&data, None, true, false);
        assert_ne!(before.intensity, changed.intensity);
        assert!(source.prepare_dynamic1d((0, 64), 40).is_err());
        assert!(source.prepare_dynamic1d((32, 64), 0).is_err());
    }
    #[test]
    fn fixed_dynamic_switches_preserve_geometry_weights_and_refresh_mask_results() {
        for fast in [false, true] {
            let mut ai = source(IntegrationMethod::PixelSplit, fast);
            let data: Vec<_> = (0..2048).map(|p| (p % 97) as f64 + 10.).collect();
            ai.prepare1d(32, 64, 40);
            let storage = |ai: &Integrator| {
                let cache = ai.cache_1d.read().unwrap();
                let lut = cache.lut.as_ref().unwrap();
                (
                    lut.row_ptr.as_ptr() as usize,
                    lut.col_idx.as_ptr() as usize,
                    lut.values.as_ref().unwrap().as_ptr() as usize,
                )
            };
            let original = storage(&ai);
            for (step, mode) in [
                MaskMode::Fixed,
                MaskMode::Dynamic,
                MaskMode::Fixed,
                MaskMode::Dynamic,
                MaskMode::Fixed,
            ]
            .into_iter()
            .enumerate()
            {
                let mask: Vec<_> = (0..2048).map(|p| u8::from((p + step) % 11 == 0)).collect();
                ai.set_mask_mode(mode);
                ai.set_mask(Mask {
                    data: mask.clone(),
                    dim1: 32,
                    dim2: 64,
                });
                let actual = ai.integrate1d_with_errors_from_slice(&data, 32, 64, 40);
                let mut fresh = source(IntegrationMethod::PixelSplit, fast);
                fresh.set_mask_mode(mode);
                fresh.set_mask(Mask {
                    data: mask,
                    dim1: 32,
                    dim2: 64,
                });
                let expected = fresh.integrate1d_with_errors_from_slice(&data, 32, 64, 40);
                close(&actual.0, &expected.0);
                close(&actual.1, &expected.1);
                close(&actual.2, &expected.2);
                assert_eq!(storage(&ai), original);
            }
        }
    }
}
