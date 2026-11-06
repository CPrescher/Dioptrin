//! Immutable 1D integration plans shared by independent frame workers.
use super::cache::OneDKey;
use super::{IntegrationMethod, Integrator, MaskMode};

/// A prepared geometry/weight/correction snapshot. All per-frame masks and
/// accumulators are supplied to or allocated by `integrate`, never stored here.
/// Sharing this plan does not prevent changing the source integrator later.
pub struct Dynamic1dPlan {
    integrator: Integrator,
    shape: (usize, usize),
    n_points: usize,
    key: OneDKey,
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
            .with_mask_mode(MaskMode::Dynamic);
        integrator.polarization_factor = self.polarization_factor;
        integrator.radial_range = self.radial_range;
        let unit = self.radial_unit;
        let radial_range = self.radial_range_key();
        let key = match self.method {
            IntegrationMethod::Counting => OneDKey::Counting {
                n_points,
                unit,
                radial_range,
            },
            IntegrationMethod::PixelSplit => OneDKey::PixelSplit {
                n_points,
                unit,
                radial_range,
            },
            IntegrationMethod::Supersampled { n_ss } => OneDKey::Supersampled {
                n_points,
                n_ss,
                unit,
                radial_range,
            },
            IntegrationMethod::Histogram => {
                return Err("dynamic plans require counting, pixel_split or supersampled".into());
            }
        };
        // Complete lazy geometry/CSR/CSC initialization ONCE, before any frame
        // workers start. There are no cache writers on the resulting plan.
        integrator.prepare1d(shape.0, shape.1, n_points);
        Ok(Dynamic1dPlan {
            integrator,
            shape,
            n_points,
            key,
        })
    }
}

impl Dynamic1dPlan {
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
        let integrator = &self.integrator;
        let normalization = integrator.get_normalization();
        let lut = integrator.ensure_lut1d(self.key);
        if azimuthal_std {
            let variance = errors.then(|| data.iter().map(|&s| s.max(0.0)).collect::<Vec<_>>());
            let result = crate::lut::azimuthal::integrate_with_columns(
                data,
                &lut.row_ptr,
                |k| lut.col_idx.get(k),
                lut.values.as_deref(),
                lut.bins(),
                mask,
                &normalization,
                variance.as_deref(),
            );
            return Dynamic1dResult {
                radial: result.radial,
                intensity: result.intensity,
                errors: result.errors,
                azimuthal_std: Some(result.azimuthal_std),
            };
        }
        if errors {
            let (radial, intensity, errors) =
                integrator.lut_integrate1d_with_errors(data, &lut, mask, &normalization);
            Dynamic1dResult {
                radial,
                intensity,
                errors: Some(errors),
                azimuthal_std: None,
            }
        } else {
            let (radial, intensity) = integrator.lut_integrate1d(data, &lut, mask, &normalization);
            Dynamic1dResult {
                radial,
                intensity,
                errors: None,
                azimuthal_std: None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{geometry::Geometry, mask::Mask, poni::Poni};
    use rayon::prelude::*;
    use std::sync::Arc;

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
                let table = plan.integrator.ensure_lut1d(plan.key).row_ptr.as_ptr() as usize;
                let norm = plan.integrator.get_normalization();
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
                        assert_eq!(
                            table,
                            plan.integrator.ensure_lut1d(plan.key).row_ptr.as_ptr() as usize
                        );
                        assert!(Arc::ptr_eq(&norm, &plan.integrator.get_normalization()));
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
