//! Immutable CSR snapshots retaining the existing reduction order.
use super::{Fixed1dResult, IntegrationGrid2d, Integrator, MaskMode};
use crate::{
    lut::{
        Lut1d,
        prepared::{Prepared1d, Prepared2d},
    },
    mask::Mask,
};
use std::sync::Arc;

/// Fixed-mask CSR execution with cached normalization and strict arithmetic.
/// LUT weights/order are unchanged; masked entries are removed once.
pub struct FixedCsr1dPlan {
    lut: Prepared1d,
    norm: Vec<f64>,
    mask: Option<Vec<u8>>,
}
impl FixedCsr1dPlan {
    /// Validate and own a LUT, per-pixel normalization and an optional fixed mask.
    /// Geometry/normalization changes require a new plan; input samples can vary.
    pub fn new(lut: Lut1d, normalization: Vec<f64>, mask: Option<Vec<u8>>) -> Result<Self, String> {
        let pixels = normalization.len();
        if pixels == 0
            || mask.as_ref().is_some_and(|m| m.len() != pixels)
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
            return Err("CSR plan requires valid LUT indices, finite weights/normalization and a matching mask".into());
        }
        let lut = Prepared1d::new(lut);
        lut.prepare_normalization(mask.as_deref(), &normalization, false);
        Ok(Self {
            lut,
            norm: normalization,
            mask,
        })
    }
    /// Preserve strict CSR signal/variance summation and division order.
    pub fn integrate_frame(&self, data: &[f64], errors: bool) -> Result<Fixed1dResult, String> {
        if data.len() != self.norm.len() {
            return Err("image size mismatch".into());
        }
        let (radial, intensity, sigma) = if errors {
            self.lut
                .integrate::<false, true>(data, self.mask.as_deref(), &self.norm)
        } else {
            self.lut
                .integrate::<false, false>(data, self.mask.as_deref(), &self.norm)
        };
        Ok(Fixed1dResult {
            radial,
            intensity,
            errors: errors.then_some(sigma),
        })
    }
}

/// Frozen shared-grid geometry, weights, fixed mask and normalization.
/// Subsequent edits to the originating integrator cannot change this snapshot.
pub struct FixedGrid2dPlan {
    lut: Prepared2d,
    shape: (usize, usize),
    mask: Option<Mask>,
    norm: Arc<Vec<f64>>,
    fast: bool,
}
impl Integrator {
    /// Freeze a validated shared-grid LUT and its fixed-mask normalization.
    /// Per-frame input dimensions and unmasked finite samples remain checked.
    /// The snapshot retains this integrator's fast/strict arithmetic policy.
    pub fn prepare_fixed2d_on_grid(
        &self,
        shape: (usize, usize),
        grid: &IntegrationGrid2d,
    ) -> Result<FixedGrid2dPlan, String> {
        if self.mask_mode != MaskMode::Fixed {
            return Err("fixed grid plans require a fixed mask".into());
        }
        let pixels = shape.0.checked_mul(shape.1).ok_or("image shape overflow")?;
        let mut snapshot = Integrator::new(self.geometry.clone())
            .with_method(self.method)
            .with_radial_unit(self.radial_unit)
            .with_fast_math(self.fast_math);
        snapshot.polarization_factor = self.polarization_factor;
        snapshot.radial_range = self.radial_range;
        snapshot.mask = self.mask.as_ref().map(|m| Mask {
            data: m.data.clone(),
            dim1: m.dim1,
            dim2: m.dim2,
        });
        snapshot.validate_2d_setup(shape.0, shape.1, pixels)?;
        snapshot.update_geometry_arrays(shape.0, shape.1);
        let norm = snapshot.get_normalization();
        snapshot.validate_grid_coordinates(&norm)?;
        let key = snapshot.grid_lut_key(grid);
        {
            let lut = snapshot.ensure_lut2d(key);
            lut.prepare_normalization(
                snapshot.mask.as_ref().map(|m| m.data.as_slice()),
                &norm,
                self.fast_math,
            );
        }
        let lut = snapshot
            .cache_2d
            .into_inner()
            .map_err(|_| "poisoned lookup cache")?
            .lut
            .ok_or("missing grid LUT")?;
        Ok(FixedGrid2dPlan {
            lut,
            shape,
            mask: snapshot.mask,
            norm,
            fast: self.fast_math,
        })
    }
}
impl FixedGrid2dPlan {
    /// Detector image dimensions, not the output cake dimensions.
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
    pub fn integrate_frame(
        &self,
        data: &[f64],
        errors: bool,
    ) -> Result<(Vec<f64>, Option<Vec<f64>>), String> {
        if data.len() != self.shape.0 * self.shape.1 {
            return Err("image size mismatch".into());
        }
        let mask = self.mask.as_ref().map(|m| m.data.as_slice());
        if data
            .iter()
            .enumerate()
            .any(|(i, x)| !x.is_finite() && mask.is_none_or(|m| m[i] == 0))
        {
            return Err("unmasked image pixels must be finite".into());
        }
        if !errors {
            let intensity = if self.fast {
                self.lut.fixed_intensity::<true>(data, mask, &self.norm)
            } else {
                self.lut.fixed_intensity::<false>(data, mask, &self.norm)
            };
            return Ok((intensity, None));
        }
        let (intensity, sigma) = if self.fast {
            self.lut.integrate::<true, true>(data, mask, &self.norm)
        } else {
            self.lut.integrate::<false, true>(data, mask, &self.norm)
        };
        Ok((intensity, Some(sigma)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        geometry::Geometry,
        integration::{IntegrationMethod, RadialUnit},
        lut,
        poni::Poni,
    };
    fn geometry(centered: bool) -> Geometry {
        let mut p = Poni::new();
        p.pixel1 = 1e-4;
        p.pixel2 = 1.2e-4;
        p.distance = 0.08;
        p.poni1 = if centered { 0.00125 } else { -0.0013 };
        p.poni2 = if centered { 0.00198 } else { -0.0021 };
        p.rot1 = 0.007;
        p.rot2 = -0.011;
        p.rot3 = 0.02;
        p.wavelength = 0.5e-10;
        Geometry::new(24, 32, p)
    }
    fn same(a: &[f64], b: &[f64]) {
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    #[test]
    fn csr_snapshot_preserves_strict_signal_errors_sectors_and_changed_frames() {
        for centered in [false, true] {
            let g = geometry(centered);
            let tth = g.calc_tth_all();
            let az = g.calc_azimuth_all();
            let mut norm = g.calc_solid_angle();
            for (n, p) in norm.iter_mut().zip(g.calc_polarization(0.6, &tth, &az)) {
                *n *= p;
            }
            let mask: Vec<_> = (0..g.size()).map(|i| u8::from(i % 19 == 0)).collect();
            for unit in [
                RadialUnit::TthRad,
                RadialUnit::TthDeg,
                RadialUnit::QAngstrom,
                RadialUnit::QNm,
            ] {
                for sector in [None, Some([-0.6, 0.9]), Some([2.6, 3.8])] {
                    let lut = lut::build_lut1d_pyfai_csr(&g, unit, 67, sector, None).unwrap();
                    let plan =
                        FixedCsr1dPlan::new(lut.clone(), norm.clone(), Some(mask.clone())).unwrap();
                    for frame in [1., 3.7] {
                        let data: Vec<_> = (0..g.size())
                            .map(|i| ((i * 997) % 103) as f64 * frame - 0.3)
                            .collect();
                        let (x, y) = lut::integrate1d(&data, &lut, Some(&mask), &norm);
                        let r = plan.integrate_frame(&data, false).unwrap();
                        same(&r.radial, &x);
                        same(&r.intensity, &y);
                        let v: Vec<_> = data.iter().map(|x| x.max(0.)).collect();
                        let (_, y, e) =
                            lut::integrate1d_with_errors(&data, &v, &lut, Some(&mask), &norm);
                        let r = plan.integrate_frame(&data, true).unwrap();
                        same(&r.intensity, &y);
                        same(&r.errors.unwrap(), &e);
                    }
                    assert!(plan.integrate_frame(&[1.], false).is_err());
                }
            }
        }
    }
    #[test]
    fn fixed_grid_preserves_raw_division_errors_and_freezes_dependencies() {
        for centered in [false, true] {
            for fast in [false, true] {
                let g = geometry(centered);
                let mut engine = Integrator::new(g.clone())
                    .with_method(IntegrationMethod::PyfaiFull)
                    .with_radial_unit(RadialUnit::TthDeg)
                    .with_polarization_factor(0.6)
                    .with_fast_math(fast);
                let mask: Vec<_> = (0..g.size()).map(|i| u8::from(i % 19 == 0)).collect();
                engine.set_mask(Mask {
                    data: mask.clone(),
                    dim1: g.dim1,
                    dim2: g.dim2,
                });
                for angles in [
                    (-std::f64::consts::PI, std::f64::consts::PI),
                    (-0.6, 0.9),
                    (2.6, 3.8),
                ] {
                    let grid = IntegrationGrid2d::new(67, 29, (0., 12.), angles).unwrap();
                    let plan = engine
                        .prepare_fixed2d_on_grid((g.dim1, g.dim2), &grid)
                        .unwrap();
                    for frame in [1., 3.7] {
                        let data: Vec<_> = (0..g.size())
                            .map(|i| ((i * 997) % 103) as f64 * frame - 0.3)
                            .collect();
                        for errors in [false, true] {
                            let raw = engine
                                .integrate2d_raw_on_grid_from_slice(
                                    &data, g.dim1, g.dim2, &grid, errors,
                                )
                                .unwrap();
                            let expected: Vec<_> = raw
                                .sum_signal
                                .iter()
                                .zip(&raw.sum_norm)
                                .map(|(s, n)| if *n == 0. { 0. } else { s / n })
                                .collect();
                            let (y, e) = plan.integrate_frame(&data, errors).unwrap();
                            same(&y, &expected);
                            if let Some(v) = raw.sum_variance {
                                let expected: Vec<_> = v
                                    .iter()
                                    .zip(&raw.sum_norm)
                                    .map(|(v, n)| {
                                        if *n == 0. {
                                            0.
                                        } else {
                                            (v / (n * n)).max(0.).sqrt()
                                        }
                                    })
                                    .collect();
                                same(&e.unwrap(), &expected);
                            }
                        }
                    }
                    let mut data = vec![2.; g.size()];
                    let expected = plan.integrate_frame(&data, false).unwrap().0;
                    engine.set_polarization_factor(Some(0.2));
                    same(&plan.integrate_frame(&data, false).unwrap().0, &expected);
                    engine.set_polarization_factor(Some(0.6));
                    engine.set_mask(Mask {
                        data: vec![1; g.size()],
                        dim1: g.dim1,
                        dim2: g.dim2,
                    });
                    same(&plan.integrate_frame(&data, false).unwrap().0, &expected);
                    engine.set_mask(Mask {
                        data: mask.clone(),
                        dim1: g.dim1,
                        dim2: g.dim2,
                    });
                    engine.set_radial_unit(RadialUnit::QAngstrom);
                    same(&plan.integrate_frame(&data, false).unwrap().0, &expected);
                    engine.set_radial_unit(RadialUnit::TthDeg);
                    data[0] = f64::NAN;
                    assert!(plan.integrate_frame(&data, false).is_ok());
                    data[1] = f64::NAN;
                    assert!(plan.integrate_frame(&data, false).is_err());
                    assert!(plan.integrate_frame(&[1.], false).is_err());
                }
            }
        }
    }
    #[test]
    fn csr_rejects_malformed_indices_and_normalization() {
        let g = geometry(false);
        let lut = lut::build_lut1d_pyfai_csr(&g, RadialUnit::TthDeg, 20, None, None).unwrap();
        let mut bad = lut.clone();
        bad.col_idx[0] = g.size();
        assert!(FixedCsr1dPlan::new(bad, vec![1.; g.size()], None).is_err());
        let mut bad = lut.clone();
        bad.row_ptr[1] = usize::MAX;
        assert!(FixedCsr1dPlan::new(bad, vec![1.; g.size()], None).is_err());
        let mut n = vec![1.; g.size()];
        n[0] = f64::NAN;
        assert!(FixedCsr1dPlan::new(lut, n, None).is_err());
    }
}
