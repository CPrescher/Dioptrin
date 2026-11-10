//! Immutable CSR snapshots retaining the existing reduction order.
use super::{Fixed1dResult, InputOrder, IntegrationGrid2d, Integrator, MaskMode};
use crate::{
    lut::{
        Lut1d,
        prepared::{Columns, Prepared1d, Prepared2d},
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

/// Shared strict CSR execution directly from a decoder's integer pixels.
/// Construction maps input addresses once, without changing bin reduction order.
/// Serial entry points execute on the caller's thread for parallelism across
/// frames. Explicit parallel entry points schedule independent bins for a
/// single image, retaining the same strict arithmetic within each bin.
pub struct FixedCsr1dInputPlan {
    plan: Arc<FixedCsr1dPlan>,
    shape: (usize, usize),
    columns: Option<Columns>,
}
impl FixedCsr1dInputPlan {
    pub fn new(
        plan: Arc<FixedCsr1dPlan>,
        shape: (usize, usize),
        order: InputOrder,
    ) -> Result<Self, String> {
        if shape.0 == 0 || shape.1 == 0 || shape.0.checked_mul(shape.1) != Some(plan.norm.len()) {
            return Err("CSR input shape must match the prepared detector".into());
        }
        let columns = (order == InputOrder::FlipVertical)
            .then(|| plan.lut.input_columns(plan.mask.as_deref(), shape));
        Ok(Self {
            plan,
            shape,
            columns,
        })
    }
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
    pub fn integrate_u16(&self, pixels: &[u16], errors: bool) -> Result<Fixed1dResult, String> {
        self.integrate::<_, false>(pixels, errors)
    }
    pub fn integrate_u32(&self, pixels: &[u32], errors: bool) -> Result<Fixed1dResult, String> {
        self.integrate::<_, false>(pixels, errors)
    }
    /// Integrate one image with independent bins scheduled on the current Rayon
    /// pool. Each bin retains the serial API's strict summation order.
    pub fn integrate_u16_parallel(
        &self,
        pixels: &[u16],
        errors: bool,
    ) -> Result<Fixed1dResult, String> {
        self.integrate::<_, true>(pixels, errors)
    }
    /// Parallel-bin counterpart of `integrate_u32`, with identical result bits.
    pub fn integrate_u32_parallel(
        &self,
        pixels: &[u32],
        errors: bool,
    ) -> Result<Fixed1dResult, String> {
        self.integrate::<_, true>(pixels, errors)
    }
    fn integrate<P: crate::frame::Pixel, const PARALLEL: bool>(
        &self,
        pixels: &[P],
        errors: bool,
    ) -> Result<Fixed1dResult, String> {
        if pixels.len() != self.plan.norm.len() {
            return Err("image size mismatch".into());
        }
        let run = |errors| {
            if errors {
                self.plan.lut.integrate_input::<true, P, PARALLEL>(
                    pixels,
                    self.plan.mask.as_deref(),
                    &self.plan.norm,
                    self.columns.as_ref(),
                )
            } else {
                self.plan.lut.integrate_input::<false, P, PARALLEL>(
                    pixels,
                    self.plan.mask.as_deref(),
                    &self.plan.norm,
                    self.columns.as_ref(),
                )
            }
        };
        let (radial, intensity, sigma) = run(errors);
        Ok(Fixed1dResult {
            radial,
            intensity,
            errors: errors.then_some(sigma),
        })
    }
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
    fn integer_input_preserves_strict_bits_for_orders_masks_ragged_and_duplicate_rows() {
        let shape = (7, 11);
        let pixels = shape.0 * shape.1;
        for weighted in [false, true] {
            for masking in 0..3 {
                let mut row_ptr = vec![0];
                let mut col_idx = vec![];
                let mut weights = vec![];
                for r in 0..19 {
                    let count = if r % 7 == 0 { 0 } else { (r * 19) % 41 };
                    for k in 0..count {
                        col_idx.push((k * 53 + r * 17) % pixels);
                        weights.push(if k % 11 == 0 {
                            1e-310
                        } else {
                            ((k * 997 + r * 11) % 4093) as f64 / 4093.
                        });
                    }
                    row_ptr.push(col_idx.len());
                }
                let lut = lut::Lut1d {
                    row_ptr,
                    col_idx,
                    values: weighted.then_some(weights),
                    hist_min: 0.1,
                    hist_step: 0.2,
                };
                let norm = (0..pixels)
                    .map(|i| {
                        if i % 31 == 0 {
                            0.
                        } else {
                            0.2 + i as f64 / 131.
                        }
                    })
                    .collect();
                let mask = (0..pixels)
                    .map(|i| u8::from(masking == 2 || (masking == 1 && i % 3 == 0)))
                    .collect();
                let plan = Arc::new(FixedCsr1dPlan::new(lut, norm, Some(mask)).unwrap());
                let raw32: Vec<u32> = (0..pixels)
                    .map(|i| {
                        if i % 13 == 0 {
                            u32::MAX
                        } else {
                            (i as u32).wrapping_mul(2654435761)
                        }
                    })
                    .collect();
                let raw16: Vec<u16> = raw32.iter().map(|v| *v as u16).collect();
                for order in [InputOrder::Detector, InputOrder::FlipVertical] {
                    let typed = FixedCsr1dInputPlan::new(plan.clone(), shape, order).unwrap();
                    assert_eq!(typed.shape(), shape);
                    for errors in [false, true] {
                        let compare = |raw: Vec<f64>, actual: Fixed1dResult| {
                            let oriented = if order == InputOrder::FlipVertical {
                                raw.chunks_exact(shape.1).rev().flatten().copied().collect()
                            } else {
                                raw
                            };
                            let expected = plan.integrate_frame(&oriented, errors).unwrap();
                            same(&actual.radial, &expected.radial);
                            same(&actual.intensity, &expected.intensity);
                            if errors {
                                same(&actual.errors.unwrap(), &expected.errors.unwrap());
                            }
                        };
                        compare(
                            raw32.iter().map(|&v| v as f64).collect(),
                            typed.integrate_u32(&raw32, errors).unwrap(),
                        );
                        compare(
                            raw16.iter().map(|&v| v as f64).collect(),
                            typed.integrate_u16(&raw16, errors).unwrap(),
                        );
                        for threads in [1, 2, 4] {
                            let pool = rayon::ThreadPoolBuilder::new()
                                .num_threads(threads)
                                .build()
                                .unwrap();
                            pool.install(|| {
                                compare(
                                    raw32.iter().map(|&v| v as f64).collect(),
                                    typed.integrate_u32_parallel(&raw32, errors).unwrap(),
                                );
                                compare(
                                    raw16.iter().map(|&v| v as f64).collect(),
                                    typed.integrate_u16_parallel(&raw16, errors).unwrap(),
                                );
                            });
                        }
                    }
                    assert!(typed.integrate_u32(&raw32[..pixels - 1], true).is_err());
                    assert!(
                        typed
                            .integrate_u16_parallel(&raw16[..pixels - 1], false)
                            .is_err()
                    );
                    assert!(
                        typed
                            .integrate_u32_parallel(&raw32[..pixels - 1], true)
                            .is_err()
                    );
                    let expected = typed.integrate_u32(&raw32, true).unwrap();
                    std::thread::scope(|scope| {
                        for _ in 0..4 {
                            let (typed, raw, expected) = (&typed, &raw32, &expected);
                            scope.spawn(move || {
                                let actual = typed.integrate_u32(raw, true).unwrap();
                                same(&actual.intensity, &expected.intensity);
                                same(
                                    actual.errors.as_ref().unwrap(),
                                    expected.errors.as_ref().unwrap(),
                                );
                            });
                        }
                    });
                }
                for invalid in [(0, 11), (7, 0), (1, pixels - 1), (usize::MAX, 2)] {
                    assert!(
                        FixedCsr1dInputPlan::new(plan.clone(), invalid, InputOrder::FlipVertical)
                            .is_err()
                    );
                }
            }
        }
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
