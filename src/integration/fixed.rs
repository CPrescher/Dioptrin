//! Fixed-mask, float64 integration of ordered native pixel blocks.
use super::{IntegrationMethod, Integrator, MaskMode, SparseLayout};
use crate::{
    frame::Pixel,
    lut::{
        csc::{FixedRows, Sums},
        prepared::Prepared1d,
    },
    mask::Mask,
};
use std::{
    hint::black_box,
    ops::Range,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

/// Mapping from incoming row-major pixels to the calibrated detector image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputOrder {
    Detector,
    FlipVertical,
}

/// Immutable geometry, float64 weights, fixed mask and cached normalization.
/// Workers borrow this snapshot; later changes to its source integrator are independent.
pub struct Fixed1dPlan {
    lut: Prepared1d,
    shape: (usize, usize),
    mask: Option<Mask>,
    normalization: Arc<Vec<f64>>,
    resident: OnceLock<Option<FixedRows>>,
    resident_choice: [OnceLock<bool>; 2],
    auto_layout: bool,
    force_csc: bool,
    auto_choice: [OnceLock<SparseLayout>; 2],
}

pub struct Fixed1dResult {
    pub radial: Vec<f64>,
    pub intensity: Vec<f64>,
    pub errors: Option<Vec<f64>>,
}

/// One worker reuses its four float64 partial histograms across frames.
/// Call begin, push contiguous blocks, then finish only after the reader succeeds.
/// A read error requires aborting; no partial scientific result should be published.
pub struct Fixed1dWorker<'a> {
    plan: &'a Fixed1dPlan,
    sums: Sums,
    order: InputOrder,
    next: usize,
    range: Range<usize>,
    active: bool,
    failed: bool,
}

impl Integrator {
    /// Snapshot fast CSC 1D execution with the current fixed mask/corrections.
    /// Dynamic masks, strict arithmetic, CSR selection and histogram are rejected.
    pub fn prepare_fixed1d(
        &self,
        shape: (usize, usize),
        n_points: usize,
    ) -> Result<Fixed1dPlan, String> {
        let pixels = shape.0.checked_mul(shape.1).ok_or("image shape overflow")?;
        if pixels == 0 || pixels > u32::MAX as usize || n_points == 0 {
            return Err("positive compact CSC dimensions required".into());
        }
        if self.mask_mode != MaskMode::Fixed
            || !self.fast_math
            || self.sparse_layout != SparseLayout::Csc
        {
            return Err("fixed block plans require fixed masks, fast arithmetic and CSC".into());
        }
        if matches!(
            self.method,
            IntegrationMethod::Histogram | IntegrationMethod::Supersampled { n_ss: 0 }
        ) {
            return Err("unsupported fixed block integration method".into());
        }
        let mut snapshot = Integrator::new(self.geometry.clone())
            .with_method(self.method)
            .with_radial_unit(self.radial_unit);
        snapshot.polarization_factor = self.polarization_factor;
        snapshot.radial_range = self.radial_range;
        if let Some(mask) = &self.mask {
            if (mask.dim1, mask.dim2) != shape || mask.data.len() != pixels {
                return Err("mask does not match image shape".into());
            }
            snapshot.mask = Some(Mask {
                data: mask.data.clone(),
                dim1: shape.0,
                dim2: shape.1,
            });
        }
        snapshot.prepare1d(shape.0, shape.1, n_points);
        let normalization = snapshot.get_normalization();
        let mask = snapshot.mask.take();
        let lut = snapshot
            .cache_1d
            .into_inner()
            .map_err(|_| "poisoned lookup cache")?
            .lut
            .ok_or("missing prepared lookup table")?;
        if lut.fixed_csc().is_none() {
            return Err("lookup table does not fit compact CSC".into());
        }
        Ok(Fixed1dPlan {
            lut,
            shape,
            mask,
            normalization,
            resident: OnceLock::new(),
            resident_choice: [OnceLock::new(), OnceLock::new()],
            auto_layout: false,
            force_csc: false,
            auto_choice: [OnceLock::new(), OnceLock::new()],
        })
    }
}

impl Fixed1dPlan {
    /// Force the pixel-oriented kernel, including complete resident frames.
    /// This bypasses the optional gather cache and automatic timing probes.
    pub fn with_csc_layout(mut self) -> Self {
        self.force_csc = true;
        self.auto_layout = false;
        self
    }
    /// Opt in to a local CSC/CSR timing choice for complete resident frames.
    /// CSC remains the baseline. CSR must win at least four of five paired
    /// probes and improve the median by more than 15%. Choices are cached per
    /// error mode. Streamed blocks and outer Rayon workers retain CSC.
    /// CSR uses identical weights but can differ by floating-point sum order.
    pub fn with_auto_layout(mut self) -> Self {
        self.auto_layout = true;
        self
    }

    /// Locally measured resident choice, if automatic selection has run.
    pub fn selected_resident_layout(&self, errors: bool) -> Option<SparseLayout> {
        self.auto_choice[usize::from(errors)].get().copied()
    }
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
    pub fn worker(&self, errors: bool, order: InputOrder) -> Fixed1dWorker<'_> {
        Fixed1dWorker {
            plan: self,
            sums: self.lut.fixed_csc().unwrap().block_sums(errors),
            order,
            next: 0,
            range: 0..self.shape.0 * self.shape.1,
            active: false,
            failed: false,
        }
    }
    /// Merge completed, ordered ranges covering one full frame, then normalize once.
    /// Every worker must borrow this plan and use the same uncertainty setting.
    /// Call only after the reader returns all states successfully.
    pub fn finish_ranges(
        &self,
        workers: &mut [Fixed1dWorker<'_>],
    ) -> Result<Fixed1dResult, String> {
        let Some(first) = workers.first() else {
            return Err("no block workers".into());
        };
        let errors = !first.sums.variance.is_empty();
        let order = first.order;
        let mut next = 0;
        for worker in workers.iter() {
            if !std::ptr::eq(worker.plan, self)
                || worker.order != order
                || !worker.active
                || worker.failed
                || worker.range.start != next
                || worker.next != worker.range.end
                || (!worker.sums.variance.is_empty()) != errors
            {
                return Err(
                    "expected complete, ordered ranges from the same plan and error mode".into(),
                );
            }
            next = worker.range.end;
        }
        if next != self.shape.0 * self.shape.1 {
            return Err("ranges do not cover the frame".into());
        }
        let mut merged = self.lut.fixed_csc().unwrap().block_sums(errors);
        for worker in workers {
            for (sum, value) in merged.signal.iter_mut().zip(&worker.sums.signal) {
                *sum += value;
            }
            for (sum, value) in merged.variance.iter_mut().zip(&worker.sums.variance) {
                *sum += value;
            }
            worker.active = false;
        }
        Ok(self.result_from_sums(&merged))
    }

    fn result_from_sums(&self, sums: &Sums) -> Fixed1dResult {
        let norm = self.lut.fixed_bin_norm();
        let n = norm.len();
        let reduce = |s: &[f64], b: usize| (s[b] + s[n + b]) + (s[2 * n + b] + s[3 * n + b]);
        let intensity = norm
            .iter()
            .enumerate()
            .map(|(b, &v)| {
                if v == 0. {
                    0.
                } else {
                    reduce(&sums.signal, b) / v
                }
            })
            .collect();
        let errors = (!sums.variance.is_empty()).then(|| {
            norm.iter()
                .enumerate()
                .map(|(b, &v)| {
                    if v == 0. {
                        0.
                    } else {
                        (reduce(&sums.variance, b) / (v * v)).max(0.).sqrt()
                    }
                })
                .collect()
        });
        Fixed1dResult {
            radial: self.lut.bins(),
            intensity,
            errors,
        }
    }

    /// Complete resident frames can use a bin-oriented cache with the same CSC
    /// reduction order. The first call in each error mode measures both layouts;
    /// gather is retained only with a clear measured advantage. Streamed blocks
    /// and outer Rayon batch workers retain CSC without probing.
    pub fn integrate_frame(&self, data: &[f64], errors: bool) -> Result<Fixed1dResult, String> {
        if data.len() != self.shape.0 * self.shape.1 {
            return Err("image size mismatch".into());
        }
        if self.auto_layout {
            if rayon::current_thread_index().is_some() {
                return Ok(self.integrate_csc(data, errors));
            }
            let choice = self.auto_choice[usize::from(errors)]
                .get_or_init(|| self.measure_auto(data, errors));
            return Ok(if *choice == SparseLayout::Csr {
                self.integrate_csr(data, errors)
            } else {
                self.integrate_csc(data, errors)
            });
        }
        if !self.force_csc
            && rayon::current_thread_index().is_none()
            && let Some(resident) = self.resident.get_or_init(|| {
                FixedRows::new(self.lut.fixed_csc().unwrap(), self.lut.fixed_pixels())
            })
        {
            let selected = self.resident_choice[usize::from(errors)]
                .get_or_init(|| self.measure_resident(resident, data, errors));
            if *selected {
                return Ok(self.integrate_rows(resident, data, errors));
            }
        }
        Ok(self.integrate_csc(data, errors))
    }

    fn integrate_rows(&self, rows: &FixedRows, data: &[f64], errors: bool) -> Fixed1dResult {
        let (intensity, sigma) = if errors {
            rows.integrate::<true>(data, self.lut.fixed_bin_norm())
        } else {
            rows.integrate::<false>(data, self.lut.fixed_bin_norm())
        };
        Fixed1dResult {
            radial: self.lut.bins(),
            intensity,
            errors: errors.then_some(sigma),
        }
    }

    fn measure_resident(&self, rows: &FixedRows, data: &[f64], errors: bool) -> bool {
        // Warm both layouts, then alternate their order. This is a plan-local
        // choice, not a pixel-count threshold inferred from one workstation.
        drop(black_box(self.integrate_csc(data, errors)));
        drop(black_box(self.integrate_rows(rows, data, errors)));
        let mut csc = [Duration::ZERO; 5];
        let mut gather = [Duration::ZERO; 5];
        for i in 0..5 {
            for use_rows in [i % 2 == 0, i % 2 != 0] {
                let start = Instant::now();
                drop(black_box(if use_rows {
                    self.integrate_rows(rows, data, errors)
                } else {
                    self.integrate_csc(data, errors)
                }));
                let elapsed = start.elapsed();
                if use_rows {
                    gather[i] = elapsed;
                } else {
                    csc[i] = elapsed;
                }
            }
        }
        resident_wins(csc, gather)
    }

    fn integrate_csc(&self, data: &[f64], errors: bool) -> Fixed1dResult {
        let mask = self.mask.as_ref().map(|m| m.data.as_slice());
        let (radial, intensity, sigma) = if errors {
            self.lut
                .integrate::<true, true>(data, mask, &self.normalization)
        } else {
            self.lut
                .integrate::<true, false>(data, mask, &self.normalization)
        };
        Fixed1dResult {
            radial,
            intensity,
            errors: errors.then_some(sigma),
        }
    }

    fn integrate_csr(&self, data: &[f64], errors: bool) -> Fixed1dResult {
        let mask = self.mask.as_ref().map(|m| m.data.as_slice());
        let (radial, intensity, sigma) = if errors {
            self.lut
                .integrate_fixed_csr::<true>(data, mask, &self.normalization)
        } else {
            self.lut
                .integrate_fixed_csr::<false>(data, mask, &self.normalization)
        };
        Fixed1dResult {
            radial,
            intensity,
            errors: errors.then_some(sigma),
        }
    }

    fn measure_auto(&self, data: &[f64], errors: bool) -> SparseLayout {
        let prior = super::predict_1d_layout(
            data.len(),
            self.lut.bins().len(),
            errors,
            false,
            rayon::current_num_threads(),
        );
        let csr_first = prior.layout == SparseLayout::Csr;
        // Include complete allocations/output in both probes. Both fixed masks
        // and normalization caches are warm before measurement.
        drop(black_box(self.integrate_csc(data, errors)));
        drop(black_box(self.integrate_csr(data, errors)));
        let mut csc = [Duration::ZERO; 5];
        let mut csr = [Duration::ZERO; 5];
        for i in 0..5 {
            for use_csr in [csr_first ^ (i % 2 != 0), !csr_first ^ (i % 2 != 0)] {
                let start = Instant::now();
                drop(black_box(if use_csr {
                    self.integrate_csr(data, errors)
                } else {
                    self.integrate_csc(data, errors)
                }));
                if use_csr {
                    csr[i] = start.elapsed();
                } else {
                    csc[i] = start.elapsed();
                }
            }
        }
        let wins = csc.iter().zip(&csr).filter(|(a, b)| b < a).count();
        csc.sort_unstable();
        csr.sort_unstable();
        if wins >= 4 && csr[2].as_secs_f64() < 0.85 * csc[2].as_secs_f64() {
            SparseLayout::Csr
        } else {
            SparseLayout::Csc
        }
    }
}

fn resident_wins(mut csc: [Duration; 5], mut gather: [Duration; 5]) -> bool {
    let wins = csc.iter().zip(&gather).filter(|(a, b)| b < a).count();
    csc.sort_unstable();
    gather.sort_unstable();
    wins >= 4 && gather[2].as_secs_f64() < 0.9 * csc[2].as_secs_f64()
}

impl Fixed1dWorker<'_> {
    /// Start/restart a frame, discarding any incomplete or aborted accumulation.
    pub fn begin_frame(&mut self) {
        self.begin_range(0..self.plan.shape.0 * self.plan.shape.1)
            .unwrap();
    }
    /// Begin a nonempty contiguous part of a frame for parallel block integration.
    pub fn begin_range(&mut self, range: Range<usize>) -> Result<(), String> {
        self.abort_frame();
        if range.start >= range.end || range.end > self.plan.shape.0 * self.plan.shape.1 {
            return Err("invalid integration range".into());
        }
        self.sums.signal.fill(0.);
        self.sums.variance.fill(0.);
        self.next = range.start;
        self.range = range;
        self.active = true;
        self.failed = false;
        Ok(())
    }
    pub fn abort_frame(&mut self) {
        self.active = false;
        self.failed = true;
    }
    pub fn push_u16_block(&mut self, offset: usize, pixels: &[u16]) -> Result<(), String> {
        self.push(offset, pixels)
    }
    pub fn push_u32_block(&mut self, offset: usize, pixels: &[u32]) -> Result<(), String> {
        self.push(offset, pixels)
    }
    pub fn push_f64_block(&mut self, offset: usize, pixels: &[f64]) -> Result<(), String> {
        self.push(offset, pixels)
    }
    fn push<P: Pixel>(&mut self, offset: usize, pixels: &[P]) -> Result<(), String> {
        if !self.active
            || self.failed
            || offset != self.next
            || pixels.is_empty()
            || offset
                .checked_add(pixels.len())
                .is_none_or(|end| end > self.range.end)
        {
            self.failed = true;
            return Err(
                "expected a nonempty, contiguous, in-bounds block in an active frame".into(),
            );
        }
        let table = &self.plan.lut;
        let csc = table.fixed_csc().unwrap();
        let active = table.fixed_pixels();
        let mut local = 0;
        while local < pixels.len() {
            let source = offset + local;
            let width = self.plan.shape.1;
            let count = if self.order == InputOrder::Detector {
                pixels.len() - local
            } else {
                (width - source % width).min(pixels.len() - local)
            };
            let start = if self.order == InputOrder::Detector {
                source
            } else {
                (self.plan.shape.0 - 1 - source / width) * width + source % width
            };
            let first = active.partition_point(|&p| (p as usize) < start);
            let last = active.partition_point(|&p| (p as usize) < start + count);
            csc.add_fixed_block(
                &pixels[local..local + count],
                start,
                &active[first..last],
                &mut self.sums,
            );
            local += count;
        }
        self.next += pixels.len();
        Ok(())
    }
    pub fn finish_frame(&mut self) -> Result<Fixed1dResult, String> {
        if !self.active
            || self.failed
            || self.next != self.range.end
            || self.range != (0..self.plan.shape.0 * self.plan.shape.1)
        {
            self.abort_frame();
            return Err("cannot finish an incomplete, partial or failed frame".into());
        }
        self.active = false;
        Ok(self.plan.result_from_sums(&self.sums))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{geometry::Geometry, poni::Poni};
    fn integrator(method: IntegrationMethod) -> Integrator {
        let mut poni = Poni::new();
        poni.pixel1 = 1e-4;
        poni.pixel2 = 1e-4;
        poni.distance = 0.15;
        poni.poni1 = 0.0007;
        poni.poni2 = 0.0012;
        Integrator::new(Geometry::new(17, 23, poni))
            .with_method(method)
            .with_polarization_factor(0.965)
            .with_mask(Mask {
                data: (0..391).map(|i| u8::from(i % 11 == 0)).collect(),
                dim1: 17,
                dim2: 23,
            })
    }
    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (&a, &b) in a.iter().zip(b) {
            assert!((a - b).abs() <= 1e-10 + 2e-12 * b.abs(), "{a} != {b}");
        }
    }
    #[test]
    fn resident_frame_is_bitwise_csc_and_keeps_outer_workers_on_csc() {
        let original = integrator(IntegrationMethod::PixelSplit);
        let shape = (768, 768);
        let pixels = shape.0 * shape.1;
        let engine = Integrator::new(Geometry::new(
            shape.0,
            shape.1,
            original.geometry.poni.clone(),
        ))
        .with_method(IntegrationMethod::PixelSplit)
        .with_polarization_factor(0.965)
        .with_mask(Mask {
            data: (0..pixels).map(|i| u8::from(i % 11 == 0)).collect(),
            dim1: shape.0,
            dim2: shape.1,
        });
        let plan = engine.prepare_fixed1d(shape, 128).unwrap();
        let data: Vec<_> = (0..pixels)
            .map(|i| ((i * 997) % 1009) as f64 - 31.7)
            .collect();
        let mask = plan.mask.as_ref().map(|m| m.data.as_slice());
        let (x, y, e) = plan
            .lut
            .integrate::<true, true>(&data, mask, &plan.normalization);
        let check = |r: Fixed1dResult| {
            for (a, b) in [
                (&r.radial, &x),
                (&r.intensity, &y),
                (r.errors.as_ref().unwrap(), &e),
            ] {
                assert_eq!(
                    a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        };
        assert!(plan.resident.get().is_none());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        check(pool.install(|| plan.integrate_frame(&data, true).unwrap()));
        assert!(
            plan.resident.get().is_none(),
            "outer batch workers must not allocate resident tables"
        );
        check(plan.integrate_frame(&data, true).unwrap());
        assert!(plan.resident.get().unwrap().is_some());
        assert!(plan.resident_choice[1].get().is_some());
        assert!(plan.resident_choice[0].get().is_none());
        check(plan.integrate_frame(&data, true).unwrap());
        check(pool.install(|| plan.integrate_frame(&data, true).unwrap()));
        let intensity = plan.integrate_frame(&data, false).unwrap();
        assert!(plan.resident_choice[0].get().is_some());
        assert_eq!(intensity.intensity, y);
        assert!(intensity.errors.is_none());
        // Exercise both decisions regardless of scheduler noise during probes.
        for use_rows in [false, true] {
            let forced = engine.prepare_fixed1d(shape, 128).unwrap();
            forced.resident_choice[1].set(use_rows).unwrap();
            check(forced.integrate_frame(&data, true).unwrap());
        }
    }

    #[test]
    fn resident_selection_requires_consistent_ten_percent_advantage() {
        let times = |ns: [u64; 5]| ns.map(Duration::from_nanos);
        let csc = times([100; 5]);
        assert!(resident_wins(csc, times([80, 81, 79, 80, 150])));
        assert!(!resident_wins(csc, times([95; 5])));
        assert!(!resident_wins(csc, times([80, 80, 80, 120, 120])));
        assert!(!resident_wins(csc, times([110; 5])));
        assert!(!resident_wins([Duration::ZERO; 5], [Duration::ZERO; 5]));
    }

    #[test]
    fn automatic_layout_keeps_weights_errors_and_changed_samples() {
        let mut engine = integrator(IntegrationMethod::PixelSplit);
        let shape = (17, 23);
        let mut mask = vec![0; shape.0 * shape.1];
        for i in (0..mask.len()).step_by(7) {
            mask[i] = 1;
        }
        engine.set_mask(Mask {
            data: mask.clone(),
            dim1: shape.0,
            dim2: shape.1,
        });
        let csc = engine
            .prepare_fixed1d(shape, 128)
            .unwrap()
            .with_csc_layout();
        let auto = engine
            .prepare_fixed1d(shape, 128)
            .unwrap()
            .with_auto_layout();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let data: Vec<_> = (0..mask.len())
            .map(|i| {
                if mask[i] != 0 {
                    f64::NAN
                } else {
                    (i % 31) as f64 + 2.
                }
            })
            .collect();
        pool.install(|| auto.integrate_frame(&data, true).unwrap());
        assert!(auto.selected_resident_layout(true).is_none());
        for errors in [false, true] {
            for mode in [SparseLayout::Csr, SparseLayout::Csc] {
                let plan = engine
                    .prepare_fixed1d(shape, 128)
                    .unwrap()
                    .with_auto_layout();
                plan.auto_choice[usize::from(errors)].set(mode).unwrap();
                for scale in [1., 1.7] {
                    let frame: Vec<_> = data.iter().map(|x| x * scale).collect();
                    let expected = csc.integrate_frame(&frame, errors).unwrap();
                    let actual = plan.integrate_frame(&frame, errors).unwrap();
                    assert_eq!(actual.radial, expected.radial);
                    for (a, b) in actual.intensity.iter().zip(&expected.intensity) {
                        assert!((a - b).abs() < 2e-12 * b.abs().max(1.));
                    }
                    assert_eq!(actual.errors.is_some(), errors);
                    if let (Some(a), Some(b)) = (actual.errors, expected.errors) {
                        for (a, b) in a.iter().zip(b) {
                            assert!((a - b).abs() < 2e-12 * b.abs().max(1.));
                        }
                    }
                }
            }
            auto.integrate_frame(&data, errors).unwrap();
            assert!(auto.selected_resident_layout(errors).is_some());
        }
        assert!(csc.resident.get().is_none());
        assert!(csc.integrate_frame(&data[..10], false).is_err());
    }
    #[test]
    fn native_blocks_preserve_float64_weights_masks_errors_and_orientation() {
        let samples: Vec<u32> = (0..391)
            .map(|i| {
                if i % 13 == 0 {
                    u32::MAX
                } else {
                    (i * 139) as u32
                }
            })
            .collect();
        for method in [
            IntegrationMethod::Counting,
            IntegrationMethod::PixelSplit,
            IntegrationMethod::Supersampled { n_ss: 3 },
        ] {
            let ai = integrator(method);
            let plan = ai.prepare_fixed1d((17, 23), 37).unwrap();
            for order in [InputOrder::Detector, InputOrder::FlipVertical] {
                let mut full: Vec<f64> = samples.iter().map(|&x| x as f64).collect();
                if order == InputOrder::FlipVertical {
                    full = full
                        .as_chunks::<23>()
                        .0
                        .iter()
                        .rev()
                        .flatten()
                        .copied()
                        .collect();
                }
                let (radial, expected, sigma) =
                    ai.integrate1d_with_errors_from_slice(&full, 17, 23, 37);
                let fallback = plan.integrate_frame(&full, true).unwrap();
                assert_eq!(fallback.radial, radial);
                close(&fallback.intensity, &expected);
                for errors in [false, true] {
                    for size in [1, 7, 23, 31, 391] {
                        let mut worker = plan.worker(errors, order);
                        for _ in 0..2 {
                            worker.begin_frame();
                            for (i, block) in samples.chunks(size).enumerate() {
                                worker.push_u32_block(i * size, block).unwrap();
                            }
                            let result = worker.finish_frame().unwrap();
                            assert_eq!(result.radial, radial);
                            close(&result.intensity, &expected);
                            if errors {
                                close(result.errors.as_ref().unwrap(), &sigma);
                            } else {
                                assert!(result.errors.is_none());
                            }
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn sessions_reject_gaps_duplicates_incomplete_and_aborted_frames() {
        let plan = integrator(IntegrationMethod::PixelSplit)
            .prepare_fixed1d((17, 23), 37)
            .unwrap();
        let data = vec![65535_u16; 391];
        let mut w = plan.worker(true, InputOrder::Detector);
        assert!(w.finish_frame().is_err());
        w.begin_frame();
        assert!(w.push_u16_block(1, &data).is_err());
        assert!(w.push_u16_block(0, &data).is_err());
        w.begin_frame();
        w.push_u16_block(0, &data[..10]).unwrap();
        assert!(w.finish_frame().is_err());
        w.begin_frame();
        w.push_u16_block(0, &data).unwrap();
        w.finish_frame().unwrap();
        assert!(w.finish_frame().is_err());
        w.begin_frame();
        w.abort_frame();
        assert!(w.finish_frame().is_err());
        w.begin_frame();
        w.push_f64_block(0, &[65535.; 391]).unwrap();
        let a = w.finish_frame().unwrap();
        let b = plan.integrate_frame(&[65535.; 391], true).unwrap();
        close(&a.intensity, &b.intensity);
    }
    #[test]
    fn fixed_snapshot_is_shared_and_independent_of_later_settings() {
        let mut ai = integrator(IntegrationMethod::PixelSplit);
        let plan = ai.prepare_fixed1d((17, 23), 37).unwrap();
        let full = vec![8.; 391];
        let expected = plan.integrate_frame(&full, true).unwrap();
        ai.set_mask(Mask {
            data: vec![1; 391],
            dim1: 17,
            dim2: 23,
        });
        ai.set_polarization_factor(Some(0.4));
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let (plan, expected) = (&plan, &expected);
                scope.spawn(move || {
                    let mut w = plan.worker(true, InputOrder::Detector);
                    w.begin_frame();
                    w.push_u16_block(0, &[8; 391]).unwrap();
                    let result = w.finish_frame().unwrap();
                    close(&result.intensity, &expected.intensity);
                });
            }
        });
        ai.set_mask_mode(MaskMode::Dynamic);
        assert!(ai.prepare_fixed1d((17, 23), 37).is_err());
        ai.set_mask_mode(MaskMode::Fixed);
        ai.set_fast_math(false);
        assert!(ai.prepare_fixed1d((17, 23), 37).is_err());
        ai.set_fast_math(true);
        ai.set_sparse_layout(SparseLayout::Csr);
        assert!(ai.prepare_fixed1d((17, 23), 37).is_err());
    }
    #[test]
    fn parallel_ranges_merge_before_normalization_and_reject_bad_coverage() {
        let ai = integrator(IntegrationMethod::PixelSplit);
        let plan = ai.prepare_fixed1d((17, 23), 37).unwrap();
        let values: Vec<u32> = (0..391).map(|p| (p * 331) as u32).collect();
        let mut full: Vec<f64> = values.iter().map(|&v| v as f64).collect();
        for order in [InputOrder::Detector, InputOrder::FlipVertical] {
            if order == InputOrder::FlipVertical {
                full = full
                    .as_chunks::<23>()
                    .0
                    .iter()
                    .rev()
                    .flatten()
                    .copied()
                    .collect();
            }
            for errors in [false, true] {
                for count in [1, 4, 8] {
                    let mut workers: Vec<_> =
                        (0..count).map(|_| plan.worker(errors, order)).collect();
                    for (i, worker) in workers.iter_mut().enumerate() {
                        let range = i * values.len() / count..(i + 1) * values.len() / count;
                        worker.begin_range(range.clone()).unwrap();
                        worker.push_u32_block(range.start, &values[range]).unwrap();
                    }
                    let result = plan.finish_ranges(&mut workers).unwrap();
                    let expected = plan.integrate_frame(&full, errors).unwrap();
                    close(&result.intensity, &expected.intensity);
                    if errors {
                        close(
                            result.errors.as_ref().unwrap(),
                            expected.errors.as_ref().unwrap(),
                        );
                    }
                    assert!(plan.finish_ranges(&mut workers).is_err());
                }
            }
        }
        let mut partial = plan.worker(false, InputOrder::Detector);
        partial.begin_range(1..391).unwrap();
        partial.push_u32_block(1, &values[1..]).unwrap();
        assert!(
            plan.finish_ranges(std::slice::from_mut(&mut partial))
                .is_err()
        );
        assert!(partial.finish_frame().is_err());
        assert!(partial.begin_range(12..12).is_err());
        assert!(partial.begin_range(0..392).is_err());
        let mut a = plan.worker(false, InputOrder::Detector);
        a.begin_range(0..200).unwrap();
        a.push_u32_block(0, &values[..200]).unwrap();
        let mut b = plan.worker(true, InputOrder::Detector);
        b.begin_range(200..391).unwrap();
        b.push_u32_block(200, &values[200..]).unwrap();
        assert!(plan.finish_ranges(&mut [a, b]).is_err());
        let mut a = plan.worker(false, InputOrder::Detector);
        a.begin_range(0..200).unwrap();
        a.push_u32_block(0, &values[..200]).unwrap();
        let mut b = plan.worker(false, InputOrder::FlipVertical);
        b.begin_range(200..391).unwrap();
        b.push_u32_block(200, &values[200..]).unwrap();
        assert!(plan.finish_ranges(&mut [a, b]).is_err());
    }
}
