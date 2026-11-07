use std::sync::Arc;

use super::Integrator;

impl Integrator {
    /// Checks whether current tth, radial, azimuth, solid angle and polarization are calculated for the same
    /// image dimension as dim1 x dim2, if not, they are recalculated.
    /// Thread-safe: uses RwLock for interior mutability.
    pub(super) fn update_geometry_arrays(&self, dim1: usize, dim2: usize) {
        let size = dim1 * dim2;

        // First check with read lock if update is needed
        let needs_update = {
            let state = self.state.read().expect("geometry state lock poisoned");
            (state.dim1 * state.dim2) != size || state.dim1 != dim1 || state.dim2 != dim2
        };

        if needs_update {
            self.refresh_geometry_arrays(dim1, dim2);
        }
    }

    fn refresh_geometry_arrays(&self, dim1: usize, dim2: usize) {
        let size = dim1 * dim2;
        // Cache readers may read normalization from state. Acquire cache
        // writers first, then state, and publish geometry plus invalidation
        // together. A late cold-start worker must not clear another worker's
        // freshly built LUT after discovering that geometry is already ready.
        let mut cache_1d = self.cache_1d.write().expect("1D LUT cache lock poisoned");
        let mut cache_2d = self.cache_2d.write().expect("2D LUT cache lock poisoned");
        let mut state = self.state.write().expect("geometry state lock poisoned");
        if (state.dim1 * state.dim2) != size || state.dim1 != dim1 || state.dim2 != dim2 {
            // Update geometry dimensions temporarily for calculations
            let mut temp_geometry = self.geometry.clone();
            temp_geometry.dim1 = dim1;
            temp_geometry.dim2 = dim2;

            state.dim1 = dim1;
            state.dim2 = dim2;
            state.tth = temp_geometry.calc_tth_all();
            state.radial = self
                .radial_unit
                .convert_from_tth(&state.tth, self.geometry.poni.wavelength);
            state.azimuth = temp_geometry.calc_azimuth_all();
            state.solid_angle = temp_geometry.calc_solid_angle();

            // Update polarization
            state.polarization = match self.polarization_factor {
                Some(factor) => temp_geometry.calc_polarization(factor, &state.tth, &state.azimuth),
                None => vec![1.0; size],
            };

            // Update normalization: pol * sa (pyFAI-style)
            state.normalization = Arc::new(
                (0..size)
                    .map(|i| state.polarization[i] * state.solid_angle[i])
                    .collect(),
            );
            cache_1d.key = None;
            cache_1d.lut = None;
            cache_2d.key = None;
            cache_2d.lut = None;
        }
    }

    pub(super) fn update_polarization(&self) {
        let mut state = self.state.write().expect("geometry state lock poisoned");
        let size = state.dim1 * state.dim2;
        if size == 0 {
            return;
        }

        // Need geometry with correct dimensions for calc_polarization
        let mut temp_geometry = self.geometry.clone();
        temp_geometry.dim1 = state.dim1;
        temp_geometry.dim2 = state.dim2;

        state.polarization = match self.polarization_factor {
            Some(factor) => temp_geometry.calc_polarization(factor, &state.tth, &state.azimuth),
            None => vec![1.0; size],
        };

        // Update normalization when polarization changes
        if !state.solid_angle.is_empty() {
            state.normalization = Arc::new(
                (0..size)
                    .map(|i| state.polarization[i] * state.solid_angle[i])
                    .collect(),
            );
        }
        drop(state);
        self.invalidate_normalization();
    }

    /// Get the normalization array (pol * sa) for weighted-average integration.
    /// Thread-safe: only reads from state. Returns an `Arc` clone so callers
    /// share the underlying buffer without copying it (cheap atomic refcount bump).
    pub(super) fn get_normalization(&self) -> Arc<Vec<f64>> {
        let state = self.state.read().expect("geometry state lock poisoned");
        Arc::clone(&state.normalization)
    }
}

#[cfg(test)]
mod concurrent_tests {
    use super::*;
    use crate::{frame::Frame, geometry::Geometry, integration::IntegrationMethod, poni::Poni};
    use std::sync::Barrier;

    #[test]
    fn late_geometry_initializer_keeps_valid_tables() {
        let mut poni = Poni::new();
        poni.pixel1 = 1.0;
        poni.pixel2 = 1.0;
        poni.distance = 10.0;
        let integrator =
            Integrator::new(Geometry::new(2, 2, poni)).with_method(IntegrationMethod::Counting);
        let frame = Frame {
            name: "late initializer".into(),
            dim1: 2,
            dim2: 2,
            data: vec![1.0; 4],
        };
        integrator.integrate1d(&frame, 4);
        integrator.integrate2d(&frame, 4, 2);
        // Model a worker that observed cold state before another worker built
        // the geometry and both LUTs, then reached the write-side double check.
        integrator.refresh_geometry_arrays(2, 2);
        assert!(integrator.cache_1d.read().unwrap().lut.is_some());
        assert!(integrator.cache_2d.read().unwrap().lut.is_some());
    }

    #[test]
    fn cold_geometry_initialization_preserves_concurrent_luts() {
        let mut poni = Poni::new();
        poni.pixel1 = 1.0;
        poni.pixel2 = 1.0;
        poni.distance = 10.0;
        let frame = Frame {
            name: "cold concurrent frame".into(),
            dim1: 7,
            dim2: 9,
            data: (0..63).map(|i| (i + 1) as f64).collect(),
        };
        let reference = Integrator::new(Geometry::new(2, 2, poni.clone()))
            .with_method(IntegrationMethod::Counting);
        let expected_1d = reference.integrate1d(&frame, 8);
        let expected_2d = reference.integrate2d(&frame, 8, 4);
        for _ in 0..16 {
            let integrator = Integrator::new(Geometry::new(2, 2, poni.clone()))
                .with_method(IntegrationMethod::Counting);
            let start = Barrier::new(8);
            std::thread::scope(|scope| {
                for _ in 0..8 {
                    scope.spawn(|| {
                        start.wait();
                        assert_eq!(integrator.integrate1d(&frame, 8), expected_1d);
                        assert_eq!(integrator.integrate2d(&frame, 8, 4), expected_2d);
                    });
                }
            });
        }
    }
}
