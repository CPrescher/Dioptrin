use std::sync::Arc;

use super::{GeometryState, Integrator};

impl Integrator {
    /// Returns tth, radial, azimuth, solid angle, polarization and
    /// normalization for a `dim1 x dim2` image, computing them if the shared
    /// state holds another shape.
    ///
    /// The result is an immutable snapshot, so it stays valid for this shape
    /// whatever other threads do. No lock is held while the arrays are computed
    /// (that uses Rayon, and a lock held across Rayon work can deadlock when
    /// the waiting thread steals another frame of the same batch). Threads
    /// that race on a cold or different-shape state may each compute the
    /// arrays once; the values are identical and the first one published wins.
    pub(super) fn geometry_for(&self, dim1: usize, dim2: usize) -> Arc<GeometryState> {
        {
            let state = self.state.read().expect("geometry state lock poisoned");
            if state.dim1 == dim1 && state.dim2 == dim2 {
                return Arc::clone(&state);
            }
        }
        let computed = Arc::new(self.compute_geometry(dim1, dim2));
        let mut state = self.state.write().expect("geometry state lock poisoned");
        if state.dim1 != dim1 || state.dim2 != dim2 {
            *state = computed;
        }
        Arc::clone(&state)
    }

    fn compute_geometry(&self, dim1: usize, dim2: usize) -> GeometryState {
        let size = dim1 * dim2;
        let mut temp_geometry = self.geometry.clone();
        temp_geometry.dim1 = dim1;
        temp_geometry.dim2 = dim2;

        let tth = temp_geometry.calc_tth_all();
        let radial = self
            .radial_unit
            .convert_from_tth(&tth, self.geometry.poni.wavelength);
        let azimuth = temp_geometry.calc_azimuth_all();
        let solid_angle = temp_geometry.calc_solid_angle();
        let polarization = match self.polarization_factor {
            Some(factor) => temp_geometry.calc_polarization(factor, &tth, &azimuth),
            None => vec![1.0; size],
        };
        // Normalization: pol * sa (pyFAI-style)
        let normalization = Arc::new(
            (0..size)
                .map(|i| polarization[i] * solid_angle[i])
                .collect(),
        );
        GeometryState {
            dim1,
            dim2,
            tth,
            radial,
            azimuth,
            solid_angle,
            polarization,
            normalization,
        }
    }

    /// Normalization of the most recently used image shape. Integration code
    /// must take it from [`Self::for_shape`] instead, which cannot be switched
    /// to another shape by a concurrent call.
    #[cfg(test)]
    pub(super) fn get_normalization(&self) -> Arc<Vec<f64>> {
        let state = self.state.read().expect("geometry state lock poisoned");
        Arc::clone(&state.normalization)
    }

    pub(super) fn update_polarization(&self) {
        let mut guard = self.state.write().expect("geometry state lock poisoned");
        let state = Arc::make_mut(&mut guard);
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
        drop(guard);
        self.invalidate_normalization();
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
        // A worker that observed cold state before another worker built the
        // geometry and both LUTs must reuse them, not clear them.
        let state = integrator.geometry_for(2, 2);
        assert!(Arc::ptr_eq(&state, &integrator.geometry_for(2, 2)));
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
