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
            // Acquire write lock and double-check (another thread may have updated)
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
                    Some(factor) => {
                        temp_geometry.calc_polarization(factor, &state.tth, &state.azimuth)
                    }
                    None => vec![1.0; size],
                };

                // Update normalization: pol * sa (pyFAI-style)
                state.normalization = (0..size)
                    .map(|i| state.polarization[i] * state.solid_angle[i])
                    .collect();
            }
            drop(state);
            self.invalidate_caches();
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
            state.normalization = (0..size)
                .map(|i| state.polarization[i] * state.solid_angle[i])
                .collect();
        }
    }

    /// Get the normalization array (pol * sa) for weighted-average integration.
    /// Thread-safe: only reads from state.
    pub(super) fn get_normalization(&self) -> Vec<f64> {
        let state = self.state.read().expect("geometry state lock poisoned");
        state.normalization.clone()
    }
}
