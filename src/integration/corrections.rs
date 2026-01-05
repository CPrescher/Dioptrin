use super::Integrator;

impl Integrator {
    /// Checks whether current tth, radial, azimuth, solid angle and polarization are calculated for the same
    /// image dimension as dim1 x dim2, if not, they are recalculated.
    /// Thread-safe: uses RwLock for interior mutability.
    pub(super) fn update_geometry_arrays(&self, dim1: usize, dim2: usize) {
        let size = dim1 * dim2;

        // First check with read lock if update is needed
        let needs_update = {
            let state = self.state.read().unwrap();
            let stale = (state.dim1 * state.dim2) != size
                || state.dim1 != dim1
                || state.dim2 != dim2;
            stale
        };

        if needs_update {
            // Acquire write lock and double-check (another thread may have updated)
            let mut state = self.state.write().unwrap();
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

                // Update correction factor
                state.correction_factor = (0..size)
                    .map(|i| 1.0 / (state.polarization[i] * state.solid_angle[i]))
                    .collect();
            }
            drop(state);
            self.invalidate_caches();
        }
    }

    pub(super) fn update_polarization(&self) {
        let mut state = self.state.write().unwrap();
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

        // Update correction factor when polarization changes
        if !state.solid_angle.is_empty() {
            state.correction_factor = (0..size)
                .map(|i| 1.0 / (state.polarization[i] * state.solid_angle[i]))
                .collect();
        }
    }

    /// Apply polarization and solid angle corrections to the intensity.
    /// Uses pre-computed correction_factor to avoid two divisions per pixel.
    /// Thread-safe: only reads from state.
    pub(super) fn correct_intensity(&self, data: &[f64]) -> Vec<f64> {
        let state = self.state.read().unwrap();
        data.iter()
            .enumerate()
            .map(|(i, y)| y * state.correction_factor[i])
            .collect::<Vec<f64>>()
    }

    pub(super) fn correct_intensity_and_variance(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let state = self.state.read().unwrap();
        let mut intensity = Vec::with_capacity(data.len());
        let mut variance = Vec::with_capacity(data.len());
        for (i, &y) in data.iter().enumerate() {
            let corr = state.correction_factor[i];
            intensity.push(y * corr);
            variance.push(y * corr * corr);
        }
        (intensity, variance)
    }
}
