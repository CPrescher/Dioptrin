use super::Integrator;

impl Integrator {
    /// checks whether current tth, radial, azimuth, solid angle and polarization are calculated for the same
    /// image dimension as dim1 x dim2, if not, they are recalculated
    pub(super) fn update_geometry_arrays(&mut self, dim1: usize, dim2: usize) {
        let size = dim1 * dim2;
        // Optimized: check cached size first (single comparison), then dimensions
        // If size matches and dimensions match, all arrays should be correct
        let stale = self.cached_size != size
            || self.geometry.dim1 != dim1
            || self.geometry.dim2 != dim2;
        if stale {
            self.geometry.dim1 = dim1;
            self.geometry.dim2 = dim2;
            self.cached_size = size;
            self.tth = self.geometry.calc_tth_all();
            self.radial = self
                .radial_unit
                .convert_from_tth(&self.tth, self.geometry.poni.wavelength);
            self.azimuth = self.geometry.calc_azimuth_all();
            self.solid_angle = self.geometry.calc_solid_angle();
            self.update_polarization(); // This will also call update_correction_factor()
            self.invalidate_caches();
        }
    }

    pub(super) fn update_polarization(&mut self) {
        self.polarization = match self.polarization_factor {
            Some(factor) => self
                .geometry
                .calc_polarization(factor, &self.tth, &self.azimuth),
            None => vec![1.0; self.geometry.size()],
        };
        // Update correction factor when polarization changes
        // Only update if arrays are already initialized (size > 0)
        if !self.solid_angle.is_empty() {
            self.update_correction_factor();
        }
    }

    /// Update the combined correction factor: 1.0 / (polarization * solid_angle)
    /// This avoids two divisions per pixel in correct_intensity()
    pub(super) fn update_correction_factor(&mut self) {
        let size = self.geometry.size();
        self.correction_factor = (0..size)
            .map(|i| 1.0 / (self.polarization[i] * self.solid_angle[i]))
            .collect();
    }

    /// Apply polarization and solid angle corrections to the intensity.
    /// Uses pre-computed correction_factor to avoid two divisions per pixel.
    pub(super) fn correct_intensity(&mut self, data: &[f64]) -> Vec<f64> {
        data.iter()
            .enumerate()
            .map(|(i, y)| y * self.correction_factor[i])
            .collect::<Vec<f64>>()
    }

    pub(super) fn correct_intensity_and_variance(
        &mut self,
        data: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let mut intensity = Vec::with_capacity(data.len());
        let mut variance = Vec::with_capacity(data.len());
        // Use pre-computed correction_factor to avoid recalculating 1/(polarization * solid_angle)
        for (i, &y) in data.iter().enumerate() {
            let corr = self.correction_factor[i];
            intensity.push(y * corr);
            variance.push(y * corr * corr);
        }
        (intensity, variance)
    }
}
