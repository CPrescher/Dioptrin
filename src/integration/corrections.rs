use super::Integrator;

impl Integrator {
    /// checks whether current tth, chi, solid angle and polarization are calculated for the same
    /// image dimension as dim1 x dim2, if not, they are recalculated
    pub(super) fn update_geometry_arrays(&mut self, dim1: usize, dim2: usize) {
        let size = dim1 * dim2;
        let stale = self.geometry.dim1 != dim1
            || self.geometry.dim2 != dim2
            || self.tth.len() != size
            || self.chi.len() != size
            || self.solid_angle.len() != size
            || self.polarization.len() != size;
        if stale {
            self.geometry.dim1 = dim1;
            self.geometry.dim2 = dim2;
            self.tth = self.geometry.calc_tth_all();
            self.chi = self.geometry.calc_chi_all();
            self.solid_angle = self.geometry.calc_solid_angle();
            self.update_polarization();
            let mut cache_1d = self.cache_1d.borrow_mut();
            cache_1d.key = None;
            cache_1d.lut = None;
            drop(cache_1d);
            let mut cache_2d = self.cache_2d.borrow_mut();
            cache_2d.key = None;
            cache_2d.lut = None;
        }
    }

    pub(super) fn update_polarization(&mut self) {
        self.polarization = match self.polarization_factor {
            Some(factor) => self
                .geometry
                .calc_polarization(factor, &self.tth, &self.chi),
            None => vec![1.0; self.geometry.size()],
        };
    }

    /// Apply polarization and solid angle corrections to the intensity.
    pub(super) fn correct_intensity(&mut self, data: &Vec<f64>) -> Vec<f64> {
        data.iter()
            .enumerate()
            .map(|(i, y)| y / self.polarization[i] / self.solid_angle[i])
            .collect::<Vec<f64>>()
    }

    pub(super) fn correct_intensity_and_variance(&mut self, data: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        let mut intensity = Vec::with_capacity(data.len());
        let mut variance = Vec::with_capacity(data.len());
        for (i, &y) in data.iter().enumerate() {
            let corr = self.polarization[i] * self.solid_angle[i];
            let inv = 1.0 / corr;
            intensity.push(y * inv);
            variance.push(y * inv * inv);
        }
        (intensity, variance)
    }
}
