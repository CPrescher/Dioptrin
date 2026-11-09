//! Fit2D / PONI interoperability. Distances mm, pixels µm, angles degrees.
use crate::poni::Poni;
#[derive(Clone, Copy, Debug)]
pub struct Fit2d {
    pub direct_dist: f64,
    pub center_x: f64,
    pub center_y: f64,
    pub tilt: f64,
    pub tilt_plane_rotation: f64,
    pub pixel_x: f64,
    pub pixel_y: f64,
}
impl Fit2d {
    pub fn from_poni(p: &Poni) -> Self {
        let ct = p.rot1.cos() * p.rot2.cos();
        let st = (1. - ct * ct).max(0.).sqrt();
        let tt = st / ct;
        let (ctp, stp) = if st == 0. {
            (1., 0.)
        } else {
            (
                (-p.rot2.cos() * p.rot1.sin() / st).clamp(-1., 1.),
                p.rot2.sin() / st,
            )
        };
        let tilt = ct.clamp(-1., 1.).acos().to_degrees();
        Self {
            direct_dist: 1000. * p.distance / ct,
            center_x: (p.poni2 + p.distance * tt * ctp) / p.pixel2,
            center_y: if tilt.abs() < 1e-5 {
                p.poni1 / p.pixel1
            } else {
                (p.poni1 + p.distance * tt * stp) / p.pixel1
            },
            tilt,
            tilt_plane_rotation: ctp.acos().to_degrees() * if stp < 0. { -1. } else { 1. },
            pixel_x: p.pixel2 * 1e6,
            pixel_y: p.pixel1 * 1e6,
        }
    }
    /// Wavelength stays in SI; Fit2D does not determine it from geometry.
    pub fn to_poni(self, wavelength: f64) -> Result<Poni, String> {
        if [
            self.direct_dist,
            self.center_x,
            self.center_y,
            self.tilt,
            self.tilt_plane_rotation,
            self.pixel_x,
            self.pixel_y,
            wavelength,
        ]
        .iter()
        .any(|v| !v.is_finite())
            || self.direct_dist <= 0.
            || self.pixel_x <= 0.
            || self.pixel_y <= 0.
            || wavelength <= 0.
            || self.tilt.abs() >= 90.
        {
            return Err("Fit2D needs positive distance, pixels and wavelength, and tilt between -90° and 90°.".into());
        }
        let (st, ct) = self.tilt.to_radians().sin_cos();
        let (stp, ctp) = self.tilt_plane_rotation.to_radians().sin_cos();
        let den = (1. - (stp * st).powi(2)).sqrt();
        let mut rot1 = (ct / den).clamp(-1., 1.).acos();
        if ctp * st > 0. {
            rot1 = -rot1;
        }
        Ok(Poni {
            pixel1: self.pixel_y * 1e-6,
            pixel2: self.pixel_x * 1e-6,
            distance: self.direct_dist * ct * 1e-3,
            poni1: self.center_y * self.pixel_y * 1e-6 - self.direct_dist * st * stp * 1e-3,
            poni2: self.center_x * self.pixel_x * 1e-6 - self.direct_dist * st * ctp * 1e-3,
            rot1,
            rot2: (st * stp).clamp(-1., 1.).asin(),
            rot3: if self.tilt == 0. {
                0.
            } else {
                std::f64::consts::FRAC_PI_2
                    - ((ct * ctp * stp - ctp * stp) / den).clamp(-1., 1.).acos()
            },
            wavelength,
            spline_file: String::new(),
        })
    }
}
