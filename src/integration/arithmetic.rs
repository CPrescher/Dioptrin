//! Dispatch private prepared kernels without changing the public LUT API.
use super::Integrator;
use crate::lut::prepared::{Prepared1d, Prepared2d};
use crate::lut::{IntermediateResult1d, IntermediateResult2d};

macro_rules! dispatch {
    ($name:ident, $kernel:ident, $var:literal, $lut:ty, $result:ty) => {
        impl Integrator {
            pub(super) fn $name(
                &self,
                s: &[f64],
                l: &$lut,
                m: Option<&[u8]>,
                n: &[f64],
            ) -> $result {
                if self.fast_math {
                    l.$kernel::<true, $var>(s, m, n)
                } else {
                    l.$kernel::<false, $var>(s, m, n)
                }
            }
        }
    };
}
// Small adapters discard unrequested outputs without changing calculation order.
impl Integrator {
    pub(super) fn lut_integrate1d(
        &self,
        s: &[f64],
        l: &Prepared1d,
        m: Option<&[u8]>,
        n: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let (x, y, _) = if self.fast_math {
            l.integrate::<true, false>(s, m, n)
        } else {
            l.integrate::<false, false>(s, m, n)
        };
        (x, y)
    }
    pub(super) fn lut_integrate2d(
        &self,
        s: &[f64],
        l: &Prepared2d,
        m: Option<&[u8]>,
        n: &[f64],
    ) -> Vec<f64> {
        if self.fast_math {
            l.integrate::<true, false>(s, m, n).0
        } else {
            l.integrate::<false, false>(s, m, n).0
        }
    }
}
dispatch!(
    lut_integrate1d_with_errors,
    integrate,
    true,
    Prepared1d,
    (Vec<f64>, Vec<f64>, Vec<f64>)
);
dispatch!(
    lut_integrate2d_with_errors,
    integrate,
    true,
    Prepared2d,
    (Vec<f64>, Vec<f64>)
);
dispatch!(
    lut_integrate1d_raw,
    raw1d,
    false,
    Prepared1d,
    IntermediateResult1d
);
dispatch!(
    lut_integrate1d_raw_with_variance,
    raw1d,
    true,
    Prepared1d,
    IntermediateResult1d
);
dispatch!(
    lut_integrate2d_raw,
    raw2d,
    false,
    Prepared2d,
    IntermediateResult2d
);
dispatch!(
    lut_integrate2d_raw_with_variance,
    raw2d,
    true,
    Prepared2d,
    IntermediateResult2d
);
