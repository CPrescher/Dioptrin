//! Bounded performance prior for resident fixed-mask polygon patterns.
use super::SparseLayout;

/// A timing recommendation, independent of the scientific splitting method.
#[derive(Clone, Copy, Debug)]
pub struct LayoutPrediction {
    /// Recommended orientation; small/unsupported differences retain CSC.
    pub layout: SparseLayout,
    /// Predicted CSC time divided by CSR time, if within the measured domain.
    pub csc_over_csr: Option<f64>,
}

/// Empirical prior fitted to four real detector files and two software-binned
/// derivatives on an M4 Pro with eight Rayon threads (2026-11-09). This is not
/// a portable crossover law. Geometry, cache state and mask topology matter;
/// local timing should confirm a switch. Unsupported inputs retain CSC.
pub fn predict_1d_layout(
    pixels: usize,
    bins: usize,
    errors: bool,
    parallel_frames: bool,
    threads: usize,
) -> LayoutPrediction {
    if threads != 8 || !(287_625..=10_138_296).contains(&pixels) || !(128..=8000).contains(&bins) {
        return LayoutPrediction {
            layout: SparseLayout::Csc,
            csc_over_csr: None,
        };
    }
    let x = (pixels as f64 / 1e6).log2();
    let z = (bins as f64 / 1000.).log2();
    let e = f64::from(errors);
    let b = f64::from(parallel_frames);
    // ln(CSC_ms / CSR_ms), ordinary least squares over 152 distinct grids.
    // Repeated grids are pooled, not treated as independent detector samples.
    let features = [1., x, z, z * z, x * z, e, e * x, e * z, b, b * x, b * z];
    let coefficients = [
        0.9625023218279644,
        -0.32014207619562346,
        -0.025054288861904386,
        -0.012208761830162198,
        0.009420331475340826,
        0.17262911494014688,
        0.0010245923933961238,
        0.031075927627037103,
        -0.6220661764539083,
        0.19181084719728966,
        0.0007575524029175193,
    ];
    let ratio = features
        .iter()
        .zip(coefficients)
        .map(|(x, c)| x * c)
        .sum::<f64>()
        .exp();
    LayoutPrediction {
        // A 30% guard band makes this a conservative prior, not a mandatory switch.
        layout: if ratio > 1.3 {
            SparseLayout::Csr
        } else {
            SparseLayout::Csc
        },
        csc_over_csr: Some(ratio),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prior_is_bounded_and_retains_large_low_bin_csc_baseline() {
        assert_eq!(
            predict_1d_layout(10_138_296, 128, false, false, 8).layout,
            SparseLayout::Csc
        );
        assert_eq!(
            predict_1d_layout(1_023_183, 1000, false, false, 8).layout,
            SparseLayout::Csr
        );
        for (p, b, t) in [
            (0, 1000, 8),
            (1_000_000, 0, 8),
            (20_000_000, 1000, 8),
            (1_000_000, 1000, 4),
        ] {
            let r = predict_1d_layout(p, b, false, false, t);
            assert_eq!(r.layout, SparseLayout::Csc);
            assert!(r.csc_over_csr.is_none());
        }
    }
}
