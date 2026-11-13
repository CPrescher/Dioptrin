//! Broad CSC/CSR estimate for complete resident 1D frames.
use super::SparseLayout;

/// Approximate layout recommendation, independent of the splitting method.
#[derive(Clone, Copy, Debug)]
pub struct LayoutPrediction {
    pub layout: SparseLayout,
    /// Approximate size above which the CSC exceptions apply.
    pub large_image_pixels: usize,
}

/// Broad estimate from the 2026-11-09 M4 Pro detector/bin sweep. Prefer CSR
/// generally. At about 8 MP and above, CSC is preferred for 2000 bins or more
/// without Poisson errors. This is not a precise portable threshold.
/// Complete parallel batches retain CSC. Thread count supplies context only;
/// the estimate has not been calibrated for every CPU/thread count.
pub fn predict_1d_layout(
    pixels: usize,
    bins: usize,
    errors: bool,
    parallel_frames: bool,
    threads: usize,
) -> LayoutPrediction {
    let large_image_pixels = 8_000_000;
    let large_csc = pixels >= large_image_pixels && !errors && bins >= 2000;
    let layout = if pixels == 0 || bins == 0 || threads == 0 || parallel_frames || large_csc {
        SparseLayout::Csc
    } else {
        SparseLayout::Csr
    };
    LayoutPrediction {
        layout,
        large_image_pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn broad_estimate_retains_large_detectors_and_parallel_csc() {
        for bins in [128, 256, 512, 1000, 2000, 4000, 8000] {
            for errors in [false, true] {
                let expected = if !errors && bins >= 2000 {
                    SparseLayout::Csc
                } else {
                    SparseLayout::Csr
                };
                assert_eq!(
                    predict_1d_layout(10_138_296, bins, errors, false, 8).layout,
                    expected
                );
                assert_eq!(
                    predict_1d_layout(1_023_183, bins, errors, false, 8).layout,
                    SparseLayout::Csr
                );
                assert_eq!(
                    predict_1d_layout(1_023_183, bins, errors, true, 8).layout,
                    SparseLayout::Csc
                );
            }
        }
        assert_eq!(
            predict_1d_layout(5_069_148, 128, false, false, 8).layout,
            SparseLayout::Csr
        );
        assert_eq!(
            predict_1d_layout(5_069_148, 1000, false, false, 8).layout,
            SparseLayout::Csr
        );
        for (p, b, t) in [(0, 1000, 8), (1_000_000, 0, 8), (1_000_000, 1000, 0)] {
            assert_eq!(
                predict_1d_layout(p, b, false, false, t).layout,
                SparseLayout::Csc
            );
        }
    }
}
