//! Broad CSC/CSR estimate for complete resident 1D frames.
use super::SparseLayout;

/// Approximate layout recommendation, independent of the splitting method.
#[derive(Clone, Copy, Debug)]
pub struct LayoutPrediction {
    pub layout: SparseLayout,
    /// Approximate pixel count above which CSC is preferred.
    pub switch_pixels: usize,
}

/// Ballpark from the 2026-11-09 M4 Pro detector/bin sweep, not a fitted timing
/// guarantee. Errors favour CSR until about 8 MP; intensity-only crosses around
/// 6 MP for 512–2000 bins and about 4 MP outside that range. Complete parallel
/// batches retain CSC. Thread count is accepted for context/invalid-input checks;
/// these approximate thresholds are not calibrated for every CPU/thread count.
pub fn predict_1d_layout(
    pixels: usize,
    bins: usize,
    errors: bool,
    parallel_frames: bool,
    threads: usize,
) -> LayoutPrediction {
    let switch_pixels = if errors {
        8_000_000
    } else if (512..=2000).contains(&bins) {
        6_000_000
    } else {
        4_000_000
    };
    let layout =
        if pixels == 0 || bins == 0 || threads == 0 || parallel_frames || pixels >= switch_pixels {
            SparseLayout::Csc
        } else {
            SparseLayout::Csr
        };
    LayoutPrediction {
        layout,
        switch_pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn broad_estimate_retains_large_detectors_and_parallel_csc() {
        for bins in [128, 256, 512, 1000, 2000, 4000, 8000] {
            for errors in [false, true] {
                assert_eq!(
                    predict_1d_layout(10_138_296, bins, errors, false, 8).layout,
                    SparseLayout::Csc
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
            SparseLayout::Csc
        );
        assert_eq!(
            predict_1d_layout(5_069_148, 1000, false, false, 8).layout,
            SparseLayout::Csr
        );
        assert_eq!(
            predict_1d_layout(5_069_148, 128, true, false, 8).layout,
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
