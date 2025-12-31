#![allow(dead_code)]

#[cfg(feature = "lut_metrics")]
use std::collections::HashSet;

#[cfg(feature = "lut_metrics")]
use std::collections::HashMap as StdHashMap;

#[cfg(feature = "lut_metrics")]
#[derive(Debug, Clone)]
pub struct LutLocalityStats {
    pub empty_bins: usize,
    pub total_bins: usize,
    pub avg_stride: f64,
    pub avg_cache_lines: f64,
}

#[cfg(feature = "lut_metrics")]
fn analyze_slices<'a, I>(slices: I) -> LutLocalityStats
where
    I: IntoIterator<Item = &'a [usize]>,
{
    let mut total_bins = 0usize;
    let mut empty_bins = 0usize;
    let mut stride_sum: f64 = 0.0;
    let mut stride_transitions: usize = 0;
    let mut cache_line_sum: f64 = 0.0;
    let mut nonempty_bins: usize = 0;

    for slice in slices {
        total_bins += 1;
        if slice.is_empty() {
            empty_bins += 1;
            continue;
        }
        nonempty_bins += 1;

        if slice.len() > 1 {
            for w in slice.windows(2) {
                let a = w[0];
                let b = w[1];
                stride_sum += (a as isize - b as isize).unsigned_abs() as f64;
                stride_transitions += 1;
            }
        }

        let mut set: HashSet<usize> = HashSet::with_capacity(slice.len());
        for &idx in slice {
            set.insert(idx / 8);
        }
        cache_line_sum += set.len() as f64;
    }

    LutLocalityStats {
        empty_bins,
        total_bins,
        avg_stride: if stride_transitions > 0 {
            stride_sum / stride_transitions as f64
        } else {
            0.0
        },
        avg_cache_lines: if nonempty_bins > 0 {
            cache_line_sum / nonempty_bins as f64
        } else {
            0.0
        },
    }
}

// For Vec<Vec<usize>> LUTs
#[cfg(feature = "lut_metrics")]
pub fn analyze_lut_locality(lut: &Vec<Vec<usize>>) -> LutLocalityStats {
    analyze_slices(lut.iter().map(|v| v.as_slice()))
}

// For Vec<Vec<Vec<usize>>>> 2D LUTs
#[cfg(feature = "lut_metrics")]
pub fn analyze_lut2d_locality(lut: &Vec<Vec<Vec<usize>>>) -> LutLocalityStats {
    analyze_slices(lut.iter().flat_map(|row| row.iter().map(|v| v.as_slice())))
}

// For supersampled HashMap-based LUTs
#[cfg(feature = "lut_metrics")]
pub fn analyze_lut_ss_locality(lut: &Vec<StdHashMap<usize, f64>>) -> LutLocalityStats {
    let tmp: Vec<Vec<usize>> = lut
        .iter()
        .map(|h| {
            let mut v: Vec<usize> = h.keys().copied().collect();
            v.sort_unstable();
            v
        })
        .collect();
    analyze_slices(tmp.iter().map(|v| v.as_slice()))
}

// For supersampled 2D HashMap-based LUTs
#[cfg(feature = "lut_metrics")]
pub fn analyze_lut2d_ss_locality(lut: &Vec<Vec<StdHashMap<usize, f64>>>) -> LutLocalityStats {
    let tmp: Vec<Vec<usize>> = lut
        .iter()
        .flat_map(|row| {
            row.iter().map(|h| {
                let mut v: Vec<usize> = h.keys().copied().collect();
                v.sort_unstable();
                v
            })
        })
        .collect();
    analyze_slices(tmp.iter().map(|v| v.as_slice()))
}

// For CSR (col_idx + row_ptr)
#[cfg(feature = "lut_metrics")]
pub fn analyze_csr_locality(col_idx: &[usize], row_ptr: &[usize]) -> LutLocalityStats {
    let mut total_bins = 0usize;
    let mut empty_bins = 0usize;
    let mut stride_sum: f64 = 0.0;
    let mut stride_transitions: usize = 0;
    let mut cache_line_sum: f64 = 0.0;
    let mut nonempty_bins: usize = 0;

    for i in 0..row_ptr.len() - 1 {
        total_bins += 1;
        let start = row_ptr[i];
        let end = row_ptr[i + 1];
        let slice = &col_idx[start..end];
        if slice.is_empty() {
            empty_bins += 1;
            continue;
        }
        nonempty_bins += 1;
        if slice.len() > 1 {
            for w in slice.windows(2) {
                let a = w[0];
                let b = w[1];
                stride_sum += (a as isize - b as isize).unsigned_abs() as f64;
                stride_transitions += 1;
            }
        }
        let mut set: HashSet<usize> = HashSet::with_capacity(slice.len());
        for &idx in slice {
            set.insert(idx / 8);
        }
        cache_line_sum += set.len() as f64;
    }

    LutLocalityStats {
        empty_bins,
        total_bins,
        avg_stride: if stride_transitions > 0 {
            stride_sum / stride_transitions as f64
        } else {
            0.0
        },
        avg_cache_lines: if nonempty_bins > 0 {
            cache_line_sum / nonempty_bins as f64
        } else {
            0.0
        },
    }
}

