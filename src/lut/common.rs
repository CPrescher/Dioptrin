/// Creates a d1, d2 subgrid for a column in a dim1 x dim2 grid
///
/// Arguments:
///   * `row` - column index for the sub grid
///   * `dim2` - number of points in horizontal direction (-)
///   * `n_ss` - supersampling factor -> number of subpixels in each direction (e.g. 2 - will result in 2x2 pixel per original point)
#[cfg(test)]
pub(crate) fn subgrid(row: usize, dim2: usize, n_ss: usize) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let size = dim2 * n_ss * n_ss;
    let mut d1 = Vec::with_capacity(size);
    let mut d2 = Vec::with_capacity(size);
    let mut actual_index = Vec::with_capacity(size);

    let dim_sub = dim2 * n_ss;

    for i in 0..n_ss {
        for j in 0..dim_sub {
            d1.push(row as f64 + (i as f64 + 0.5) / n_ss as f64);
            d2.push((j as f64 + 0.5) / n_ss as f64);
            actual_index.push(row * dim2 + j / n_ss);
        }
    }

    (d1, d2, actual_index)
}

/// Creates a grid of d1, d2 coordinates for the hull of dim1 x dim2 grid (i.e. the outermost points)
///
/// Arguments:
/// * `dim1` - number of points in vertical direction (-)
/// * `dim2` - number of points in horizontal direction (-)
/// * `n_ss` - supersampling factor -> number of subpixels in each direction (e.g. 2 - will result in 2x2 pixel per original point)
pub fn hull_grid(dim1: usize, dim2: usize, n_ss: usize) -> (Vec<f64>, Vec<f64>) {
    let size = 2 * (dim1 + dim2 - 2) * n_ss;
    let mut d1 = Vec::with_capacity(size);
    let mut d2 = Vec::with_capacity(size);

    let border_offset = 0.5 / n_ss as f64;
    let step = 1.0 / n_ss as f64;

    // left
    for i in 0..dim1 * n_ss {
        d1.push(border_offset + i as f64 * step);
        d2.push(border_offset);
    }

    // right
    for i in 0..dim1 * n_ss {
        d1.push(border_offset + i as f64 * step);
        d2.push(dim2 as f64 - border_offset);
    }

    // bottom
    for i in 1..(dim2 * n_ss) {
        d1.push(border_offset);
        d2.push(border_offset + i as f64 * step);
    }

    // top
    for i in 1..(dim2 * n_ss) {
        d1.push(dim1 as f64 - border_offset);
        d2.push(border_offset + i as f64 * step);
    }
    (d1, d2)
}
