use std::fmt::Display;
use std::ops::Mul;

use rayon::prelude::*;

use crate::math::matrix::Matrix;
use crate::math::max;
use crate::math::sparse::csr::CsrMatrix;

/// A CooMatrix stores a sparse matrix as three vectors: one for the row indices, one for the column indices and one for the values.
/// The row and column indices are stored in the same order as the values. The CooMatrix row and column indices are not sorted by default.
/// But it can be sorted using the sort() method.
/// The CooMatrix can be converted to a CSR matrix using the to_csr() method.
/// The CooMatrix can be converted to a dense matrix using the to_dense() method.
/// This implementation is not optimized for performance (e.g. multiplication), it is rather intended to be used for small matrices
/// and sparse matrix creation and manipulation. For random access patterns it might be also suitable for computations.
///
/// Typical usage:
/// ```
/// use dioptrin::math::sparse::coo::CooMatrix;
///
/// let mut matrix = CooMatrix::new(vec![0, 2, 2], vec![0, 0, 1], vec![1.0, 3.0, 5.0]);
///
/// matrix.add_or_insert(0, 0, 1.0); // adds a value to an existing element
/// matrix.add_or_insert(1, 0, 1.0); // adds a new element
/// matrix.set(2, 2, 1.0); // inserts a new element
/// matrix.set(3, 100, 20.0); // inserts a new element
///```
pub struct CooMatrix {
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
}

impl CooMatrix {
    pub fn new(row_indices: Vec<usize>, col_indices: Vec<usize>, values: Vec<f64>) -> CooMatrix {
        CooMatrix {
            row_indices,
            col_indices,
            values,
        }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        let mut value = 0.0;
        for (i, x_i) in self.row_indices.iter().enumerate() {
            if *x_i == x && self.col_indices[i] == y {
                value = self.values[i];
            }
        }
        value
    }

    pub fn set(&mut self, x: usize, y: usize, value: f64) {
        for (i, x_i) in self.row_indices.iter().enumerate() {
            if *x_i == x && self.col_indices[i] == y {
                self.values[i] = value;
            }
        }
    }

    pub fn add(&mut self, x: usize, y: usize, value: f64) {
        for (i, x_i) in self.row_indices.iter().enumerate() {
            if *x_i == x && self.col_indices[i] == y {
                self.values[i] += value;
            }
        }
    }

    pub fn add_or_insert(&mut self, x: usize, y: usize, value: f64) {
        let mut found = false;
        for (i, x_i) in self.row_indices.iter().enumerate() {
            if *x_i == x && self.col_indices[i] == y {
                self.values[i] += value;
                found = true;
            }
        }
        if !found {
            self.insert(x, y, value)
        }
    }

    pub fn exists(&self, x: usize, y: usize) -> bool {
        let mut exists = false;
        for (i, x_i) in self.row_indices.iter().enumerate() {
            if *x_i == x && self.col_indices[i] == y {
                exists = true;
            }
        }
        exists
    }

    pub fn to_csr(&mut self) -> CsrMatrix {
        let sorted_matrix = self.sort();

        let m = max(&sorted_matrix.row_indices) + 1; // find the maximum row index to get the number of rows
        let mut csr_row_pointer = vec![0; m + 1];

        // count the number of times each row appears in the row_indices
        for i in 0..sorted_matrix.values.len() {
            csr_row_pointer[sorted_matrix.row_indices[i] + 1] += 1;
        }

        // sum up the counts by adding the count for the previous row to get the row pointer
        for i in 1..csr_row_pointer.len() {
            csr_row_pointer[i] += csr_row_pointer[i - 1];
        }

        CsrMatrix::new(
            sorted_matrix.col_indices,
            csr_row_pointer,
            sorted_matrix.values,
        )
    }

    pub fn to_dense(&self, m: usize, n: usize) -> Matrix {
        let mut values = vec![0.0; m * n];
        for (i, (row, col)) in self.row_indices.iter().zip(&self.col_indices).enumerate() {
            values[row * n + col] = self.values[i];
        }
        Matrix::new(m, n, values)
    }

    pub fn sort(&self) -> CooMatrix {
        let mut indices = (0..self.row_indices.len()).collect::<Vec<usize>>();
        indices.sort_by(|a, b| {
            self.row_indices[*a]
                .cmp(&self.row_indices[*b])
                .then(self.col_indices[*a].cmp(&self.col_indices[*b]))
        });
        CooMatrix::new(
            indices.iter().map(|i| self.row_indices[*i]).collect(),
            indices.iter().map(|i| self.col_indices[*i]).collect(),
            indices.iter().map(|i| self.values[*i]).collect(),
        )
    }

    pub fn insert(&mut self, x: usize, y: usize, value: f64) {
        self.row_indices.push(x);
        self.col_indices.push(y);
        self.values.push(value);
    }

    pub fn transpose(&self) -> CooMatrix {
        CooMatrix::new(
            self.col_indices.clone(),
            self.row_indices.clone(),
            self.values.clone(),
        )
    }

    /// multiplies the sparse matrix with a CooMatrix matrix and sums the results
    ///
    /// # Arguments
    /// `matrix` - the CooMatrix matrix to multiply with
    pub fn mul_and_sum(&self, other: &Matrix) -> f64 {
        self.row_indices
            .par_iter()
            .enumerate()
            .zip(self.col_indices.par_iter())
            .map(|((i, x), y)| other.get(*x, *y) * self.values[i])
            .sum()
    }
}

impl Display for CooMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, x_i) in self.row_indices.iter().enumerate() {
            write!(f, "({}, {}) = {}", x_i, self.col_indices[i], self.values[i])?;
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Mul<&Matrix> for &CooMatrix {
    type Output = CooMatrix;

    fn mul(self, other: &Matrix) -> CooMatrix {
        let result_vec = self
            .row_indices
            .iter()
            .enumerate()
            .zip(self.col_indices.iter())
            .map(|((i, x), y)| other.get(*x, *y) * self.values[i])
            .collect::<Vec<f64>>();
        CooMatrix::new(
            self.col_indices.clone(),
            self.row_indices.clone(),
            result_vec,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_get() {
        let sparse = CooMatrix::new(
            vec![0, 0, 1, 2, 2],
            vec![0, 1, 0, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        );

        assert_eq!(sparse.get(0, 0), 1.0);
        assert_eq!(sparse.get(0, 1), 2.0);
        assert_eq!(sparse.get(1, 0), 3.0);
        assert_eq!(sparse.get(2, 0), 4.0);
        assert_eq!(sparse.get(2, 1), 5.0);
        assert_eq!(sparse.get(1, 1), 0.0);
        assert_eq!(sparse.get(3, 0), 0.0);
        println!("{}", sparse);
    }

    #[test]
    fn test_sparse_add_or_insert() {
        let mut matrix = CooMatrix::new(vec![0, 2, 2], vec![0, 0, 1], vec![1.0, 3.0, 5.0]);

        matrix.add_or_insert(0, 0, 1.0); // adds a value to an existing element
        matrix.add_or_insert(1, 0, 1.0); // adds a new element

        assert_eq!(matrix.get(0, 0), 2.0);
        assert_eq!(matrix.get(1, 0), 1.0);
    }

    #[test]
    fn test_sparse_sort() {
        let matrix = CooMatrix::new(
            vec![0, 2, 2, 1, 0, 3, 3, 1, 0],
            vec![4, 0, 1, 0, 1, 4, 1, 0, 5],
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0],
        );
        let matrix = matrix.sort();
        assert!(is_sorted(&matrix.row_indices));

        // create a vector of y vectors for each x_index
        let mut y_vecs: Vec<Vec<usize>> = vec![];
        for i in 0..matrix.row_indices.len() {
            if i == 0 || matrix.row_indices[i] != matrix.row_indices[i - 1] {
                y_vecs.push(vec![]);
            }
            y_vecs.last_mut().unwrap().push(matrix.col_indices[i]);
        }

        y_vecs.iter().for_each(|y_vec| assert!(is_sorted(y_vec)));
    }

    #[test]
    fn test_mul_sparse_by_dense() {
        let coo_matrix = CooMatrix::new(vec![0, 1], vec![0, 1], vec![1.0, 3.0]);
        let dense = Matrix::new(2, 3, vec![4.0, 1.0, 2.0, 3.0, 2.0, 5.0]);
        let result = &coo_matrix * &dense;

        let expected = CooMatrix::new(vec![0, 1], vec![0, 1], vec![4.0, 6.0]);
        assert_eq!(result.values, expected.values);
        assert_eq!(result.row_indices, expected.row_indices);
        assert_eq!(result.col_indices, expected.col_indices);
    }

    #[test]
    fn test_mul_sparse_by_dense_and_sum() {
        let sparse = CooMatrix::new(vec![0, 1], vec![0, 1], vec![1.0, 3.0]);
        let dense = Matrix::new(2, 3, vec![4.0, 1.0, 2.0, 3.0, 2.0, 5.0]);
        let result = &sparse.mul_and_sum(&dense);
        assert_eq!(result, &10.0);
    }

    #[test]
    fn test_to_csr() {
        let mut sparse = CooMatrix::new(
            vec![1, 1, 2, 2, 5, 5],
            vec![1, 0, 0, 1, 0, 2],
            vec![2.0, 3.0, 4.0, 5.0, 100., 102.],
        );
        let csr = sparse.to_csr();
        assert_eq!(csr.values, vec![3.0, 2.0, 4.0, 5.0, 100.0, 102.0]);
        assert_eq!(csr.col_indices, vec![0, 1, 0, 1, 0, 2]);
        assert_eq!(csr.row_pointer, vec![0, 0, 2, 4, 4, 4, 6]);
    }

    fn is_sorted<I>(data: &I) -> bool
    where
        I: IntoIterator + Clone,
        I::Item: Ord + Clone,
    {
        let data = data.clone();
        let mut it = data.into_iter();
        match it.next() {
            None => true,
            Some(first) => it
                .scan(first, |state, next| {
                    let cmp = *state <= next;
                    *state = next;
                    Some(cmp)
                })
                .all(|b| b),
        }
    }
}
