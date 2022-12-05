use std::fmt::Display;
use std::ops::Mul;

use rayon::prelude::*;

use crate::math::matrix::Matrix;

pub struct SparseMatrix {
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
}

impl SparseMatrix {
    fn new(row_indices: Vec<usize>, col_indices: Vec<usize>, values: Vec<f64>) -> SparseMatrix {
        SparseMatrix {
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
            self.insert_element(x, y, value)
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

    pub fn to_csr(&self, n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let mut csr_x_index = vec![0; n + 1];
        let mut csr_y_index = vec![];
        let mut csr_values = vec![];
        for i in 0..n {
            let mut count = 0;
            for (j, x_j) in self.row_indices.iter().enumerate() {
                if *x_j == i {
                    csr_y_index.push(self.col_indices[j]);
                    csr_values.push(self.values[j]);
                    count += 1;
                }
            }
            csr_x_index[i + 1] = csr_x_index[i] + count;
        }
        (csr_x_index, csr_y_index, csr_values)
    }

    pub fn sort(&mut self) {
        let mut indices = (0..self.row_indices.len()).collect::<Vec<usize>>();
        indices.sort_by(|a, b| {
            self.row_indices[*a]
                .cmp(&self.row_indices[*b])
                .then(self.col_indices[*a].cmp(&self.col_indices[*b]))
        });
        self.row_indices = indices.iter().map(|i| self.row_indices[*i]).collect();
        self.col_indices = indices.iter().map(|i| self.col_indices[*i]).collect();
        self.values = indices.iter().map(|i| self.values[*i]).collect();
    }

    pub fn insert_element(&mut self, x: usize, y: usize, value: f64) {
        self.row_indices.push(x);
        self.col_indices.push(y);
        self.values.push(value);
    }

    pub fn transpose(&self) -> SparseMatrix {
        SparseMatrix::new(
            self.col_indices.clone(),
            self.row_indices.clone(),
            self.values.clone(),
        )
    }

    /// multiplies the sparse matrix with a dense matrix and sums the results
    ///
    /// # Arguments
    /// `matrix` - the dense matrix to multiply with
    pub fn mul_and_sum(&self, other: &Matrix) -> f64 {
        self.row_indices
            .par_iter()
            .enumerate()
            .zip(self.col_indices.par_iter())
            .map(|((i, x), y)| other.get(*x, *y) * self.values[i])
            .sum()
    }
}

impl Display for SparseMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, x_i) in self.row_indices.iter().enumerate() {
            write!(f, "({}, {}) = {}", x_i, self.col_indices[i], self.values[i])?;
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Mul<&Matrix> for &SparseMatrix {
    type Output = SparseMatrix;

    fn mul(self, other: &Matrix) -> SparseMatrix {
        let result_vec = self
            .row_indices
            .iter()
            .enumerate()
            .zip(self.col_indices.iter())
            .map(|((i, x), y)| other.get(*x, *y) * self.values[i])
            .collect::<Vec<f64>>();
        SparseMatrix::new(
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
        let sparse = SparseMatrix::new(
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
        let mut sparse = SparseMatrix::new(vec![0, 2, 2], vec![0, 0, 1], vec![1.0, 3.0, 5.0]);

        sparse.add_or_insert(0, 0, 1.0); // adds a value to an existing element
        sparse.add_or_insert(1, 0, 1.0); // adds a new element

        assert_eq!(sparse.get(0, 0), 2.0);
        assert_eq!(sparse.get(1, 0), 1.0);
    }

    #[test]
    fn test_sparse_sort() {
        let mut sparse = SparseMatrix::new(
            vec![0, 2, 2, 1, 0, 3, 3, 1, 0],
            vec![4, 0, 1, 0, 1, 4, 1, 0, 5],
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0],
        );
        sparse.sort();
        assert!(is_sorted(&sparse.row_indices));

        // create a vector of y vectors for each x_index
        let mut y_vecs: Vec<Vec<usize>> = vec![];
        for i in 0..sparse.row_indices.len() {
            if i == 0 || sparse.row_indices[i] != sparse.row_indices[i - 1] {
                y_vecs.push(vec![]);
            }
            y_vecs.last_mut().unwrap().push(sparse.col_indices[i]);
        }

        y_vecs.iter().for_each(|y_vec| assert!(is_sorted(y_vec)));
    }

    #[test]
    fn test_mul_sparse_by_dense() {
        let sparse = SparseMatrix::new(vec![0, 1], vec![0, 1], vec![1.0, 3.0]);
        let dense = Matrix::new(2, 3, vec![4.0, 1.0, 2.0, 3.0, 2.0, 5.0]);
        let result = &sparse * &dense;

        let expected = SparseMatrix::new(vec![0, 1], vec![0, 1], vec![4.0, 6.0]);
        assert_eq!(result.values, expected.values);
        assert_eq!(result.row_indices, expected.row_indices);
        assert_eq!(result.col_indices, expected.col_indices);
    }

    #[test]
    fn test_mul_sparse_by_dense_and_sum() {
        let sparse = SparseMatrix::new(vec![0, 1], vec![0, 1], vec![1.0, 3.0]);
        let dense = Matrix::new(2, 3, vec![4.0, 1.0, 2.0, 3.0, 2.0, 5.0]);
        let result = &sparse.mul_and_sum(&dense);
        assert_eq!(result, &10.0);
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
