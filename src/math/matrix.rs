use std::{
    fmt::Display,
    ops::{Add, Div, Mul},
};

use rayon::prelude::*;

struct Matrix {
    pub values: Vec<f64>,
    pub n: usize,
    pub m: usize,
}

impl Matrix {
    pub fn new(n: usize, m: usize, values: Vec<f64>) -> Matrix {
        Matrix { values, n, m }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.values[x * self.m + y]
    }

    pub fn set(&mut self, x: usize, y: usize, value: f64) {
        self.values[x * self.m + y] = value;
    }

    pub fn add_to_index(&mut self, x: usize, y: usize, value: f64) {
        self.values[x * self.m + y] += value;
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.m, other.n);
        let mut result = Matrix::new(self.n, other.m, vec![0.0; self.n * other.m]);
        for i in 0..self.n {
            for j in 0..other.m {
                for k in 0..self.m {
                    result.add_to_index(i, j, self.get(i, k) * other.get(k, j));
                }
            }
        }
        result
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.n {
            for j in 0..self.m {
                write!(f, "{} ", self.get(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, other: &Matrix) -> Matrix {
        assert!(
            self.n == other.n,
            "Matrices must have the same number of rows"
        );
        assert!(
            self.m == other.m,
            "Matrices must have the same number of columns"
        );

        let result_vec = self
            .values
            .par_iter()
            .zip(other.values.par_iter())
            .map(|(a, b)| a * b)
            .collect::<Vec<f64>>();
        Matrix {
            values: result_vec,
            n: self.n,
            m: self.m,
        }
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, other: f64) -> Matrix {
        let mut result = Matrix::new(self.n, self.m, vec![0.0; self.values.len()]);
        for i in 0..self.n {
            for j in 0..self.m {
                result.add_to_index(i, j, self.get(i, j) * other);
            }
        }
        result
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Matrix {
        assert!(
            self.n == other.n,
            "Matrices must have the same number of rows"
        );
        assert!(
            self.m == other.m,
            "Matrices must have the same number of columns"
        );

        let result_vec = self
            .values
            .par_iter()
            .zip(other.values.par_iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<f64>>();
        Matrix {
            values: result_vec,
            n: self.n,
            m: self.m,
        }
    }
}

impl Div<&Matrix> for &Matrix {
    type Output = Matrix;

    fn div(self, other: &Matrix) -> Matrix {
        assert!(
            self.n == other.n,
            "Matrices must have the same number of rows"
        );
        assert!(
            self.m == other.m,
            "Matrices must have the same number of columns"
        );

        let result_vec = self
            .values
            .par_iter()
            .zip(other.values.par_iter())
            .map(|(a, b)| a / b)
            .collect::<Vec<f64>>();
        Matrix {
            values: result_vec,
            n: self.n,
            m: self.m,
        }
    }
}

impl Div<f64> for &Matrix {
    type Output = Matrix;

    fn div(self, other: f64) -> Matrix {
        let result_vec = self
            .values
            .par_iter()
            .map(|val| val / other)
            .collect::<Vec<f64>>();

        Matrix {
            values: result_vec,
            n: self.n,
            m: self.m,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_dense() {
        let (n, m) = (2, 3);
        let dense = Matrix::new(n, m, vec![0.0; n * m]);
        assert_eq!(dense.values, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(dense.n, 2);
        assert_eq!(dense.m, 3);
    }

    #[test]
    fn test_add_dense() {
        let (n, m) = (2, 3);
        let dense_1 = &Matrix::new(n, m, vec![1.0; n * m]);
        let dense_2 = &Matrix::new(n, m, vec![2.0; n * m]);
        let result = dense_1 + dense_2;
        let expected = Matrix::new(n, m, vec![3.0; n * m]);
        assert_eq!(result.values, expected.values);
    }

    #[test]
    fn test_mul_dense_by_scalar() {
        let (n, m) = (2, 3);
        let dense = &Matrix::new(n, m, vec![1.0; n * m]);

        let expected = &Matrix::new(n, m, vec![2.0; n * m]);
        let result = dense * 2.0;
        assert_eq!(result.values, expected.values);
    }

    #[test]
    fn test_mul_dense_by_dense() {
        let (n, m) = (2, 3);
        let dense_1 = &Matrix::new(n, m, vec![1.0; n * m]);
        let dense_2 = &Matrix::new(n, m, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = dense_1 * dense_2;
        assert_eq!(result.values, dense_2.values);
    }

    #[test]
    fn test_div_dense_by_dense() {
        let (n, m) = (2, 3);
        let dense_1 = &Matrix::new(n, m, vec![1.0; n * m]);
        let dense_2 = &Matrix::new(n, m, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let result = dense_1 / dense_2;
        let expected = &Matrix::new(n, m, vec![1.0, 0.5, 1.0 / 3.0, 0.25, 0.2, 1.0 / 6.0]);
        assert_eq!(result.values, expected.values);
    }

    fn test_div_dense_by_scalr() {
        let (n, m) = (2, 3);
        let dense = &Matrix::new(n, m, vec![1.0; n * m]);

        let result = dense / 2.0;
        let expected = &Matrix::new(n, m, vec![0.5; n * m]);
        assert_eq!(result.values, expected.values);
    }
}
