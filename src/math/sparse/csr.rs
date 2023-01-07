use std::fmt::Display;

/// A sparse matrix in the Compressed Sparse Row (CSR) format.
/// The CSR format stores the column indices, row pointers and values in three separate vectors.
/// The column indices are behaving the same as in the CooMatrix, but the row pointer are rather giving the start index of each row
/// in respect to the column matrix. See https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
/// for more information.
pub struct CsrMatrix {
    pub col_indices: Vec<usize>,
    pub row_pointer: Vec<usize>,
    pub values: Vec<f64>,
}

impl CsrMatrix {
    pub fn new(col_indices: Vec<usize>, row_pointer: Vec<usize>, values: Vec<f64>) -> CsrMatrix {
        CsrMatrix {
            col_indices,
            row_pointer,
            values,
        }
    }
}

impl Display for CsrMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // find dimension of matrix
        for i in 0..self.row_pointer.len() - 1 {
            for j in self.row_pointer[i]..self.row_pointer[i + 1] {
                write!(f, "({}, {}) = {}", i, self.col_indices[j], self.values[j])?;
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_csr_matrix() {
        let matrix = CsrMatrix::new(
            vec![0, 2, 2, 1, 0, 3, 3, 1, 0],
            vec![0, 2, 4, 6, 8, 9],
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0],
        );
        assert_eq!(matrix.col_indices, vec![0, 2, 2, 1, 0, 3, 3, 1, 0]);
        assert_eq!(matrix.row_pointer, vec![0, 2, 4, 6, 8, 9]);
        assert_eq!(
            matrix.values,
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0]
        );
        print!("{}", matrix);
    }
}
