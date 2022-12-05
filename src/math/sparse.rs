pub struct Sparse {
    x_index: Vec<usize>,
    y_index: Vec<usize>,
    values: Vec<f64>,
}

impl Sparse {
    fn new(x_index: Vec<usize>, y_index: Vec<usize>, values: Vec<f64>) -> Sparse {
        Sparse {
            x_index,
            y_index,
            values,
        }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        let mut value = 0.0;
        for (i, x_i) in self.x_index.iter().enumerate() {
            if *x_i == x && self.y_index[i] == y {
                value = self.values[i];
            }
        }
        value
    }

    pub fn set(&mut self, x: usize, y: usize, value: f64) {
        for (i, x_i) in self.x_index.iter().enumerate() {
            if *x_i == x && self.y_index[i] == y {
                self.values[i] = value;
            }
        }
    }

    pub fn add(&mut self, x: usize, y: usize, value: f64) {
        for (i, x_i) in self.x_index.iter().enumerate() {
            if *x_i == x && self.y_index[i] == y {
                self.values[i] += value;
            }
        }
    }

    pub fn add_or_insert(&mut self, x: usize, y: usize, value: f64) {
        let mut found = false;
        for (i, x_i) in self.x_index.iter().enumerate() {
            if *x_i == x && self.y_index[i] == y {
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
        for (i, x_i) in self.x_index.iter().enumerate() {
            if *x_i == x && self.y_index[i] == y {
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
            for (j, x_j) in self.x_index.iter().enumerate() {
                if *x_j == i {
                    csr_y_index.push(self.y_index[j]);
                    csr_values.push(self.values[j]);
                    count += 1;
                }
            }
            csr_x_index[i + 1] = csr_x_index[i] + count;
        }
        (csr_x_index, csr_y_index, csr_values)
    }

    pub fn sort(&mut self) {
        let mut indices = (0..self.x_index.len()).collect::<Vec<usize>>();
        indices.sort_by(|a, b| {
            self.x_index[*a]
                .cmp(&self.x_index[*b])
                .then(self.y_index[*a].cmp(&self.y_index[*b]))
        });
        self.x_index = indices.iter().map(|i| self.x_index[*i]).collect();
        self.y_index = indices.iter().map(|i| self.y_index[*i]).collect();
        self.values = indices.iter().map(|i| self.values[*i]).collect();
    }

    pub fn insert_element(&mut self, x: usize, y: usize, value: f64) {
        self.x_index.push(x);
        self.y_index.push(y);
        self.values.push(value);
    }

    pub fn transpose(&self) -> Sparse {
        Sparse::new(
            self.y_index.clone(),
            self.x_index.clone(),
            self.values.clone(),
        )
    }

    pub fn print(&self) {
        for (i, x_i) in self.x_index.iter().enumerate() {
            println!("({}, {}) = {}", x_i, self.y_index[i], self.values[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_get() {
        let sparse = Sparse::new(
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
    }

    #[test]
    fn test_sparse_add_or_insert() {
        let mut sparse = Sparse::new(vec![0, 2, 2], vec![0, 0, 1], vec![1.0, 3.0, 5.0]);

        sparse.add_or_insert(0, 0, 1.0); // adds a value to an existing element
        sparse.add_or_insert(1, 0, 1.0); // adds a new element

        assert_eq!(sparse.get(0, 0), 2.0);
        assert_eq!(sparse.get(1, 0), 1.0);
    }

    #[test]
    fn test_sparse_sort() {
        let mut sparse = Sparse::new(
            vec![0, 2, 2, 1, 0, 3, 3, 1, 0],
            vec![4, 0, 1, 0, 1, 4, 1, 0, 5],
            vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0],
        );
        sparse.sort();
        assert!(is_sorted(&sparse.x_index));

        // create a vector of y vectors for each x_index
        let mut y_vecs: Vec<Vec<usize>> = vec![];
        for i in 0..sparse.x_index.len() {
            if i == 0 || sparse.x_index[i] != sparse.x_index[i - 1] {
                y_vecs.push(vec![]);
            }
            y_vecs.last_mut().unwrap().push(sparse.y_index[i]);
        }

        y_vecs.iter().for_each(|y_vec| assert!(is_sorted(y_vec)));
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
