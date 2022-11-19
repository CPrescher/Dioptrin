use std::fs;
use std::io::prelude::*;
use std::io::Write;
use std::path::Path;

/// Downloads the CeO2_Pilatus1M.tiff file from the Dioptas repository into the test_data directory
pub fn download_pilatus_test_file() {
    let url =
        "https://github.com/Dioptas/Dioptas/raw/develop/dioptas/tests/data/CeO2_Pilatus1M.tif";
    let path = Path::new("test_data/CeO2_Pilatus1M.tif");
    if !path.exists() {
        let mut file = fs::File::create(path).unwrap();
        let mut response = reqwest::blocking::get(url).unwrap();
        response.copy_to(&mut file).unwrap();
    }
}

/// Saves a vector of floats as a plain text file into the test_data/test_output directory
pub fn save_vec(path: &str, vec: &Vec<f64>) {
    let dir_str: String = "test_data/test_output/".to_owned();
    fs::create_dir(&dir_str).unwrap_or_default();

    let path_str = dir_str + path;
    let path: &Path = Path::new(path_str.as_str());
    let mut file = fs::File::create(path).unwrap();
    for val in vec.iter() {
        file.write_all(format!("{}\n", val).as_bytes()).unwrap();
    }
}

/// Reads a vector of floats from a plain text file in the test_data/test_reference directory
pub fn read_vec(path: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let path_str = "test_data/test_reference/".to_owned() + path;
    let path: &Path = Path::new(path_str.as_str());
    let mut file = fs::File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let mut vec = Vec::new();
    for line in contents.lines() {
        vec.push(line.parse::<f64>()?);
    }
    Ok(vec)
}

/// compares two floats arrays with each other
pub fn compare_vecs(actual: &[f64], expected: &[f64], what: &str, precision: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{} actual length = {}, expected length = {}",
        what,
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        if (*e - *a).abs() > precision {
            panic!("{}: {} actual = {}, expected = {}", i, what, *a, *e);
        }
    }
}
