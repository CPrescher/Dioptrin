//! Poni file handling
//!
//! Provides functions to read and write poni files into and from the Poni struct.
#![allow(dead_code)]

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

/// A struct to hold the poni file data
/// which provides the calibration parameters for the detector
/// poni stands for point of normal incidence
pub struct Poni {
    pixel1: f32,
    pixel2: f32,
    distance: f32,
    poni1: f32,
    poni2: f32,
    rot1: f32,
    rot2: f32,
    rot3: f32,
    spline_file: String,
    wavelength: f32,
}

impl Poni {
    /// Create a new Poni struct with all parameters set to 0
    pub fn new() -> Poni {
        Poni {
            pixel1: 0.0,
            pixel2: 0.0,
            distance: 0.0,
            poni1: 0.0,
            poni2: 0.0,
            rot1: 0.0,
            rot2: 0.0,
            rot3: 0.0,
            spline_file: String::new(),
            wavelength: 0.0,
        }
    }

    /// Read a poni file into a Poni struct
    pub fn read(filename: &str) -> Poni {
        println!("Reading {}", filename);
        let mut poni = Poni::new();

        if let Ok(lines) = read_lines(filename) {
            for line in lines {
                if let Ok(line) = line {
                    if !is_comment(&line) {
                        let parts: Vec<&str> = line.split(":").collect();
                        let key = parts[0].trim();
                        let value = parts[1].trim();
                        match key {
                            "PixelSize1" => poni.pixel1 = value.parse().unwrap(),
                            "PixelSize2" => poni.pixel2 = value.parse().unwrap(),
                            "Distance" => poni.distance = value.parse().unwrap(),
                            "Poni1" => poni.poni1 = value.parse().unwrap(),
                            "Poni2" => poni.poni2 = value.parse().unwrap(),
                            "Rot1" => poni.rot1 = value.parse().unwrap(),
                            "Rot2" => poni.rot2 = value.parse().unwrap(),
                            "Rot3" => poni.rot3 = value.parse().unwrap(),
                            "SplineFile" => poni.spline_file = value.to_string(),
                            "Wavelength" => poni.wavelength = value.parse().unwrap(),
                            _ => println!("Unknown key: {}", key),
                        }
                    }
                }
            }
        }
        return poni;
    }
}

/// Read a file line by line
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn is_comment(str: &String) -> bool {
    str[0..1] == "#".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = Poni::new();
        assert_eq!(result.pixel1, 0.0);
    }

    #[test]
    fn read_poni_version_1() {
        let result = Poni::read("test_data/v1.poni");
        assert_eq!(result.pixel1, 0.000172);
    }
}
