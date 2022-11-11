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

    /// Prints the poni data
    pub fn print(&self) {
        println!("pixel1: {}", self.pixel1);
        println!("pixel2: {}", self.pixel2);
        println!("distance: {}", self.distance);
        println!("poni1: {}", self.poni1);
        println!("poni2: {}", self.poni2);
        println!("rot1: {}", self.rot1);
        println!("rot2: {}", self.rot2);
        println!("rot3: {}", self.rot3);
        println!("spline_file: {}", self.spline_file);
        println!("wavelength: {}", self.wavelength);
    }

    /// Read a poni file into a Poni struct
    pub fn read(filename: &str) -> Poni {
        println!("Reading {}", filename);
        let mut poni = Poni::new();

        if let Ok(lines) = read_lines(filename) {
            for line in lines {
                if let Ok(line) = line {
                    if !is_comment(&line) {
                        Poni::assign_values(&mut poni, &line);
                    }
                }
            }
        }
        return poni;
    }

    /// Assigns values in each line to the corresponding field in the Poni struct
    fn assign_values(poni: &mut Poni, line: &str) {
        let parts: Vec<&str> = line.split(":").collect();
        let key = parts[0].trim();
        let temp = parts[1..].join(":");
        let value = temp.trim();
        match key {
            "PixelSize1" => poni.pixel1 = value.parse().unwrap(),
            "PixelSize2" => poni.pixel2 = value.parse().unwrap(),
            "Detector_config" => {
                let (pixel1, pixel2) = Poni::parse_detector_config(value);
                poni.pixel1 = pixel1;
                poni.pixel2 = pixel2;
            }
            "Distance" => poni.distance = value.parse().unwrap(),
            "Poni1" => poni.poni1 = value.parse().unwrap(),
            "Poni2" => poni.poni2 = value.parse().unwrap(),
            "Rot1" => poni.rot1 = value.parse().unwrap(),
            "Rot2" => poni.rot2 = value.parse().unwrap(),
            "Rot3" => poni.rot3 = value.parse().unwrap(),
            "SplineFile" => poni.spline_file = value.to_string(),
            "Wavelength" => poni.wavelength = value.parse().unwrap(),
            _ => (),
        }
    }

    /// Parse the detector config string to get the pixel sizes
    /// typical format of the string is {"pixel1": 0.0002, "pixel2": 0.0002, "max_shape": [605, 1070]}
    fn parse_detector_config(value: &str) -> (f32, f32) {
        let mut pixel1: f32 = 0.0;
        let mut pixel2: f32 = 0.0;

        let key_value_parts: Vec<&str> = value[1..value.len() - 1].split(",").collect();
        for part in key_value_parts {
            let key_value: Vec<&str> = part.split(":").collect();
            if key_value.len() == 2 {
                let key = key_value[0].trim();
                let temp = key_value[1..].join(":");
                let value = temp.trim();
                match key {
                    "\"pixel1\"" => pixel1 = value.parse().unwrap(),
                    "\"pixel2\"" => pixel2 = value.parse().unwrap(),
                    _ => (),
                }
            }
        }
        return (pixel1, pixel2);
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

    #[test]
    fn read_detector_config() {
        let config_str = "{\"pixel1\": 0.0002, \"pixel2\": 0.0003, \"max_shape\": [605, 1070]}";
        let (pixel1, pixel2) = Poni::parse_detector_config(config_str);
        assert_eq!(pixel1, 0.0002);
        assert_eq!(pixel2, 0.0003);
    }

    #[test]
    fn read_poni_version2() {
        let result = Poni::read("test_data/v2.poni");
        assert_eq!(result.pixel1, 0.0002);
        result.print();
    }
}
