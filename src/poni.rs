//! Poni file handling
//!
//! Provides functions to read and write poni files into and from the Poni struct.
#![allow(dead_code)]

use std::fs::File;
use std::io::{self, BufRead};
use std::num::ParseFloatError;
use std::path::Path;

/// A struct to hold the poni file data
/// which provides the calibration parameters for the detector
/// poni stands for point of normal incidence
#[derive(Clone, Debug)]
pub struct Poni {
    pub pixel1: f64,
    pub pixel2: f64,
    pub distance: f64,
    pub poni1: f64,
    pub poni2: f64,
    pub rot1: f64,
    pub rot2: f64,
    pub rot3: f64,
    pub spline_file: String,
    pub wavelength: f64,
}

impl Default for Poni {
    fn default() -> Self {
        Self::new()
    }
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

    /// Read a poni file into a Poni struct.
    ///
    /// When a v2/v2.1 PONI file names a built-in detector (`Detector: ...`) and
    /// omits explicit `pixel1`/`pixel2` values in `Detector_config`, the pixel
    /// sizes are resolved from the built-in detector catalog.
    pub fn load(filename: &str) -> Result<Poni, Box<dyn std::error::Error>> {
        let mut poni = Poni::new();
        let mut detector_name: Option<String> = None;

        let lines = read_lines(filename)?;

        for line in lines.map_while(Result::ok) {
            if !is_comment(&line) {
                Poni::assign_values(&mut poni, &mut detector_name, &line)?;
            }
        }

        if (poni.pixel1 == 0.0 || poni.pixel2 == 0.0)
            && let Some(name) = detector_name.as_deref()
            && !name.is_empty()
            && !name.eq_ignore_ascii_case("detector")
        {
            let det = crate::detector::named(name)?;
            if poni.pixel1 == 0.0 {
                poni.pixel1 = det.pixel1;
            }
            if poni.pixel2 == 0.0 {
                poni.pixel2 = det.pixel2;
            }
        }

        Ok(poni)
    }

    /// Assigns values in each line to the corresponding field in the Poni struct
    fn assign_values(
        poni: &mut Poni,
        detector_name: &mut Option<String>,
        line: &str,
    ) -> Result<(), ParseFloatError> {
        let parts: Vec<&str> = line.split(":").collect();
        let key = parts[0].trim();
        let temp = parts[1..].join(":");
        let value = temp.trim();
        match key {
            "PixelSize1" => poni.pixel1 = value.parse()?,
            "PixelSize2" => poni.pixel2 = value.parse()?,
            "Detector" => *detector_name = Some(value.to_string()),
            "Detector_config" => {
                let (pixel1, pixel2) = Poni::parse_detector_config(value)?;
                if let Some(p1) = pixel1 {
                    poni.pixel1 = p1;
                }
                if let Some(p2) = pixel2 {
                    poni.pixel2 = p2;
                }
            }
            "Distance" => poni.distance = value.parse()?,
            "Poni1" => poni.poni1 = value.parse()?,
            "Poni2" => poni.poni2 = value.parse()?,
            "Rot1" => poni.rot1 = value.parse()?,
            "Rot2" => poni.rot2 = value.parse()?,
            "Rot3" => poni.rot3 = value.parse()?,
            "SplineFile" => poni.spline_file = value.to_string(),
            "Wavelength" => poni.wavelength = value.parse()?,
            _ => (),
        };
        Ok(())
    }

    /// Parse the detector config string to get optional pixel sizes.
    /// Typical format: `{"pixel1": 0.0002, "pixel2": 0.0002, "max_shape": [605, 1070]}`
    /// or `{"orientation": 2}` (when pixel sizes come from the named `Detector`).
    fn parse_detector_config(value: &str) -> Result<(Option<f64>, Option<f64>), ParseFloatError> {
        let mut pixel1: Option<f64> = None;
        let mut pixel2: Option<f64> = None;

        if value.len() < 2 {
            return Ok((None, None));
        }
        let key_value_parts: Vec<&str> = value[1..value.len() - 1].split(",").collect();
        for part in key_value_parts {
            let key_value: Vec<&str> = part.split(":").collect();
            if key_value.len() == 2 {
                let key = key_value[0].trim();
                let temp = key_value[1..].join(":");
                let value = temp.trim();
                match key {
                    "\"pixel1\"" => pixel1 = Some(value.parse()?),
                    "\"pixel2\"" => pixel2 = Some(value.parse()?),
                    _ => (),
                }
            }
        }
        Ok((pixel1, pixel2))
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

fn is_comment(s: &str) -> bool {
    s.starts_with('#')
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
    fn read_poni_version1() {
        let result = Poni::load("test_data/v1.poni").unwrap();
        assert_eq!(result.pixel1, 0.000172);
    }

    #[test]
    fn read_detector_config() {
        let config_str = "{\"pixel1\": 0.0002, \"pixel2\": 0.0003, \"max_shape\": [605, 1070]}";
        let (pixel1, pixel2) = Poni::parse_detector_config(config_str).unwrap();
        assert_eq!(pixel1, Some(0.0002));
        assert_eq!(pixel2, Some(0.0003));

        let orientation_only = "{\"orientation\": 2}";
        assert_eq!(
            Poni::parse_detector_config(orientation_only).unwrap(),
            (None, None)
        );
    }

    #[test]
    fn read_poni_version2() {
        let result = Poni::load("test_data/v2.poni").unwrap();
        assert_eq!(result.pixel1, 0.0002);
    }

    #[test]
    fn read_poni_named_detector_without_pixel_size() {
        let result = Poni::load("test_data/eiger_0000.poni").unwrap();
        assert_eq!(result.pixel1, 7.5e-5);
        assert_eq!(result.pixel2, 7.5e-5);
    }

    #[test]
    fn read_poni_explicit_pixel_size_preserved_with_detector_name() {
        let result = Poni::load("test_data/ceo2_side.poni").unwrap();
        assert_eq!(result.pixel1, 0.00017199999999999998);
        assert_eq!(result.pixel2, 0.00017199999999999998);
    }

    #[test]
    fn read_poni_unknown_detector_without_pixel_size_errors() {
        let dir = std::env::temp_dir();
        let path = dir.join("dioptrin_unknown_detector_test.poni");
        std::fs::write(
            &path,
            "poni_version: 2.1\nDetector: NonExistentDetector999\nDetector_config: {\"orientation\": 2}\nDistance: 0.2\n",
        )
        .unwrap();
        let res = Poni::load(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        assert!(res.is_err());
    }

    #[test]
    fn read_non_existent_poni_file() {
        let result = Poni::load("test_data/non_existent.poni");
        assert!(result.is_err());
    }
}
