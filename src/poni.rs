//! Poni file handling
//!
//! Provides functions to read and write poni files into and from the Poni struct.

use std::fs::File;
use std::io::{self, BufRead};
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

/// Parsed fields extracted from a v2/v2.1 `Detector_config` JSON object.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct DetectorConfig {
    pixel1: Option<f64>,
    pixel2: Option<f64>,
    binning: Option<[usize; 2]>,
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
    /// omits `pixel1` and/or `pixel2` in `Detector_config`, any missing pixel
    /// size is resolved from the built-in detector catalog and scaled by
    /// `binning` if present in `Detector_config`. Note that spline-based
    /// detectors (`splineFile` in `Detector_config`) can carry their own pixel
    /// size in the spline file, which may differ from the catalog default.
    pub fn load(filename: &str) -> Result<Poni, Box<dyn std::error::Error>> {
        let mut poni = Poni::new();
        let mut detector_name: Option<String> = None;
        let mut binning: Option<[usize; 2]> = None;

        let lines = read_lines(filename)?;

        for line in lines.map_while(Result::ok) {
            if !is_comment(&line) {
                Poni::assign_values(&mut poni, &mut detector_name, &mut binning, &line)?;
            }
        }

        if poni.pixel1 == 0.0 || poni.pixel2 == 0.0 {
            if let Some(name) = detector_name.as_deref()
                && !name.is_empty()
                && !name.eq_ignore_ascii_case("detector")
            {
                let det = crate::detector::named(name).map_err(|_| {
                    format!(
                        "{filename}: unknown detector '{name}' and no pixel sizes in Detector_config"
                    )
                })?;
                let [b1, b2] = binning.unwrap_or([1, 1]);
                if !det.binning_pitches.is_empty()
                    && [b1, b2]
                        .iter()
                        .any(|b| !det.binning_pitches.iter().any(|(k, _)| k == b))
                {
                    return Err(format!(
                        "{filename}: unsupported binning [{b1}, {b2}] for detector '{name}'"
                    )
                    .into());
                }
                let pitch = |b: usize, base: f64| {
                    det.binning_pitches
                        .iter()
                        .find(|(k, _)| *k == b)
                        .map(|(_, p)| *p)
                        .unwrap_or(base * b as f64)
                };
                if poni.pixel1 == 0.0 {
                    poni.pixel1 = pitch(b1, det.pixel1);
                }
                if poni.pixel2 == 0.0 {
                    poni.pixel2 = pitch(b2, det.pixel2);
                }
            } else {
                return Err(format!(
                    "{filename}: missing pixel sizes (pixel1/pixel2) in PONI file"
                )
                .into());
            }
        }

        Ok(poni)
    }

    /// Assigns values in each line to the corresponding field in the Poni struct
    fn assign_values(
        poni: &mut Poni,
        detector_name: &mut Option<String>,
        binning: &mut Option<[usize; 2]>,
        line: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let parts: Vec<&str> = line.split(':').collect();
        let key = parts[0].trim();
        let temp = parts[1..].join(":");
        let value = temp.trim();
        match key {
            "PixelSize1" => poni.pixel1 = value.parse()?,
            "PixelSize2" => poni.pixel2 = value.parse()?,
            "Detector" => *detector_name = Some(value.to_string()),
            "Detector_config" => {
                let cfg = Poni::parse_detector_config(value)?;
                if let Some(p1) = cfg.pixel1 {
                    poni.pixel1 = p1;
                }
                if let Some(p2) = cfg.pixel2 {
                    poni.pixel2 = p2;
                }
                if let Some(b) = cfg.binning {
                    *binning = Some(b);
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

    /// Parse the `Detector_config` JSON object to extract optional pixel sizes
    /// and binning.
    fn parse_detector_config(value: &str) -> Result<DetectorConfig, Box<dyn std::error::Error>> {
        let parsed: serde_json::Value = serde_json::from_str(value)?;
        let Some(obj) = parsed.as_object() else {
            return Err("Detector_config must be a JSON object".into());
        };
        let parse_pixel = |key: &str| -> Result<Option<f64>, Box<dyn std::error::Error>> {
            match obj.get(key) {
                None | Some(serde_json::Value::Null) => Ok(None),
                Some(v) => v
                    .as_f64()
                    .filter(|x| x.is_finite())
                    .map(Some)
                    .ok_or_else(|| {
                        format!("Detector_config '{key}' must be a finite number").into()
                    }),
            }
        };
        let binning = match obj.get("binning") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::Number(n)) => {
                let b = usize::try_from(
                    n.as_u64()
                        .ok_or("Detector_config 'binning' must be a positive integer")?,
                )?;
                if b == 0 {
                    return Err("Detector_config 'binning' must be positive".into());
                }
                Some([b, b])
            }
            Some(serde_json::Value::Array(arr)) if arr.len() == 2 => {
                let b1 = usize::try_from(
                    arr[0]
                        .as_u64()
                        .ok_or("Detector_config 'binning' entries must be positive integers")?,
                )?;
                let b2 = usize::try_from(
                    arr[1]
                        .as_u64()
                        .ok_or("Detector_config 'binning' entries must be positive integers")?,
                )?;
                if b1 == 0 || b2 == 0 {
                    return Err("Detector_config 'binning' entries must be positive".into());
                }
                Some([b1, b2])
            }
            Some(_) => {
                return Err(
                    "Detector_config 'binning' must be a positive integer or [b1, b2] array".into(),
                );
            }
        };
        Ok(DetectorConfig {
            pixel1: parse_pixel("pixel1")?,
            pixel2: parse_pixel("pixel2")?,
            binning,
        })
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

    fn load_temp_poni(tag: &str, content: &str) -> Result<Poni, Box<dyn std::error::Error>> {
        let path =
            std::env::temp_dir().join(format!("dioptrin_poni_{}_{}.poni", std::process::id(), tag));
        std::fs::write(&path, content)?;
        let res = Poni::load(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        res
    }

    #[test]
    fn read_detector_config() {
        let config_str = "{\"pixel1\": 0.0002, \"pixel2\": 0.0003, \"max_shape\": [605, 1070]}";
        let cfg = Poni::parse_detector_config(config_str).unwrap();
        assert_eq!(
            cfg,
            DetectorConfig {
                pixel1: Some(0.0002),
                pixel2: Some(0.0003),
                binning: None,
            }
        );

        let orientation_only = "{\"orientation\": 2}";
        assert_eq!(
            Poni::parse_detector_config(orientation_only).unwrap(),
            DetectorConfig::default()
        );

        let binned = "{\"binning\": [2, 4], \"orientation\": 2}";
        assert_eq!(
            Poni::parse_detector_config(binned).unwrap().binning,
            Some([2, 4])
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
    fn read_poni_named_detector_with_binning() {
        let result = load_temp_poni(
            "binned_pair",
            "poni_version: 2.1\nDetector: Eiger2CdTe_9M\nDetector_config: {\"binning\": [2, 4], \"orientation\": 2}\nDistance: 0.2\n",
        )
        .unwrap();
        assert_eq!(result.pixel1, 1.5e-4);
        assert_eq!(result.pixel2, 3.0e-4);

        let scalar = load_temp_poni(
            "binned_scalar",
            "poni_version: 2.1\nDetector: Eiger2CdTe_9M\nDetector_config: {\"binning\": 2}\nDistance: 0.2\n",
        )
        .unwrap();
        assert_eq!(scalar.pixel1, 1.5e-4);
        assert_eq!(scalar.pixel2, 1.5e-4);
    }

    #[test]
    fn read_poni_detector_config_before_detector_name() {
        let result = load_temp_poni(
            "config_before_name",
            "poni_version: 2.1\nDetector_config: {\"orientation\": 2}\nDetector: Eiger2CdTe_9M\nDistance: 0.2\n",
        )
        .unwrap();
        assert_eq!(result.pixel1, 7.5e-5);
        assert_eq!(result.pixel2, 7.5e-5);
    }

    #[test]
    fn read_poni_v1_pixelsize_overrides_detector_name() {
        let result = load_temp_poni(
            "v1_override",
            "Detector: Eiger2CdTe_9M\nPixelSize1: 0.0002\nPixelSize2: 0.0003\nDistance: 0.2\n",
        )
        .unwrap();
        assert_eq!(result.pixel1, 0.0002);
        assert_eq!(result.pixel2, 0.0003);
    }

    #[test]
    fn read_poni_explicit_pixel_size_preserved_with_detector_name() {
        let result = Poni::load("test_data/ceo2_side.poni").unwrap();
        assert_eq!(result.pixel1, 0.00017199999999999998);
        assert_eq!(result.pixel2, 0.00017199999999999998);
    }

    #[test]
    fn read_poni_generic_detector_without_pixel_size_errors() {
        let err = load_temp_poni(
            "generic_no_pixels",
            "poni_version: 2.1\nDetector: Detector\nDetector_config: {\"orientation\": 2}\nDistance: 0.2\n",
        )
        .unwrap_err();
        assert!(err.to_string().contains("missing pixel sizes"));
    }

    #[test]
    fn read_poni_unknown_detector_without_pixel_size_errors() {
        let err = load_temp_poni(
            "unknown_detector",
            "poni_version: 2.1\nDetector: NonExistentDetector999\nDetector_config: {\"orientation\": 2}\nDistance: 0.2\n",
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NonExistentDetector999"), "{msg}");
        assert!(msg.contains("dioptrin_poni_"), "{msg}");
    }

    #[test]
    fn read_non_existent_poni_file() {
        let result = Poni::load("test_data/non_existent.poni");
        assert!(result.is_err());
    }
}
