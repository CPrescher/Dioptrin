//! Calibrant: a reference powder with known d-spacings (e.g. LaB6, CeO2).
//!
//! pyFAI uses a simple text format with `.D` extension. Comments start with
//! `#`; on a data line, the first whitespace-separated token is the d-spacing
//! in Ångström, and anything after an inline `#` is ignored (typically `(h k
//! l) mult intensity`).

use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Calibrant {
    pub name: String,
    /// d-spacings in Ångström, sorted descending (lowest 2θ first, matching
    /// pyFAI's ordering — ring index 0 is the lowest-angle ring).
    pub d_spacings: Vec<f64>,
}

#[derive(Debug)]
pub enum CalibrantError {
    Io(std::io::Error),
    /// File parsed but contained no d-spacing entries.
    Empty,
    /// A data line could not be parsed as a number.
    Parse { line: usize, content: String },
}

impl fmt::Display for CalibrantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalibrantError::Io(err) => write!(f, "I/O error: {}", err),
            CalibrantError::Empty => write!(f, "no d-spacing entries found in file"),
            CalibrantError::Parse { line, content } => {
                write!(f, "could not parse d-spacing on line {}: {:?}", line, content)
            }
        }
    }
}

impl std::error::Error for CalibrantError {}

impl From<std::io::Error> for CalibrantError {
    fn from(err: std::io::Error) -> Self {
        CalibrantError::Io(err)
    }
}

impl Calibrant {
    /// Construct from an explicit list of d-spacings (Å).
    pub fn from_d_spacings(name: impl Into<String>, d_spacings: Vec<f64>) -> Self {
        let mut sorted = d_spacings;
        // Keep pyFAI's convention: descending d (= ascending 2θ).
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Self {
            name: name.into(),
            d_spacings: sorted,
        }
    }

    /// Parse a pyFAI-style `.D` calibrant file.
    pub fn from_d_file<P: AsRef<Path>>(path: P) -> Result<Self, CalibrantError> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("calibrant")
            .to_string();
        Self::from_d_str(name, &contents)
    }

    /// Parse a pyFAI-style `.D` calibrant from an in-memory string.
    pub fn from_d_str(name: impl Into<String>, contents: &str) -> Result<Self, CalibrantError> {
        let mut d_spacings = Vec::new();
        for (i, raw_line) in contents.lines().enumerate() {
            // Strip inline comments (everything from the first `#`).
            let line = match raw_line.split_once('#') {
                Some((before, _)) => before,
                None => raw_line,
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // The first whitespace-separated token is the d-spacing.
            let token = trimmed.split_whitespace().next().unwrap();
            let d: f64 = token.parse().map_err(|_| CalibrantError::Parse {
                line: i + 1,
                content: raw_line.to_string(),
            })?;
            if d > 0.0 {
                d_spacings.push(d);
            }
        }
        if d_spacings.is_empty() {
            return Err(CalibrantError::Empty);
        }
        Ok(Self::from_d_spacings(name, d_spacings))
    }

    /// Number of rings in this calibrant.
    pub fn n_rings(&self) -> usize {
        self.d_spacings.len()
    }

    /// 2θ (radians) of each ring at the given wavelength (meters), using
    /// Bragg's law: `2θ = 2 · arcsin(λ / 2d)`.
    ///
    /// Rings whose Bragg condition is not physical (λ > 2d) are returned as
    /// `None` so the caller can drop them.
    pub fn two_theta_rad(&self, wavelength_m: f64) -> Vec<Option<f64>> {
        // d_spacings are in Å; convert wavelength to Å for the ratio.
        let wl_angstrom = wavelength_m * 1.0e10;
        self.d_spacings
            .iter()
            .map(|&d| {
                let arg = wl_angstrom / (2.0 * d);
                if !(-1.0..=1.0).contains(&arg) {
                    None
                } else {
                    Some(2.0 * arg.asin())
                }
            })
            .collect()
    }
}
