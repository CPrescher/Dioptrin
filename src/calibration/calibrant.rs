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
    /// Optional relative intensities parsed from `.D` file inline comments
    /// (e.g. `# (1 1 1) 8 100.0`), aligned with `d_spacings`.
    pub intensities: Vec<Option<f64>>,
}

#[derive(Debug)]
pub enum CalibrantError {
    Io(std::io::Error),
    /// File parsed but contained no d-spacing entries.
    Empty,
    /// A data line could not be parsed as a number.
    Parse {
        line: usize,
        content: String,
    },
}

impl fmt::Display for CalibrantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalibrantError::Io(err) => write!(f, "I/O error: {}", err),
            CalibrantError::Empty => write!(f, "no d-spacing entries found in file"),
            CalibrantError::Parse { line, content } => {
                write!(
                    f,
                    "could not parse d-spacing on line {}: {:?}",
                    line, content
                )
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
    ///
    /// Non-finite and non-positive values are ignored.
    pub fn from_d_spacings(name: impl Into<String>, d_spacings: Vec<f64>) -> Self {
        let mut sorted: Vec<f64> = d_spacings
            .into_iter()
            .filter(|d| d.is_finite() && *d > 0.0)
            .collect();
        // Keep pyFAI's convention: descending d (= ascending 2θ).
        sorted.sort_by(|a, b| b.total_cmp(a));
        let n = sorted.len();
        Self {
            name: name.into(),
            d_spacings: sorted,
            intensities: vec![None; n],
        }
    }

    /// Construct from d-spacings (Å) and optional per-ring relative intensities.
    ///
    /// Non-finite and non-positive d-spacings are ignored. Remaining entries
    /// are sorted by descending `d_spacing`, keeping each intensity paired with
    /// its ring. If `intensities` is shorter than `d_spacings`, remaining rings
    /// get `None`.
    pub fn from_d_spacings_and_intensities(
        name: impl Into<String>,
        d_spacings: Vec<f64>,
        intensities: Vec<Option<f64>>,
    ) -> Self {
        let mut pairs: Vec<(f64, Option<f64>)> = d_spacings
            .into_iter()
            .enumerate()
            .filter(|(_, d)| d.is_finite() && *d > 0.0)
            .map(|(idx, d)| {
                let inten = intensities
                    .get(idx)
                    .copied()
                    .flatten()
                    .filter(|v| v.is_finite() && *v >= 0.0);
                (d, inten)
            })
            .collect();
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        let (d_spacings, intensities) = pairs.into_iter().unzip();
        Self {
            name: name.into(),
            d_spacings,
            intensities,
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
        let mut intensities = Vec::new();
        for (i, raw_line) in contents.lines().enumerate() {
            // Split inline comment (everything from the first `#`).
            let (line, comment) = match raw_line.split_once('#') {
                Some((before, after)) => (before, Some(after)),
                None => (raw_line, None),
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
                intensities.push(comment.and_then(parse_comment_intensity));
            }
        }
        if d_spacings.is_empty() {
            return Err(CalibrantError::Empty);
        }
        Ok(Self::from_d_spacings_and_intensities(
            name,
            d_spacings,
            intensities,
        ))
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

/// Extract an optional relative intensity from a pyFAI `.D` inline comment.
///
/// Standard format: `(h k l)  mult  intensity` (e.g. `(1 1 1)     8 100.0`).
/// Lines with only multiplicity (e.g. `(5 1 1)    32`) return `None`.
fn parse_comment_intensity(comment: &str) -> Option<f64> {
    let (_, after_hkl) = comment.split_once(')')?;
    let mut tokens = after_hkl.split_whitespace();
    let _mult = tokens.next()?;
    let intensity: f64 = tokens.next()?.parse().ok()?;
    if intensity.is_finite() && intensity >= 0.0 {
        Some(intensity)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_comment_intensity_handles_edge_cases() {
        assert_eq!(parse_comment_intensity(" (1 1 1)     8 100.0"), Some(100.0));
        assert_eq!(parse_comment_intensity(" (2 0 0) 6 27.73"), Some(27.73));
        // Multiplicity only, no intensity
        assert_eq!(parse_comment_intensity(" (5 1 1)    32"), None);
        // No parentheses
        assert_eq!(parse_comment_intensity(" 8 100.0"), None);
        // Negative or non-numeric intensity
        assert_eq!(parse_comment_intensity(" (1 0 0) 6 -5.0"), None);
        assert_eq!(parse_comment_intensity(" (1 0 0) 6 nan"), None);
        assert_eq!(parse_comment_intensity(" (1 0 0) 6 junk"), None);
    }

    #[test]
    fn from_d_spacings_filters_non_positive_and_non_finite() {
        let cal = Calibrant::from_d_spacings(
            "filtered",
            vec![1.5, -2.0, 0.0, f64::NAN, f64::INFINITY, 3.0],
        );
        assert_eq!(cal.d_spacings, vec![3.0, 1.5]);
        assert_eq!(cal.intensities, vec![None, None]);

        let cal2 = Calibrant::from_d_spacings_and_intensities(
            "paired",
            vec![1.5, -1.0, 3.0],
            vec![Some(20.0), Some(99.0), Some(100.0)],
        );
        assert_eq!(cal2.d_spacings, vec![3.0, 1.5]);
        assert_eq!(cal2.intensities, vec![Some(100.0), Some(20.0)]);
    }
}
