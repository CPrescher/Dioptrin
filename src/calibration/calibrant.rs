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

const BUILTIN_CALIBRANTS: &[(&str, &str)] = &[
    ("AgBh", include_str!("../../calibrants/AgBh.D")),
    ("Al", include_str!("../../calibrants/Al.D")),
    ("Au", include_str!("../../calibrants/Au.D")),
    ("C14H30O", include_str!("../../calibrants/C14H30O.D")),
    ("CeO2", include_str!("../../calibrants/CeO2.D")),
    ("Cr2O3", include_str!("../../calibrants/Cr2O3.D")),
    ("CrOx", include_str!("../../calibrants/CrOx.D")),
    ("CuO", include_str!("../../calibrants/CuO.D")),
    ("LaB6", include_str!("../../calibrants/LaB6.D")),
    (
        "LaB6_SRM660a",
        include_str!("../../calibrants/LaB6_SRM660a.D"),
    ),
    (
        "LaB6_SRM660b",
        include_str!("../../calibrants/LaB6_SRM660b.D"),
    ),
    (
        "LaB6_SRM660c",
        include_str!("../../calibrants/LaB6_SRM660c.D"),
    ),
    ("NaCl", include_str!("../../calibrants/NaCl.D")),
    ("Ni", include_str!("../../calibrants/Ni.D")),
    ("PBBA", include_str!("../../calibrants/PBBA.D")),
    ("Si", include_str!("../../calibrants/Si.D")),
    ("Si_SRM640", include_str!("../../calibrants/Si_SRM640.D")),
    ("Si_SRM640a", include_str!("../../calibrants/Si_SRM640a.D")),
    ("Si_SRM640b", include_str!("../../calibrants/Si_SRM640b.D")),
    ("Si_SRM640c", include_str!("../../calibrants/Si_SRM640c.D")),
    ("Si_SRM640d", include_str!("../../calibrants/Si_SRM640d.D")),
    ("Si_SRM640e", include_str!("../../calibrants/Si_SRM640e.D")),
    ("TiO2", include_str!("../../calibrants/TiO2.D")),
    ("ZnO", include_str!("../../calibrants/ZnO.D")),
    (
        "alpha_Al2O3",
        include_str!("../../calibrants/alpha_Al2O3.D"),
    ),
    (
        "cristobaltite",
        include_str!("../../calibrants/cristobaltite.D"),
    ),
    ("quartz", include_str!("../../calibrants/quartz.D")),
];

const BUILTIN_NAMES: &[&str] = &[
    "AgBh",
    "Al",
    "Au",
    "C14H30O",
    "CeO2",
    "Cr2O3",
    "CrOx",
    "CuO",
    "LaB6",
    "LaB6_SRM660a",
    "LaB6_SRM660b",
    "LaB6_SRM660c",
    "NaCl",
    "Ni",
    "PBBA",
    "Si",
    "Si_SRM640",
    "Si_SRM640a",
    "Si_SRM640b",
    "Si_SRM640c",
    "Si_SRM640d",
    "Si_SRM640e",
    "TiO2",
    "ZnO",
    "alpha_Al2O3",
    "cristobaltite",
    "quartz",
];

#[derive(Debug)]
#[non_exhaustive]
pub enum CalibrantError {
    Io(std::io::Error),
    /// File parsed but contained no d-spacing entries.
    Empty,
    /// A data line could not be parsed as a number.
    Parse {
        line: usize,
        content: String,
    },
    /// No built-in calibrant matched the requested name.
    UnknownBuiltin(String),
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
            CalibrantError::UnknownBuiltin(name) => {
                write!(f, "unknown built-in calibrant: {:?}", name)
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

fn find_builtin(name: &str) -> Option<(&'static str, &'static str)> {
    BUILTIN_CALIBRANTS
        .iter()
        .find(|(n, _)| *n == name)
        .or_else(|| {
            BUILTIN_CALIBRANTS
                .iter()
                .find(|(n, _)| n.eq_ignore_ascii_case(name))
        })
        .copied()
}

impl Calibrant {
    /// Names of all built-in calibrants bundled with Dioptrin.
    pub fn builtin_names() -> &'static [&'static str] {
        BUILTIN_NAMES
    }

    /// Raw `.D` contents of a built-in calibrant (exact match, then ASCII
    /// case-insensitive fallback).
    pub fn builtin_str(name: &str) -> Option<&'static str> {
        find_builtin(name).map(|(_, contents)| contents)
    }

    /// Load a built-in calibrant by name (e.g. `"CeO2"`, `"LaB6"`, `"Si"`).
    ///
    /// Matches exact names first, then falls back to ASCII case-insensitive
    /// matching.
    pub fn builtin(name: &str) -> Result<Self, CalibrantError> {
        let (canonical, contents) =
            find_builtin(name).ok_or_else(|| CalibrantError::UnknownBuiltin(name.to_string()))?;
        Self::from_d_str(canonical, contents)
    }

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
        Some(if intensity == 0.0 { 0.0 } else { intensity })
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

    #[test]
    fn all_builtin_calibrants_parse_and_inorganic_ones_have_intensities_and_high_q() {
        let names = Calibrant::builtin_names();
        assert_eq!(names.len(), 27);
        assert_eq!(
            names,
            &BUILTIN_CALIBRANTS
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()[..]
        );
        for &name in names {
            let cal = Calibrant::builtin(name)
                .unwrap_or_else(|e| panic!("failed to load built-in {name}: {e}"));
            assert_eq!(cal.name, name);
            assert!(cal.n_rings() > 0, "{name} should have rings");
            assert!(Calibrant::builtin_str(name).is_some());
            assert!(
                cal.intensities
                    .iter()
                    .flatten()
                    .all(|v| !v.is_sign_negative()),
                "{name} contains negative zero or negative intensity"
            );
        }
        // Case-insensitive lookup returns canonical name.
        let ceo2 = Calibrant::builtin("ceo2").unwrap();
        assert_eq!(ceo2.name, "CeO2");
        assert!(ceo2.n_rings() > 150, "CeO2 should extend down to ~0.25 A");
        assert!(ceo2.d_spacings.last().copied().unwrap() < 0.26);

        // Unknown name returns UnknownBuiltin error.
        assert!(matches!(
            Calibrant::builtin("not_a_calibrant"),
            Err(CalibrantError::UnknownBuiltin(_))
        ));

        // All enriched inorganic calibrants must have parsed intensities on every line.
        for name in [
            "CeO2",
            "LaB6",
            "LaB6_SRM660c",
            "Si",
            "Si_SRM640e",
            "Al",
            "Au",
            "Ni",
            "NaCl",
            "TiO2",
            "ZnO",
            "alpha_Al2O3",
            "Cr2O3",
            "quartz",
        ] {
            let cal = Calibrant::builtin(name).unwrap();
            assert!(
                cal.intensities.iter().all(|v| v.is_some()),
                "{name} has lines with missing intensity"
            );
        }
    }
}
