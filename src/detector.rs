//! Physical detector coordinates, independent of image decoding.
//!
//! Coordinates are grid-boundary coordinates (pixel centres are i + 0.5),
//! positions are [z, slow, fast] in metres. Orientations follow pyFAI 0..4.
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};
pub mod spline;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum PixelMap {
    #[default]
    Uniform,
    /// Independent lower/upper edges allow module gaps and curved surfaces.
    Separable {
        slow: Vec<[[f64; 3]; 2]>,
        fast: Vec<[[f64; 3]; 2]>,
    },
    /// NeXus pixel corners, ordered A(low,low), B(high,low), C, D.
    Corners {
        shape: [usize; 2],
        values: Arc<Vec<[[f64; 3]; 4]>>,
    },
    Transformed {
        base: Arc<Detector>,
        operation: DetectorTransform,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum DetectorTransform {
    Identity,
    Half,
    Transpose,
    Transverse,
    Left,
    Right,
    Horizontal,
    Vertical,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Detector {
    pub name: String,
    pub aliases: Vec<String>,
    /// Factory options represented by this frozen physical mapping.
    #[serde(default)]
    pub config: serde_json::Value,
    pub pixel1: f64,
    pub pixel2: f64,
    /// Manufacturer-specific binning pitches (not always exact multiples).
    #[serde(default)]
    pub binning_pitches: Vec<(usize, f64)>,
    pub max_shape: Option<[usize; 2]>,
    pub orientation: u8,
    pub map: PixelMap,
    /// Half-open flattened intervals in the unbinned detector grid.
    pub mask: Vec<[usize; 2]>,
    #[serde(default)]
    pub circular_mask: bool,
    pub spline: Option<Arc<spline::Spline>>,
    /// Unsupported catalog models are never approximated as uniform pixels.
    pub unsupported: Option<String>,
}
impl Detector {
    pub fn is_contiguous(&self) -> bool {
        match &self.map {
            PixelMap::Uniform => true,
            PixelMap::Separable { slow, fast } => slow
                .windows(2)
                .chain(fast.windows(2))
                .all(|w| (0..3).all(|k| (w[0][1][k] - w[1][0][k]).abs() < 1e-14)),
            _ => false,
        }
    }
    pub fn custom(pixel1: f64, pixel2: f64) -> Self {
        Self {
            name: "Detector".into(),
            aliases: vec![],
            config: serde_json::Value::Null,
            pixel1,
            pixel2,
            binning_pitches: vec![],
            max_shape: None,
            orientation: 3,
            map: PixelMap::Uniform,
            mask: vec![],
            circular_mask: false,
            spline: None,
            unsupported: None,
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if !self.pixel1.is_finite()
            || self.pixel1 <= 0.
            || !self.pixel2.is_finite()
            || self.pixel2 <= 0.
        {
            return Err("Detector pixel sizes must be positive and finite.".into());
        }
        if self
            .binning_pitches
            .iter()
            .any(|(b, p)| *b == 0 || !p.is_finite() || *p <= 0.)
        {
            return Err("Invalid detector binning pitches.".into());
        }
        if self.orientation > 4 {
            return Err("Detector orientations 5–8 are not supported by pyFAI's physical detector model or this importer.".into());
        }
        if let Some(reason) = &self.unsupported {
            return Err(reason.clone());
        }
        if self
            .max_shape
            .is_some_and(|s| s.contains(&0) || s[0].checked_mul(s[1]).is_none())
        {
            return Err("Invalid detector shape.".into());
        }
        match &self.map {
            PixelMap::Separable { slow, fast } => {
                if Some([slow.len(), fast.len()]) != self.max_shape
                    || slow
                        .iter()
                        .chain(fast)
                        .flatten()
                        .flatten()
                        .any(|v| !v.is_finite())
                {
                    return Err("Invalid separable detector map.".into());
                }
            }
            PixelMap::Corners { shape, values } => {
                if Some(*shape) != self.max_shape
                    || shape[0].checked_mul(shape[1]) != Some(values.len())
                    || values.iter().flatten().flatten().any(|v| !v.is_finite())
                {
                    return Err("Invalid detector pixel corners.".into());
                }
            }
            PixelMap::Transformed { base, .. } => base.validate()?,
            _ => {}
        }
        if let Some(s) = &self.spline {
            s.validate()?;
        }
        if self
            .mask
            .iter()
            .any(|[a, b]| a >= b || self.max_shape.is_none_or(|s| *b > s[0] * s[1]))
        {
            return Err("Invalid detector mask intervals.".into());
        }
        Ok(())
    }
    /// Integer binning only. Never guess crops or silently reshape masks.
    pub fn binning(&self, shape: [usize; 2]) -> Result<[usize; 2], String> {
        if shape.contains(&0) || self.max_shape.is_some_and(|s| s.contains(&0)) {
            return Err("Invalid detector shape.".into());
        }
        if let Some(reason) = &self.unsupported {
            return Err(reason.clone());
        }
        let Some(max) = self.max_shape else {
            return Ok([1, 1]);
        };
        if shape.contains(&0) || max[0] % shape[0] != 0 || max[1] % shape[1] != 0 {
            return Err(format!(
                "Detector {} has shape {max:?}; image {shape:?} is not an integer binning. Select Custom for a cropped image.",
                self.name
            ));
        }
        let bin = [max[0] / shape[0], max[1] / shape[1]];
        if !self.binning_pitches.is_empty()
            && bin
                .iter()
                .any(|b| !self.binning_pitches.iter().any(|(k, _)| k == b))
        {
            return Err("Detector binning has no manufacturer pixel-pitch definition.".into());
        }
        Ok(bin)
    }
    pub fn effective_pixels(&self, shape: [usize; 2]) -> [f64; 2] {
        let bin = self
            .max_shape
            .map(|s| [s[0] / shape[0], s[1] / shape[1]])
            .unwrap_or([1, 1]);
        let pitch = |b: usize, p: f64| {
            self.binning_pitches
                .iter()
                .find(|(k, _)| *k == b)
                .map(|(_, p)| *p)
                .unwrap_or(p * b as f64)
        };
        [pitch(bin[0], self.pixel1), pitch(bin[1], self.pixel2)]
    }
    fn oriented(&self, shape: [usize; 2], row: f64, col: f64) -> (f64, f64) {
        (
            if matches!(self.orientation, 1 | 2) {
                shape[0] as f64 - row
            } else {
                row
            },
            if matches!(self.orientation, 1 | 4) {
                shape[1] as f64 - col
            } else {
                col
            },
        )
    }
    pub fn position(&self, shape: [usize; 2], row: f64, col: f64) -> [f64; 3] {
        let logical = [row, col];
        let (mut row, mut col) = self.oriented(shape, row, col);
        let bin = self
            .max_shape
            .map(|s| [s[0] / shape[0], s[1] / shape[1]])
            .unwrap_or([1, 1]);
        if bin != [1, 1] && !matches!(self.map, PixelMap::Uniform) && self.spline.is_none() {
            // corners handles orientation itself.
            let (unoriented_r, unoriented_c) = (logical[0], logical[1]);
            let ur = (unoriented_r.floor().max(0.) as usize).min(shape[0] - 1);
            let uc = (unoriented_c.floor().max(0.) as usize).min(shape[1] - 1);
            return interpolate(
                &self.corners(shape, ur, uc),
                unoriented_r - ur as f64,
                unoriented_c - uc as f64,
            );
        }
        row *= bin[0] as f64;
        col *= bin[1] as f64;
        if let Some(s) = &self.spline {
            let (dx, dy) = s.displacement(col, row);
            return [0., (row + dy) * self.pixel1, (col + dx) * self.pixel2];
        }
        match &self.map {
            PixelMap::Uniform => {
                let p = self.effective_pixels(shape);
                [0., row / bin[0] as f64 * p[0], col / bin[1] as f64 * p[1]]
            }
            PixelMap::Separable { slow, fast } => {
                let a = edge(slow, row);
                let b = edge(fast, col);
                std::array::from_fn(|k| a[k] + b[k])
            }
            PixelMap::Corners { shape, values } => {
                let r = (row.floor().max(0.) as usize).min(shape[0] - 1);
                let c = (col.floor().max(0.) as usize).min(shape[1] - 1);
                interpolate(&values[r * shape[1] + c], row - r as f64, col - c as f64)
            }
            PixelMap::Transformed { base, operation } => {
                let full = self.max_shape.unwrap_or(shape);
                let r = (row.floor().max(0.) as usize).min(full[0] - 1);
                let c = (col.floor().max(0.) as usize).min(full[1] - 1);
                let (original, rr, cc) = inverse_pixel(*operation, full, r, c);
                base.position(
                    original,
                    rr as f64 + row - r as f64,
                    cc as f64 + col - c as f64,
                )
            }
        }
    }
    /// Physical quadrilateral of one logical pixel, retaining both sides of gaps.
    pub fn corners(&self, shape: [usize; 2], row: usize, col: usize) -> [[f64; 3]; 4] {
        // Evaluate just inside the pixel to choose its own edge at discontinuities.
        let (r, c) = self.oriented(shape, row as f64 + 0.5, col as f64 + 0.5);
        if let PixelMap::Transformed { base, operation } = &self.map {
            let (original, r, c) =
                inverse_pixel(*operation, shape, r.floor() as usize, c.floor() as usize);
            return base.corners(original, r, c);
        }
        let bin = self
            .max_shape
            .map(|s| [s[0] / shape[0], s[1] / shape[1]])
            .unwrap_or([1, 1]);
        let rr = (r.floor() as usize) * bin[0];
        let cc = (c.floor() as usize) * bin[1];
        let sample = |r: usize, c: usize, dr: f64, dc: f64| match &self.map {
            PixelMap::Separable { slow, fast } => std::array::from_fn(|k| {
                slow[r][usize::from(dr > 0.)][k] + fast[c][usize::from(dc > 0.)][k]
            }),
            PixelMap::Corners { shape, values } => interpolate(&values[r * shape[1] + c], dr, dc),
            _ => self.position(shape, row as f64 + dr, col as f64 + dc),
        };
        if matches!(self.map, PixelMap::Uniform) || self.spline.is_some() {
            return [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
                .map(|[a, b]| self.position(shape, row as f64 + a, col as f64 + b));
        }
        let lowr = rr;
        let highr = rr + bin[0] - 1;
        let lowc = cc;
        let highc = cc + bin[1] - 1;
        let mut v = [
            sample(lowr, lowc, 0., 0.),
            sample(highr, lowc, 1., 0.),
            sample(highr, highc, 1., 1.),
            sample(lowr, highc, 0., 1.),
        ];
        if matches!(self.orientation, 1 | 2) {
            v.swap(0, 1);
            v.swap(3, 2);
        }
        if matches!(self.orientation, 1 | 4) {
            v.swap(0, 3);
            v.swap(1, 2);
        }
        v
    }
    pub fn builtin_mask(&self, shape: [usize; 2]) -> Result<Vec<u8>, String> {
        self.validate()?;
        let bin = self.binning(shape)?;
        let mut mask = vec![0; shape[0] * shape[1]];
        if self.circular_mask {
            let cy = (shape[0] / 2) as f64;
            let cx = (shape[1] / 2) as f64;
            for r in 0..shape[0] {
                for c in 0..shape[1] {
                    let (y, x) = self.oriented(shape, r as f64 + 0.5, c as f64 + 0.5);
                    mask[r * shape[1] + c] =
                        u8::from((y - cy).powi(2) + (x - cx).powi(2) > cy.powi(2));
                }
            }
            return Ok(mask);
        }
        let Some(max) = self.max_shape else {
            return Ok(mask);
        };
        for [start, end] in &self.mask {
            for i in *start..*end {
                let mut r = (i / max[1]) / bin[0];
                let mut c = (i % max[1]) / bin[1];
                if matches!(self.orientation, 1 | 2) {
                    r = shape[0] - 1 - r;
                }
                if matches!(self.orientation, 1 | 4) {
                    c = shape[1] - 1 - c;
                }
                mask[r * shape[1] + c] = 1;
            }
        }
        Ok(mask)
    }
    pub fn has_spline(&self) -> bool {
        self.spline.is_some()
            || matches!(&self.map,PixelMap::Transformed{base,..} if base.has_spline())
    }
    pub fn clear_spline(&mut self) {
        self.spline = None;
        if let PixelMap::Transformed { base, .. } = &mut self.map {
            Arc::make_mut(base).clear_spline();
            if matches!(base.map, PixelMap::Uniform) {
                self.map = PixelMap::Uniform;
            }
        }
    }
    pub fn is_flat(&self) -> bool {
        match &self.map {
            PixelMap::Uniform => true,
            PixelMap::Separable { slow, fast } => {
                slow.iter().chain(fast).flatten().all(|p| p[0] == 0.)
            }
            PixelMap::Corners { values, .. } => values.iter().flatten().all(|p| p[0] == 0.),
            PixelMap::Transformed { base, .. } => base.is_flat(),
        }
    }
    pub fn spline_area(&self, shape: [usize; 2], row: f64, col: f64) -> f64 {
        if let PixelMap::Transformed { base, operation } = &self.map {
            let (original, r, c) = inverse_pixel(
                *operation,
                shape,
                row.floor() as usize,
                col.floor() as usize,
            );
            return base.spline_area(original, r as f64 + row.fract(), c as f64 + col.fract());
        }
        let Some(s) = &self.spline else { return 1. };
        // pyFAI evaluates the spline-area finite differences in image-index
        // order, independently of the orientation used for physical positions.
        let (r, c) = (row, col);
        let bin = self
            .max_shape
            .map(|m| [m[0] / shape[0], m[1] / shape[1]])
            .unwrap_or([1, 1]);
        let (br, bc) = (bin[0] as f64, bin[1] as f64);
        let (r, c) = (r * br, c * bc);
        (1. + (s.x.evaluate(c + bc * 0.5, r) - s.x.evaluate(c - bc * 0.5, r)) / bc)
            * (1. + (s.y.evaluate(c, r + br * 0.5) - s.y.evaluate(c, r - br * 0.5)) / br)
    }
    /// Match Dioptas' image transformations: permute cached physical pixels;
    /// regular uniform detectors retain their pitch-based geometry convention.
    pub fn transformed(&self, operation: DetectorTransform) -> Result<Self, String> {
        self.validate()?;
        if operation == DetectorTransform::Identity {
            return Ok(self.clone());
        }
        let mut result = self.clone();
        if !matches!(self.map, PixelMap::Uniform) || self.spline.is_some() {
            let (base, combined) = match &self.map {
                PixelMap::Transformed {
                    base,
                    operation: previous,
                } => (base.clone(), compose(*previous, operation)),
                _ => (Arc::new(self.clone()), operation),
            };
            if combined == DetectorTransform::Identity {
                result.map = base.map.clone();
                result.spline = base.spline.clone();
                result.orientation = base.orientation;
            } else {
                result.map = PixelMap::Transformed {
                    base,
                    operation: combined,
                };
                result.spline = None;
                result.orientation = 3;
            }
        }
        if let Some(mut shape) = self.max_shape {
            let old = self.builtin_mask(shape)?;
            if swaps_axes(operation) {
                shape.swap(0, 1);
            }
            let mut start = None;
            let mut intervals = vec![];
            for i in 0..=shape[0] * shape[1] {
                let value = if i == shape[0] * shape[1] {
                    0
                } else {
                    let (r, c) = result.oriented(
                        shape,
                        (i / shape[1]) as f64 + 0.5,
                        (i % shape[1]) as f64 + 0.5,
                    );
                    let (s, r, c) =
                        inverse_pixel(operation, shape, r.floor() as usize, c.floor() as usize);
                    old[r * s[1] + c]
                };
                if value != 0 {
                    start.get_or_insert(i);
                } else if let Some(a) = start.take() {
                    intervals.push([a, i]);
                }
            }
            result.mask = intervals;
            result.circular_mask =
                self.circular_mask && shape[0] == shape[1] && shape[0].is_multiple_of(2);
            if result.circular_mask {
                result.mask.clear();
            }
            result.max_shape = Some(shape);
        }
        if swaps_axes(operation) {
            std::mem::swap(&mut result.pixel1, &mut result.pixel2);
        }
        Ok(result)
    }
}
fn inverse_pixel(
    op: DetectorTransform,
    shape: [usize; 2],
    r: usize,
    c: usize,
) -> ([usize; 2], usize, usize) {
    let [h, w] = shape;
    match op {
        DetectorTransform::Identity => (shape, r, c),
        DetectorTransform::Half => (shape, h - 1 - r, w - 1 - c),
        DetectorTransform::Transpose => ([w, h], c, r),
        DetectorTransform::Transverse => ([w, h], w - 1 - c, h - 1 - r),
        DetectorTransform::Left => ([w, h], c, h - 1 - r),
        DetectorTransform::Right => ([w, h], w - 1 - c, r),
        DetectorTransform::Horizontal => (shape, r, w - 1 - c),
        DetectorTransform::Vertical => (shape, h - 1 - r, c),
    }
}
fn swaps_axes(op: DetectorTransform) -> bool {
    matches!(
        op,
        DetectorTransform::Left
            | DetectorTransform::Right
            | DetectorTransform::Transpose
            | DetectorTransform::Transverse
    )
}
fn compose(before: DetectorTransform, after: DetectorTransform) -> DetectorTransform {
    use DetectorTransform::*;
    let mut middle = [3, 5];
    if swaps_axes(before) {
        middle.swap(0, 1)
    }
    let mut output = middle;
    if swaps_axes(after) {
        output.swap(0, 1)
    }
    for candidate in [
        Identity, Half, Left, Right, Horizontal, Vertical, Transpose, Transverse,
    ] {
        let same = [(0, 0), (output[0] - 1, 0), (0, output[1] - 1)]
            .iter()
            .all(|&(r, c)| {
                let (size, r1, c1) = inverse_pixel(after, output, r, c);
                let expected = inverse_pixel(before, size, r1, c1);
                inverse_pixel(candidate, output, r, c) == expected
            });
        if same {
            return candidate;
        }
    }
    unreachable!("dihedral detector permutation")
}
fn edge(values: &[[[f64; 3]; 2]], p: f64) -> [f64; 3] {
    let i = (p.floor().max(0.) as usize).min(values.len() - 1);
    let d = p - i as f64;
    std::array::from_fn(|k| values[i][0][k] * (1. - d) + values[i][1][k] * d)
}
fn interpolate(v: &[[f64; 3]; 4], r: f64, c: f64) -> [f64; 3] {
    std::array::from_fn(|k| {
        v[0][k] * (1. - r) * (1. - c)
            + v[1][k] * r * (1. - c)
            + v[2][k] * r * c
            + v[3][k] * (1. - r) * c
    })
}
pub fn catalog() -> &'static [Detector] {
    static CATALOG: OnceLock<Vec<Detector>> = OnceLock::new();
    CATALOG.get_or_init(|| {
        use std::io::Read;
        let mut data = String::new();
        flate2::read::GzDecoder::new(&include_bytes!("detector/catalog.json.gz")[..])
            .take(32 * 1024 * 1024)
            .read_to_string(&mut data)
            .expect("bounded detector catalogue");
        serde_json::from_str(&data).expect("validated detector catalog")
    })
}
pub fn named(name: &str) -> Result<Detector, String> {
    let norm = |s: &str| s.to_lowercase().replace([' ', '_', '-'], "");
    catalog()
        .iter()
        .find(|d| norm(&d.name) == norm(name) || d.aliases.iter().any(|a| norm(a) == norm(name)))
        .cloned()
        .ok_or_else(|| format!("Unknown detector {name}."))
}

#[derive(Debug)]
pub struct Identification {
    pub selected: Option<Detector>,
    pub candidates: Vec<String>,
    pub reason: String,
}
/// Conservative first-load identification. Metadata never changes image samples.
pub fn identify(shape: [usize; 2], pixels: Option<[f64; 2]>, description: &str) -> Identification {
    let norm = |s: &str| {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    let text = norm(description);
    let mut identities: Vec<_> = catalog()
        .iter()
        .filter(|d| d.unsupported.is_none())
        .filter_map(|d| {
            let score = std::iter::once(&d.name)
                .chain(&d.aliases)
                .map(|s| norm(s))
                .filter(|s| s.len() >= 5 && text.contains(s))
                .map(|s| s.len())
                .max()?;
            Some((score, d))
        })
        .collect();
    let best = identities.iter().map(|(s, _)| *s).max();
    identities.retain(|(s, _)| Some(*s) == best);
    let explicit = !identities.is_empty();
    let pool: Vec<_> = if explicit {
        identities.into_iter().map(|(_, d)| d).collect()
    } else {
        catalog()
            .iter()
            .filter(|d| d.unsupported.is_none() && d.max_shape == Some(shape))
            .collect()
    };
    let valid: Vec<_> = pool
        .into_iter()
        .filter(|d| {
            d.binning(shape).is_ok_and(|_bin| {
                pixels.is_none_or(|p| {
                    d.effective_pixels(shape)
                        .iter()
                        .zip(p)
                        .all(|(a, b)| b.is_finite() && b > 0. && ((a - b) / a).abs() < 1e-4)
                })
            })
        })
        .collect();
    let candidates = valid.iter().map(|d| d.name.clone()).collect();
    let selected = if valid.len() == 1 {
        Some(valid[0].clone())
    } else {
        None
    };
    let reason = if selected.is_some() {
        if explicit {
            "Matched detector metadata"
        } else if pixels.is_some() {
            "Matched shape and pixel pitch"
        } else {
            "Unique detector shape"
        }
    } else if valid.len() > 1 {
        "Several detector definitions match; choose a detector"
    } else if explicit {
        "Detector metadata conflicts with image shape or pixel pitch; kept Custom"
    } else {
        "No unambiguous detector definition; kept Custom"
    };
    Identification {
        selected,
        candidates,
        reason: reason.into(),
    }
}
