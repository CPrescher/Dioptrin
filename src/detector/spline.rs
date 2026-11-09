//! Fit2D cubic tensor-product B splines. No FITPACK/Python runtime required.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Surface {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub coefficients: Vec<f64>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Spline {
    pub region: [f64; 4],
    /// X then Y pixel size, in micrometres as in the Fit2D file.
    pub pixels: [f64; 2],
    pub grid: f64,
    pub x: Surface,
    pub y: Surface,
    /// Original content travels with projects and can be saved without its source.
    pub content: String,
}
impl Spline {
    pub fn parse(content: &str) -> Result<Self, String> {
        if content.len() > 16 * 1024 * 1024 || !content.is_ascii() {
            return Err("Invalid or oversized Fit2D spline.".into());
        }
        let lines: Vec<_> = content.lines().collect();
        let at = |label: &str| {
            lines
                .iter()
                .position(|l| l.trim().eq_ignore_ascii_case(label))
                .ok_or_else(|| format!("Missing spline {label}."))
        };
        let floats = |line: &str| -> Result<Vec<f64>, String> {
            if !line.len().is_multiple_of(14) {
                return Err("Invalid fixed-width spline numbers.".into());
            }
            line.as_bytes()
                .chunks(14)
                .map(|b| {
                    std::str::from_utf8(b)
                        .unwrap()
                        .trim()
                        .replace(['D', 'd'], "E")
                        .parse::<f64>()
                        .map_err(|_| "Invalid spline number.".into())
                })
                .collect()
        };
        let region = floats(
            lines
                .get(at("VALID REGION")? + 1)
                .ok_or("Missing spline region.")?,
        )?;
        let pixels = floats(
            lines
                .get(at("GRID SPACING, X-PIXEL SIZE, Y-PIXEL SIZE")? + 1)
                .ok_or("Missing spline pixels.")?,
        )?;
        let surface = |label: &str| -> Result<Surface, String> {
            let index = at(label)?;
            let n: Vec<usize> = lines
                .get(index + 1)
                .ok_or("Missing spline knot counts.")?
                .split_whitespace()
                .map(|s| s.parse().map_err(|_| "Invalid spline knot count."))
                .collect::<Result<_, _>>()?;
            if n.len() != 2 || n.iter().any(|n| *n < 8 || *n > 4096) {
                return Err("Invalid spline knot counts.".into());
            }
            let expected = n[0] + n[1] + (n[0] - 4) * (n[1] - 4);
            let mut data = Vec::with_capacity(expected);
            for line in lines
                .iter()
                .skip(index + 2)
                .take_while(|l| !l.trim().is_empty())
            {
                data.extend(floats(line.trim_end())?);
                if data.len() > expected {
                    return Err("Too many spline coefficients.".into());
                }
            }
            if data.len() != expected {
                return Err("Missing spline coefficients.".into());
            }
            // pyFAI reads knot vectors and coefficients as float32 before FITPACK.
            for value in &mut data {
                *value = (*value as f32) as f64;
            }
            Ok(Surface {
                x: data[..n[0]].to_vec(),
                y: data[n[0]..n[0] + n[1]].to_vec(),
                coefficients: data[n[0] + n[1]..].to_vec(),
            })
        };
        if region.len() != 4 || pixels.len() != 3 {
            return Err("Invalid spline header.".into());
        }
        let spline = Self {
            region: region.try_into().unwrap(),
            pixels: [pixels[1], pixels[2]],
            grid: pixels[0],
            x: surface("X-DISTORTION")?,
            y: surface("Y-DISTORTION")?,
            content: content.into(),
        };
        spline.validate()?;
        Ok(spline)
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.pixels.iter().any(|v| !v.is_finite() || *v <= 0.)
            || self.region.iter().any(|v| !v.is_finite())
            || self.region[2] <= self.region[0]
            || self.region[3] <= self.region[1]
            || !self.grid.is_finite()
            || self.grid <= 0.
        {
            return Err("Invalid spline region or pixel sizes.".into());
        }
        for s in [&self.x, &self.y] {
            if s.x.len() < 8
                || s.y.len() < 8
                || (s.x.len() - 4).checked_mul(s.y.len() - 4) != Some(s.coefficients.len())
                || s.x
                    .iter()
                    .chain(&s.y)
                    .chain(&s.coefficients)
                    .any(|v| !v.is_finite())
                || s.x.windows(2).chain(s.y.windows(2)).any(|w| w[0] > w[1])
                || s.x[3] >= s.x[s.x.len() - 4]
                || s.y[3] >= s.y[s.y.len() - 4]
            {
                return Err("Invalid cubic spline surface.".into());
            }
        }
        Ok(())
    }
    pub fn displacement(&self, x: f64, y: f64) -> (f64, f64) {
        (self.x.evaluate(x, y), self.y.evaluate(x, y))
    }
    /// Match pyFAI's finite-difference area correction, including boundary clamps.
    pub fn area(&self, row: f64, col: f64) -> f64 {
        (1. + self.x.evaluate(col + 0.5, row) - self.x.evaluate(col - 0.5, row))
            * (1. + self.y.evaluate(col, row + 0.5) - self.y.evaluate(col, row - 0.5))
    }
}
impl Surface {
    pub fn evaluate(&self, x: f64, y: f64) -> f64 {
        let (ix, bx) = basis(&self.x, x);
        let (iy, by) = basis(&self.y, y);
        let ny = self.y.len() - 4;
        let mut out = 0.;
        for (a, wx) in bx.iter().enumerate() {
            for (b, wy) in by.iter().enumerate() {
                out += wx * wy * self.coefficients[(ix + a) * ny + iy + b];
            }
        }
        out
    }
}
fn basis(t: &[f64], value: f64) -> (usize, [f64; 4]) {
    let n = t.len() - 4;
    let x = value.clamp(t[3], t[n]);
    let span = if x >= t[n] {
        n - 1
    } else {
        t.partition_point(|k| *k <= x)
            .saturating_sub(1)
            .clamp(3, n - 1)
    };
    let mut b = [1., 0., 0., 0.];
    let mut left = [0.; 4];
    let mut right = [0.; 4];
    for j in 1..=3 {
        left[j] = x - t[span + 1 - j];
        right[j] = t[span + j] - x;
        let mut saved = 0.;
        for r in 0..j {
            let den = right[r + 1] + left[j - r];
            let temp = if den == 0. { 0. } else { b[r] / den };
            b[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        b[j] = saved;
    }
    (span - 3, b)
}
