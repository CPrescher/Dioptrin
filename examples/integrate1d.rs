//! Example: 1D integration of a detector image
//!
//! This example demonstrates how to:
//! 1. Load a PONI calibration file
//! 2. Load a TIFF detector image
//! 3. Set up the geometry and integrator
//! 4. Perform 1D azimuthal integration
//!
//! Run with: cargo run --example integrate1d [image.tif] [calibration.poni]

use std::env;

use dioptrin::frame::Frame;
use dioptrin::geometry::Geometry;
use dioptrin::integration::{IntegrationMethod, Integrator, RadialUnit};
use dioptrin::poni::Poni;

const DEFAULT_IMAGE: &str = "test_data/CeO2_Pilatus1M.tif";
const DEFAULT_PONI: &str = "test_data/v1.poni";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = env::args().nth(1).unwrap_or_else(|| DEFAULT_IMAGE.to_string());
    let poni_path = env::args().nth(2).unwrap_or_else(|| DEFAULT_PONI.to_string());

    // Load the calibration parameters from a PONI file
    let poni = Poni::load(&poni_path)?;

    // Load the detector image from a TIFF file
    let frame = Frame::from_tiff(image_path)?;

    println!("Loaded frame: {} x {} pixels", frame.dim1, frame.dim2);

    // Create geometry from frame dimensions and calibration
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    // Create an integrator with the desired settings
    let mut integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::PixelSplit)
        .with_radial_unit(RadialUnit::TthDeg);

    // Perform 1D integration with 1000 radial bins
    let n_bins = 1000;
    let (radial, intensity) = integrator.integrate1d(&frame, n_bins);

    // Print some results
    println!("\n1D Integration Results:");
    println!("Number of bins: {}", radial.len());
    println!(
        "Radial range: {:.4} - {:.4} degrees",
        radial.first().unwrap_or(&0.0),
        radial.last().unwrap_or(&0.0)
    );

    // Print first few data points
    println!("\nFirst 10 data points:");
    println!("{:>12} {:>15}", "2θ (deg)", "Intensity");
    for i in 0..10.min(radial.len()) {
        println!("{:>12.4} {:>15.2}", radial[i], intensity[i]);
    }

    Ok(())
}
