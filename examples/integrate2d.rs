//! Example: 2D integration of a detector image
//!
//! This example demonstrates how to:
//! 1. Load a PONI calibration file
//! 2. Load a TIFF detector image
//! 3. Set up the geometry and integrator
//! 4. Perform 2D integration (radial and azimuthal)
//!
//! Run with: cargo run --example integrate2d [image.tif] [calibration.poni]

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
    let integrator = Integrator::new(geometry)
        .with_method(IntegrationMethod::PixelSplit)
        .with_radial_unit(RadialUnit::TthDeg);

    // Perform 2D integration
    let n_radial = 100;
    let n_azimuthal = 36; // 10 degree bins
    let result = integrator.integrate2d(&frame, n_radial, n_azimuthal);

    // Print results
    println!("\n2D Integration Results:");
    println!("Radial bins: {}", n_radial);
    println!("Azimuthal bins: {}", n_azimuthal);
    println!("Total output size: {}", result.len()); 

    // The result is a flattened 2D array of shape (n_azimuthal, n_radial)
    // Print a summary of the first azimuthal slice
    println!("\nFirst azimuthal slice (first 10 radial bins):");
    for i in 0..10.min(n_radial) {
        println!("  Radial bin {}: {:.2}", i, result[i]);
    }

    Ok(())
}
