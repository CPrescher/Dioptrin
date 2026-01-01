use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use rayon::ThreadPoolBuilder;

use crate::frame::Frame;
use crate::geometry::Geometry;
use crate::integration::{IntegrationMethod, Integrator};
use crate::poni::Poni;

fn parse_method(method: Option<&str>, n_ss: Option<usize>) -> Result<IntegrationMethod, PyErr> {
    match method.unwrap_or("pixel_split").to_lowercase().as_str() {
        "pixel_split" | "pixelsplit" | "ps" => Ok(IntegrationMethod::PixelSplit),
        "counting" | "count" => Ok(IntegrationMethod::Counting),
        "histogram" | "hist" => Ok(IntegrationMethod::Histogram),
        "supersampled" | "supersample" | "ss" => {
            let n_ss = n_ss.unwrap_or(5);
            Ok(IntegrationMethod::Supersampled { n_ss })
        }
        other => Err(PyValueError::new_err(format!(
            "Unknown integration method: {}",
            other
        ))),
    }
}

#[pyfunction]
#[pyo3(
    text_signature = "(poni_path, tiff_path, n_points, method=None, polarization_factor=None, n_ss=None)"
)]
fn integrate1d(
    poni_path: &str,
    tiff_path: &str,
    n_points: usize,
    method: Option<&str>,
    polarization_factor: Option<f64>,
    n_ss: Option<usize>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let poni = Poni::load(poni_path)
        .map_err(|err| PyValueError::new_err(format!("Failed to load PONI: {}", err)))?;
    let frame = Frame::from_tiff(tiff_path.to_string())
        .map_err(|err| PyValueError::new_err(format!("Failed to load TIFF: {}", err)))?;
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let mut integrator = Integrator::new(geometry).with_method(parse_method(method, n_ss)?);
    if let Some(polarization_factor) = polarization_factor {
        integrator = integrator.with_polarization_factor(polarization_factor);
    }

    Ok(integrator.integrate1d(&frame, n_points))
}

#[pyfunction]
#[pyo3(
    text_signature = "(poni_path, tiff_path, n_points, method=None, polarization_factor=None, n_ss=None)"
)]
fn integrate1d_with_errors(
    poni_path: &str,
    tiff_path: &str,
    n_points: usize,
    method: Option<&str>,
    polarization_factor: Option<f64>,
    n_ss: Option<usize>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let poni = Poni::load(poni_path)
        .map_err(|err| PyValueError::new_err(format!("Failed to load PONI: {}", err)))?;
    let frame = Frame::from_tiff(tiff_path.to_string())
        .map_err(|err| PyValueError::new_err(format!("Failed to load TIFF: {}", err)))?;
    let geometry = Geometry::new(frame.dim1, frame.dim2, poni);

    let mut integrator = Integrator::new(geometry).with_method(parse_method(method, n_ss)?);
    if let Some(polarization_factor) = polarization_factor {
        integrator = integrator.with_polarization_factor(polarization_factor);
    }

    Ok(integrator.integrate1d_with_errors(&frame, n_points))
}

#[pyfunction]
fn rayon_num_threads() -> usize {
    rayon::current_num_threads()
}

#[pyfunction]
#[pyo3(text_signature = "(threads)")]
fn set_rayon_threads(threads: usize) -> PyResult<()> {
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|err| PyValueError::new_err(format!("Failed to set Rayon threads: {}", err)))
}

#[pyclass(name = "Integrator", unsendable)]
struct PyIntegrator {
    integrator: Integrator,
    dim1: usize,
    dim2: usize,
}

#[pymethods]
impl PyIntegrator {
    #[new]
    #[pyo3(signature = (poni_path, dim1, dim2, method=None, polarization_factor=None, n_ss=None))]
    fn new(
        poni_path: &str,
        dim1: usize,
        dim2: usize,
        method: Option<&str>,
        polarization_factor: Option<f64>,
        n_ss: Option<usize>,
    ) -> PyResult<Self> {
        let poni = Poni::load(poni_path)
            .map_err(|err| PyValueError::new_err(format!("Failed to load PONI: {}", err)))?;
        let geometry = Geometry::new(dim1, dim2, poni);
        let mut integrator = Integrator::new(geometry).with_method(parse_method(method, n_ss)?);
        if let Some(polarization_factor) = polarization_factor {
            integrator = integrator.with_polarization_factor(polarization_factor);
        }

        Ok(Self {
            integrator,
            dim1,
            dim2,
        })
    }

    #[classmethod]
    #[pyo3(signature = (poni_path, tiff_path, method=None, polarization_factor=None, n_ss=None))]
    fn from_tiff(
        _cls: &Bound<'_, PyType>,
        poni_path: &str,
        tiff_path: &str,
        method: Option<&str>,
        polarization_factor: Option<f64>,
        n_ss: Option<usize>,
    ) -> PyResult<Self> {
        let poni = Poni::load(poni_path)
            .map_err(|err| PyValueError::new_err(format!("Failed to load PONI: {}", err)))?;
        let frame = Frame::from_tiff(tiff_path.to_string())
            .map_err(|err| PyValueError::new_err(format!("Failed to load TIFF: {}", err)))?;
        let geometry = Geometry::new(frame.dim1, frame.dim2, poni);
        let mut integrator = Integrator::new(geometry).with_method(parse_method(method, n_ss)?);
        if let Some(polarization_factor) = polarization_factor {
            integrator = integrator.with_polarization_factor(polarization_factor);
        }

        Ok(Self {
            integrator,
            dim1: frame.dim1,
            dim2: frame.dim2,
        })
    }

    #[pyo3(signature = (tiff_path, n_points))]
    fn integrate1d(
        &mut self,
        py: Python<'_>,
        tiff_path: &str,
        n_points: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let frame = Frame::from_tiff(tiff_path.to_string())
            .map_err(|err| PyValueError::new_err(format!("Failed to load TIFF: {}", err)))?;
        if frame.dim1 != self.dim1 || frame.dim2 != self.dim2 {
            return Err(PyValueError::new_err(format!(
                "Frame dimensions {}x{} do not match integrator dimensions {}x{}",
                frame.dim1, frame.dim2, self.dim1, self.dim2
            )));
        }
        Ok(py.allow_threads(|| self.integrator.integrate1d(&frame, n_points)))
    }

    #[pyo3(signature = (array, n_points))]
    fn integrate1d_array(
        &mut self,
        py: Python<'_>,
        array: PyReadonlyArray2<f64>,
        n_points: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let view = array.as_array();
        if !view.is_standard_layout() {
            return Err(PyValueError::new_err(
                "Array must be C-contiguous (row-major) and float64",
            ));
        }
        let shape = view.shape();
        let dim1 = shape[0];
        let dim2 = shape[1];

        if dim1 != self.dim1 || dim2 != self.dim2 {
            return Err(PyValueError::new_err(format!(
                "Array dimensions {}x{} do not match integrator dimensions {}x{}",
                dim1, dim2, self.dim1, self.dim2
            )));
        }

        let slice = array.as_slice().map_err(|err| {
            PyValueError::new_err(format!("Array must be contiguous: {}", err))
        })?;
        Ok(py.allow_threads(|| {
            self.integrator
                .integrate1d_from_slice(slice, dim1, dim2, n_points)
        }))
    }

    #[pyo3(signature = (tiff_path, n_points))]
    fn integrate1d_with_errors(
        &mut self,
        py: Python<'_>,
        tiff_path: &str,
        n_points: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let frame = Frame::from_tiff(tiff_path.to_string())
            .map_err(|err| PyValueError::new_err(format!("Failed to load TIFF: {}", err)))?;
        if frame.dim1 != self.dim1 || frame.dim2 != self.dim2 {
            return Err(PyValueError::new_err(format!(
                "Frame dimensions {}x{} do not match integrator dimensions {}x{}",
                frame.dim1, frame.dim2, self.dim1, self.dim2
            )));
        }
        Ok(py.allow_threads(|| {
            self.integrator.integrate1d_with_errors(&frame, n_points)
        }))
    }

    #[pyo3(signature = (method, n_ss=None))]
    fn set_method(&mut self, method: &str, n_ss: Option<usize>) -> PyResult<()> {
        self.integrator
            .set_method(parse_method(Some(method), n_ss)?);
        Ok(())
    }

    fn set_polarization_factor(&mut self, polarization_factor: Option<f64>) {
        self.integrator
            .set_polarization_factor(polarization_factor);
    }

    fn dimensions(&self) -> (usize, usize) {
        (self.dim1, self.dim2)
    }
}

#[pymodule]
fn dioptrin(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(integrate1d, m)?)?;
    m.add_function(wrap_pyfunction!(integrate1d_with_errors, m)?)?;
    m.add_function(wrap_pyfunction!(rayon_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_rayon_threads, m)?)?;
    m.add_class::<PyIntegrator>()?;
    Ok(())
}
