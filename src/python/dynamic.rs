//! Sendable handles to a shared immutable plan, with one compute pool per worker.
use super::results::PyIntegrationResult;
use crate::integration::Dynamic1dPlan;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass(name = "Dynamic1dPlan", frozen)]
pub struct PyDynamic1dPlan {
    pub(crate) plan: Arc<Dynamic1dPlan>,
}

#[pyclass(name = "Dynamic1dWorker", frozen)]
pub struct PyDynamic1dWorker {
    plan: Arc<Dynamic1dPlan>,
    pool: rayon::ThreadPool,
}

#[pymethods]
impl PyDynamic1dPlan {
    /// Construct a light worker sharing this plan's geometry and float64 tables.
    /// Default: one compute thread, for application-level frame parallelism.
    #[pyo3(signature = (num_threads=1))]
    fn worker(&self, num_threads: usize) -> PyResult<PyDynamic1dWorker> {
        if num_threads == 0 {
            return Err(PyValueError::new_err("num_threads must be positive"));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyDynamic1dWorker {
            plan: Arc::clone(&self.plan),
            pool,
        })
    }
    #[getter]
    fn shared_plan_id(&self) -> usize {
        Arc::as_ptr(&self.plan) as usize
    }
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.plan.shape()
    }
    #[getter]
    fn n_points(&self) -> usize {
        self.plan.n_points()
    }
}

#[pymethods]
impl PyDynamic1dWorker {
    /// Integrate a contiguous float64 frame with its complete uint8 mask.
    /// Inputs are borrowed without copying and must not be mutated concurrently.
    /// The GIL is released during numerical work. No mask/normalization state
    /// is cached across frames; every call has independent accumulators.
    #[pyo3(signature = (image, mask=None, *, errors=false, azimuthal_std=false))]
    fn integrate1d(
        &self,
        py: Python<'_>,
        image: PyReadonlyArray2<'_, f64>,
        mask: Option<PyReadonlyArray2<'_, u8>>,
        errors: bool,
        azimuthal_std: bool,
    ) -> PyResult<PyIntegrationResult> {
        use numpy::PyUntypedArrayMethods;
        let shape = self.plan.shape();
        if image.shape() != [shape.0, shape.1] {
            return Err(PyValueError::new_err(
                "image shape does not match prepared plan",
            ));
        }
        if !image.is_c_contiguous() {
            return Err(PyValueError::new_err("image must be C contiguous"));
        }
        let data = image
            .as_slice()
            .map_err(|_| PyValueError::new_err("image must be C contiguous"))?;
        let mask_data = if let Some(mask) = mask.as_ref() {
            if mask.shape() != image.shape() {
                return Err(PyValueError::new_err(
                    "mask shape does not match prepared plan",
                ));
            }
            if !mask.is_c_contiguous() {
                return Err(PyValueError::new_err("mask must be C contiguous"));
            }
            Some(
                mask.as_slice()
                    .map_err(|_| PyValueError::new_err("mask must be C contiguous"))?,
            )
        } else {
            None
        };
        let result = py.detach(|| {
            self.pool
                .install(|| self.plan.integrate(data, mask_data, errors, azimuthal_std))
        });
        Ok(PyIntegrationResult {
            radial: Some(result.radial),
            intensity: result.intensity,
            errors: result.errors,
            azimuthal_std: result.azimuthal_std,
            azimuthal: None,
            n_radial: None,
            n_azimuthal: None,
        })
    }
    #[getter]
    fn shared_plan_id(&self) -> usize {
        Arc::as_ptr(&self.plan) as usize
    }
    #[getter]
    fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }
}
