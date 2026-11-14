"""Type stubs for dioptrin - High-performance 1D/2D diffraction integration library."""

from typing import Iterable, Iterator, Literal, Optional, Tuple, Union

import numpy as np

__all__ = [
    "Integrator",
    "Dynamic1dPlan",
    "Dynamic1dWorker",
    "MultiIntegrator",
    "IntegrationResult",
    "BatchResult",
    "BatchIterator",
    "rayon_num_threads",
    "set_rayon_threads",
    "validate_license",
    "LicenseError",
    "LicenseNotFoundError",
    "LicenseExpiredError",
    "Calibrant",
    "CalibrationPoints",
    "CalibrationResult",
    "EllipseFit",
    "RansacEllipseResult",
    "RingIdResult",
    "AutoCalibrateResult",
    "refine",
    "find_peaks_massif",
    "ransac_ellipses",
    "identify_rings",
    "find_peaks_image",
    "auto_calibrate",
    "estimate_ring_center",
]


class LicenseError(Exception):
    """Legacy compatibility exception; the open-source package never raises it."""


class LicenseNotFoundError(LicenseError): ...


class LicenseExpiredError(LicenseError): ...


def validate_license() -> None:
    """Compatibility no-op; no activation or license file is required."""
    ...


def rayon_num_threads() -> int:
    """
    Get the current number of threads used by Rayon for parallel processing.

    Returns:
        Number of threads currently configured for parallel operations

    Example:
        >>> import dioptrin
        >>> threads = dioptrin.rayon_num_threads()
        >>> print(f"Using {threads} threads")
    """
    ...


def set_rayon_threads(threads: int) -> None:
    """
    Set the number of threads for Rayon parallel processing.

    This affects all parallel operations in the library, including geometry
    calculations and integration.

    Args:
        threads: Number of threads to use (must be > 0)

    Raises:
        ValueError: If thread count is invalid or cannot be set

    Example:
        >>> import dioptrin
        >>> dioptrin.set_rayon_threads(8)  # Use 8 threads
    """
    ...


class IntegrationResult:
    """
    Result of an integration operation.

    Contains the integrated intensity data along with optional radial/azimuthal
    coordinates and error estimates, depending on the integration type performed.

    Attributes:
        radial: Radial bin centers (1D and MultiIntegrator 2D; otherwise None)
        azimuthal: Azimuthal bin centers in radians (MultiIntegrator 2D and Integrator
            2D with azimuthal_range; otherwise None)
        intensity: Integrated intensity values
        errors: Error estimates (present if errors=True was specified)
        azimuthal_std: Weighted population spread (1D, when explicitly requested)
        n_radial: Number of radial bins (for reshaping 2D data)
        n_azimuthal: Number of azimuthal bins (for reshaping 2D data)
    """

    @property
    def radial(self) -> Optional[list[float]]:
        """Radial bin centers (1D and MultiIntegrator 2D; otherwise None)."""
        ...

    @property
    def azimuthal(self) -> Optional[list[float]]:
        """Azimuthal bin centers in radians (MultiIntegrator 2D, Integrator 2D with azimuthal_range)."""
        ...

    @property
    def intensity(self) -> list[float]:
        """Integrated intensity values."""
        ...

    @property
    def errors(self) -> Optional[list[float]]:
        """Error estimates (if errors=True was specified)."""
        ...

    @property
    def azimuthal_std(self) -> Optional[list[float]]:
        """Corrected pixel population standard deviation; NaN for empty bins."""
        ...

    @property
    def n_radial(self) -> Optional[int]:
        """Number of radial bins (for reshaping 2D data)."""
        ...

    @property
    def n_azimuthal(self) -> Optional[int]:
        """Number of azimuthal bins (for reshaping 2D data)."""
        ...


class BatchResult:
    """
    Result of a single file in a batch integration.

    Attributes:
        filename: Path to the processed file
        result: IntegrationResult if successful, None if failed
        error: Error message if failed, None if successful
    """

    @property
    def filename(self) -> str:
        """Path of processed file."""
        ...

    @property
    def result(self) -> Optional[IntegrationResult]:
        """IntegrationResult or None if failed."""
        ...

    @property
    def error(self) -> Optional[str]:
        """Error message or None if successful."""
        ...

    def is_ok(self) -> bool:
        """Check if the integration was successful."""
        ...

    def is_err(self) -> bool:
        """Check if the integration failed."""
        ...


class BatchIterator(Iterator[BatchResult]):
    """
    Iterator for streaming batch integration results.

    Yields BatchResult objects as each file completes processing.
    Use this for memory-efficient processing of large batches.

    Attributes:
        total: Total number of files being processed
        completed: Number of files processed so far
        progress: Fraction complete (0.0 to 1.0)

    Example:
        >>> it = integrator.batch1d_iter(files, n_points=1000)
        >>> for result in it:
        ...     print(f"Progress: {it.completed}/{it.total}")
        ...     if result.is_ok():
        ...         process(result.result)
    """

    @property
    def total(self) -> int:
        """Total number of files being processed."""
        ...

    @property
    def completed(self) -> int:
        """Number of files processed so far."""
        ...

    @property
    def progress(self) -> Optional[float]:
        """Fraction complete (0.0 to 1.0), or None when total is unknown (generator input)."""
        ...

    def __iter__(self) -> "BatchIterator":
        ...

    def __next__(self) -> BatchResult:
        ...


class Integrator:
    """
    Integrator for 1D/2D radial integration of diffraction patterns.

    The integrator is configured with detector geometry from a PONI file
    and supports various integration methods and corrections.
    """

    def __init__(
        self,
        poni_path: str,
        method: Optional[str] = None,
        polarization_factor: Optional[float] = None,
        n_ss: Optional[int] = None,
        unit: Optional[str] = None,
        radial_range: Optional[Tuple[float, float]] = None,
        *,
        fast_math: bool = True,
        mask_mode: Literal["fixed", "dynamic"] = "fixed",
        sparse_layout: Literal["csc", "csr"] = "csc",
    ) -> None:
        """
        Create a new Integrator instance.

        The integrator is configured with detector geometry from a PONI file.
        The dimensions will be determined automatically when the first image
        is integrated.

        Args:
            poni_path: Path to the PONI calibration file (v1 or v2 format)
            method: Integration method: "pixel_split" (default), "pyfai_full", "counting", "histogram", or "supersampled"
            polarization_factor: Optional polarization factor P in [-1, 1]: +1 fully
                polarised along the horizontal axis (chi = 0, detector dim2; the
                usual synchrotron case, typically 0.95-0.99), -1 vertical,
                0 unpolarised. None (default) applies no polarization correction.
            n_ss: Number of subpixels per pixel for supersampled method (default: 5)
            unit: Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"
            radial_range: Optional (min, max) tuple for explicit radial bin range.
                Required when using MultiIntegrator to ensure all detectors
                share the same bin grid. Units must match the selected unit.
                Not supported with method="histogram", which always uses the
                automatic radial range ("counting", "pixel_split", "pyfai_full"
                and "supersampled" honour it).

        Raises:
            ValueError: If an option is invalid: unknown method/unit, n_ss = 0,
                polarization_factor outside [-1, 1], radial_range not finite with
                min < max, radial_range combined with method="histogram", or a
                q unit without a positive PONI wavelength

        Example:
            >>> import dioptrin
            >>> integrator = dioptrin.Integrator(
            ...     "calibration.poni",
            ...     method="pixel_split",
            ...     unit="q_A^-1",
            ...     radial_range=(0.5, 25.0),
            ... )
        """
        ...

    @classmethod
    def from_poni_dict(
        cls,
        poni_dict: dict[str, float],
        method: Optional[str] = None,
        polarization_factor: Optional[float] = None,
        n_ss: Optional[int] = None,
        unit: Optional[str] = None,
        radial_range: Optional[Tuple[float, float]] = None,
        *,
        fast_math: bool = True,
        mask_mode: Literal["fixed", "dynamic"] = "fixed",
        sparse_layout: Literal["csc", "csr"] = "csc",
    ) -> "Integrator":
        """
        Create an Integrator from a PONI dictionary.

        The dictionary should contain PONI calibration parameters as keys:
        "pixel1", "pixel2", "distance", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength".

        Args:
            poni_dict: Dictionary containing PONI calibration parameters
            method: Integration method: "pixel_split" (default), "pyfai_full", "counting", "histogram", or "supersampled"
            polarization_factor: Optional polarization factor P in [-1, 1], as for
                Integrator(): +1 horizontal (chi = 0, detector dim2; typical
                synchrotron values 0.95-0.99), -1 vertical, 0 unpolarised
            n_ss: Number of subpixels per pixel for supersampled method (default: 5)
            unit: Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"
            radial_range: Optional (min, max) tuple for explicit radial bin range.
                Not supported with method="histogram", which always uses the
                automatic radial range.

        Returns:
            New Integrator instance

        Raises:
            ValueError: If an option is invalid, as for Integrator() (including
                radial_range with method="histogram"); a q unit needs a positive
                "wavelength" in the dictionary

        Example:
            >>> import dioptrin
            >>> poni_dict = {
            ...     "pixel1": 0.000172,
            ...     "pixel2": 0.000172,
            ...     "distance": 0.2,
            ...     "poni1": 0.1,
            ...     "poni2": 0.1,
            ...     "wavelength": 1e-10
            ... }
            >>> integrator = dioptrin.Integrator.from_poni_dict(poni_dict)
        """
        ...

    @property
    def fast_math(self) -> bool:
        """Reassociated CSR accumulation; geometry/LUT construction is unchanged."""
        ...

    def set_fast_math(self, enabled: bool) -> None:
        """Toggle fast accumulation; rejects changes during an async batch."""
        ...

    @property
    def mask_mode(self) -> Literal["fixed", "dynamic"]:
        """Fixed caches mask-compacted CSR; dynamic recomputes per-frame normalization."""
        ...

    def set_mask_mode(self, mode: Literal["fixed", "dynamic"]) -> None:
        """Switch mask policy without rebuilding geometry; rejects changes during a batch."""
        ...

    @property
    def sparse_layout(self) -> Literal["csc", "csr"]: ...
    def set_sparse_layout(self, layout: Literal["csc", "csr"]) -> None: ...

    def prepare_dynamic1d(self, shape: Tuple[int, int], n_points: int) -> Dynamic1dPlan:
        """Prepare one immutable geometry/weight/correction snapshot for all workers.

        The stored mask is excluded. Workers accept a complete uint8 mask per
        frame. Subsequent changes to this Integrator do not change the plan.
        """
        ...

    def clear_file_workers(self) -> None:
        """Release cached file workers and TIFF buffers; admitted streams finish normally."""
        ...

    def integrate1d(
        self,
        image: Union[str, np.ndarray],
        n_points: int,
        errors: bool = False,
        *,
        azimuthal_std: bool = False,
    ) -> IntegrationResult:
        """
        Perform 1D radial integration.

        Args:
            image: Path to the TIFF image file or 2D NumPy array
            n_points: Number of bins for the 1D integration output
            errors: Whether to compute Poisson error estimates (default: False)
            azimuthal_std: Also compute weighted population spread (default: False)

        Returns:
            IntegrationResult with radial values, intensities, and optional errors

        Raises:
            ValueError: If n_points is 0, the image is empty or not C-contiguous,
                or its shape does not match the mask

        Example:
            >>> integrator = dioptrin.Integrator("calibration.poni")
            >>> # From file
            >>> result = integrator.integrate1d("pattern.tif", n_points=1000)
            >>> # From array
            >>> import numpy as np
            >>> data = np.array(image_data, dtype=np.float64)
            >>> result = integrator.integrate1d(data, n_points=1000)
        """
        ...

    def integrate2d(
        self,
        image: Union[str, np.ndarray],
        n_radial: int,
        n_azimuthal: Optional[int] = None,
        errors: bool = False,
        azimuthal_range: Optional[Tuple[float, float]] = None,
        *,
        azimuthal_bracket: Optional[float] = None,
    ) -> IntegrationResult:
        """
        Perform 2D integration.

        Args:
            image: Path to the TIFF image file or 2D NumPy array
            n_radial: Number of radial bins
            n_azimuthal: Number of azimuthal bins. Give either this or
                azimuthal_bracket.
            azimuthal_bracket: Keyword only, instead of n_azimuthal: half-width in
                radians of contiguous rows given by their centres. The first row is
                centred on the start of azimuthal_range (default (0, 2*pi)), every
                row covers its centre +- the bracket, and the number of rows is
                span / (2 * bracket), which must be a whole number (else
                ValueError). E.g. azimuthal_bracket=np.radians(2.5) gives 72 rows
                centred on 0, 5, ..., 355 degrees. result.azimuthal holds the
                centres and result.n_azimuthal the row count.
                In this mode azimuthal_range is (first centre, end) rather than
                edges, and the end is exclusive: (0, pi/2) with a 5 degree
                bracket gives centres 0, 10, ..., 80 degrees, covering -5 to 85
                degrees. A first centre in [-pi, 2*pi] is kept as given; any
                other is corrected to the equivalent angle in [-pi, pi). With method="pyfai_full" the
                seam lies one bracket below the first centre, so rows differ
                slightly from a default pyFAI cake.
            errors: Whether to compute error estimates (default: False)
            azimuthal_range: Optional (min, max) azimuthal edges in radians, with min
                at any finite, resolvable angle and 0 < max - min <= 2*pi. Starts outside ±2*pi
                are corrected to [-pi, pi), including the returned axis. The n_azimuthal bins split
                it evenly (bin i covers [min + i*step, min + (i+1)*step)) and pixels
                outside are left out, so sectors work, also across the +-pi cut.
                result.azimuthal holds the bin centres. Radial bins are the same as
                without a range. Invalid ranges raise ValueError.
                With method="pyfai_full", individual bins depend on how the range
                is written (a range shifted by 2*pi, or a sector vs the same rows
                of a full circle, can differ, as in pyFAI itself); prefer
                "pixel_split" for sectors.

        Returns:
            IntegrationResult with flat intensity (and optional errors) sized
            n_radial * n_azimuthal

        Raises:
            ValueError: If the integrator uses the "histogram" method, which has
                no azimuthal binning; if n_radial or n_azimuthal is 0; or if the
                image is empty or its shape does not match the mask

        Example:
            >>> integrator = dioptrin.Integrator("calibration.poni")
            >>> # From file
            >>> result = integrator.integrate2d("pattern.tif", n_radial=100, n_azimuthal=360)
            >>> # From array
            >>> result = integrator.integrate2d(data, n_radial=100, n_azimuthal=360)
        """
        ...

    def set_method(self, method: str, n_ss: Optional[int] = None) -> None:
        """
        Set the integration method.

        Changes the integration method used for subsequent integrations.
        Available methods: "pixel_split", "pyfai_full", "counting", "histogram", or "supersampled".

        Args:
            method: Integration method name
            n_ss: Number of subpixels per pixel for supersampled method (required if method is "supersampled")

        Raises:
            ValueError: If the method name is invalid or n_ss is 0, or if the
                method is "histogram" while the Integrator has a radial_range
                (Histogram always uses the automatic radial range)

        Example:
            >>> integrator.set_method("supersampled", n_ss=7)
        """
        ...

    def set_polarization_factor(self, polarization_factor: Optional[float]) -> None:
        """
        Set the polarization correction factor.

        Applies the correction 0.5 * (1 + cos²2θ - P·cos2χ·sin²2θ). P must be
        in [-1, 1]: +1 fully polarised along the horizontal axis (χ = 0,
        detector dim2; the usual synchrotron case, typically 0.95-0.99),
        -1 vertical, 0 unpolarised. Set to None to disable polarization correction.

        Args:
            polarization_factor: Polarization factor P in [-1, 1], or None to disable

        Raises:
            ValueError: If the factor is not finite or outside [-1, 1]

        Example:
            >>> integrator.set_polarization_factor(0.99)  # Enable correction
            >>> integrator.set_polarization_factor(None)  # Disable correction
        """
        ...

    def set_unit(self, unit: str) -> None:
        """
        Set the radial unit for integration output.

        Changes the unit used for the radial (scattering angle) axis in integration results.

        Args:
            unit: Radial unit: "2th_rad", "2th_deg", "q_A^-1", or "q_nm^-1"

        Raises:
            ValueError: If the unit name is invalid, or a q unit is requested
                and the PONI wavelength is not positive

        Example:
            >>> integrator.set_unit("q_A^-1")  # Switch to q-space in inverse Angstroms
        """
        ...

    def set_mask(self, mask: Optional[Union[str, np.ndarray]]) -> None:
        """
        Set the mask for the integrator.

        The mask can be provided as a path to a TIFF file or as a NumPy array.
        In the mask, non-zero values indicate pixels to be masked (ignored).

        Args:
            mask: Path to a TIFF file (str), a NumPy array (uint8 or bool), or None to clear the mask.

        Raises:
            ValueError: If the mask file cannot be loaded or if the array has an incorrect shape or type.

        Example:
            >>> import numpy as np
            >>> # From file
            >>> integrator.set_mask("mask.tif")
            >>>
            >>> # From array
            >>> mask_array = np.zeros((1043, 981), dtype=np.uint8)
            >>> mask_array[500:600, :] = 1
            >>> integrator.set_mask(mask_array)
            >>>
            >>> # Clear mask
            >>> integrator.set_mask(None)
        """
        ...

    def clear_mask(self) -> None:
        """
        Clear the mask from the integrator.

        Example:
            >>> integrator.clear_mask()
        """
        ...

    def get_unit(self) -> str:
        """
        Get the current radial unit as a string.

        Returns the unit currently configured for the radial axis.

        Returns:
            String representation of the current radial unit: "2th_rad", "2th_deg", "q_A^-1", or "q_nm^-1"

        Example:
            >>> unit = integrator.get_unit()
            >>> print(f"Current unit: {unit}")
        """
        ...

    def batch1d(
        self,
        images: Union[list[str], list[np.ndarray]],
        n_points: int,
        errors: bool = False,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
        *,
        azimuthal_std: bool = False,
    ) -> list[BatchResult]:
        """
        Perform batch 1D integration.

        Returns a list of BatchResult objects with IntegrationResult containing
        optional error estimates.

        C-contiguous float64 NumPy images are borrowed without copying their
        pixels for this synchronous call. Keep inputs unchanged until it returns;
        they must not be mutated concurrently. Native workers finish before the
        input borrow guards are released. Streaming APIs retain owned frames.
        The GIL is released while the workers run, so other Python threads
        (including a generator feeding a batch*_iter stream) keep running.

        Args:
            images: List of TIFF file paths or 2D float64 NumPy arrays
            n_points: Number of bins for the 1D integration output
            errors: Whether to compute Poisson error estimates (default: False)
            azimuthal_std: Also compute weighted population spread (default: False)
            num_workers: Number of parallel workers (optional)
            num_threads: Threads per worker (optional)

        Returns:
            List of BatchResult objects, each containing filename and IntegrationResult.

        Raises:
            ValueError: If n_points is 0, or an array is empty, not C-contiguous,
                or does not match the mask shape (file errors go to BatchResult.error)
            TypeError: If images is not a list of paths or 2D float64 arrays

        Example:
            >>> results = integrator.batch1d(files, n_points=1000, errors=True)
            >>> for r in results:
            ...     if r.is_ok():
            ...         print(r.result.intensity)
            ...         if r.result.errors:
            ...             print(r.result.errors)
        """
        ...

    def batch1d_iter(
        self,
        images: Union[list[str], list[np.ndarray], Iterable[Union[str, np.ndarray]]],
        n_points: int,
        errors: bool = False,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
        *,
        azimuthal_std: bool = False,
    ) -> BatchIterator:
        """
        Perform streaming batch 1D integration.

        Returns an iterator that yields results as each frame completes.
        Accepts lists or generators/iterators for constant-memory processing.

        Args:
            images: List of TIFF file paths, list of NumPy arrays (2D float64),
                or a generator/iterator yielding file paths or NumPy arrays
            n_points: Number of bins for the 1D integration output
            errors: Whether to compute Poisson error estimates (default: False)
            azimuthal_std: Also compute weighted population spread (default: False)
            num_workers: Number of parallel workers (optional)
            num_threads: Threads per worker (optional)

        Returns:
            BatchIterator that yields BatchResult objects as they complete.
            The iterator has `total`, `completed`, and `progress` properties.
            When using a generator, `total` is 0 and `progress` returns None.

        Raises:
            ValueError: If n_points is 0, or an image is empty, not C-contiguous,
                or does not match the mask shape
            TypeError: If an item is neither a path nor a 2D float64 array.
                Lists are checked up front; for a generator the error (or any
                exception the generator raises) is raised by the iterator at
                that item's position, after all earlier results

        Example:
            >>> # With a list
            >>> it = integrator.batch1d_iter(files, n_points=1000)
            >>> for result in it:
            ...     print(f"Progress: {it.completed}/{it.total}")
            ...     if result.is_ok():
            ...         process(result.result)
            >>>
            >>> # With a generator (constant memory)
            >>> def frame_gen():
            ...     with h5py.File('data.h5') as f:
            ...         for i in range(1000):
            ...             yield f['/data'][i].astype(np.float64)
            >>> for result in integrator.batch1d_iter(frame_gen(), n_points=1000):
            ...     process(result.result)
        """
        ...

    def batch2d(
        self,
        tiff_paths: list[str],
        n_radial: int,
        n_azimuthal: Optional[int] = None,
        errors: bool = False,
        azimuthal_range: Optional[Tuple[float, float]] = None,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
        *,
        azimuthal_bracket: Optional[float] = None,
    ) -> list[BatchResult]:
        """
        Perform batch 2D integration.

        Returns a list of BatchResult objects with IntegrationResult containing
        optional error estimates.

        Args:
            tiff_paths: List of TIFF file paths to process
            n_radial: Number of radial bins
            n_azimuthal: Number of azimuthal bins. Give either this or
                azimuthal_bracket.
            azimuthal_bracket: Keyword only, instead of n_azimuthal: half-width in
                radians of contiguous rows given by their centres. The first row is
                centred on the start of azimuthal_range (default (0, 2*pi)), every
                row covers its centre +- the bracket, and the number of rows is
                span / (2 * bracket), which must be a whole number (else
                ValueError). E.g. azimuthal_bracket=np.radians(2.5) gives 72 rows
                centred on 0, 5, ..., 355 degrees. result.azimuthal holds the
                centres and result.n_azimuthal the row count.
                In this mode azimuthal_range is (first centre, end) rather than
                edges, and the end is exclusive: (0, pi/2) with a 5 degree
                bracket gives centres 0, 10, ..., 80 degrees, covering -5 to 85
                degrees. A first centre in [-pi, 2*pi] is kept as given; any
                other is corrected to the equivalent angle in [-pi, pi). With method="pyfai_full" the
                seam lies one bracket below the first centre, so rows differ
                slightly from a default pyFAI cake.
            errors: Whether to compute error estimates (default: False)
            azimuthal_range: Optional (min, max) azimuthal edges in radians, with min
                at any finite, resolvable angle and 0 < max - min <= 2*pi. Starts outside ±2*pi
                are corrected to [-pi, pi), including the returned axis. The n_azimuthal bins split
                it evenly (bin i covers [min + i*step, min + (i+1)*step)) and pixels
                outside are left out, so sectors work, also across the +-pi cut.
                result.azimuthal holds the bin centres. Radial bins are the same as
                without a range. Invalid ranges raise ValueError.
                With method="pyfai_full", individual bins depend on how the range
                is written (a range shifted by 2*pi, or a sector vs the same rows
                of a full circle, can differ, as in pyFAI itself); prefer
                "pixel_split" for sectors.
            num_workers: Number of parallel workers (optional)
            num_threads: Threads per worker (optional)

        Returns:
            List of BatchResult objects, each containing filename and IntegrationResult.

        Raises:
            ValueError: If the integrator uses the "histogram" method, which has
                no azimuthal binning, or if n_radial or n_azimuthal is 0

        Example:
            >>> # Full 2D integration
            >>> results = integrator.batch2d(files, n_radial=100, n_azimuthal=360)
            >>>
            >>> # With azimuthal range and errors
            >>> results = integrator.batch2d(
            ...     files, n_radial=100, n_azimuthal=90,
            ...     azimuthal_range=(0, 1.57),  # 0 to pi/2
            ...     errors=True
            ... )
        """
        ...

    def batch2d_iter(
        self,
        images: Union[list[str], list[np.ndarray], Iterable[Union[str, np.ndarray]]],
        n_radial: int,
        n_azimuthal: Optional[int] = None,
        errors: bool = False,
        azimuthal_range: Optional[Tuple[float, float]] = None,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
        *,
        azimuthal_bracket: Optional[float] = None,
    ) -> BatchIterator:
        """
        Perform streaming batch 2D integration.

        Returns an iterator that yields results as each frame completes.
        Accepts lists or generators/iterators for constant-memory processing.

        Args:
            images: List of TIFF file paths, list of NumPy arrays (2D float64),
                or a generator/iterator yielding file paths or NumPy arrays
            n_radial: Number of radial bins
            n_azimuthal: Number of azimuthal bins. Give either this or
                azimuthal_bracket.
            azimuthal_bracket: Keyword only, instead of n_azimuthal: half-width in
                radians of contiguous rows given by their centres. The first row is
                centred on the start of azimuthal_range (default (0, 2*pi)), every
                row covers its centre +- the bracket, and the number of rows is
                span / (2 * bracket), which must be a whole number (else
                ValueError). E.g. azimuthal_bracket=np.radians(2.5) gives 72 rows
                centred on 0, 5, ..., 355 degrees. result.azimuthal holds the
                centres and result.n_azimuthal the row count.
                In this mode azimuthal_range is (first centre, end) rather than
                edges, and the end is exclusive: (0, pi/2) with a 5 degree
                bracket gives centres 0, 10, ..., 80 degrees, covering -5 to 85
                degrees. A first centre in [-pi, 2*pi] is kept as given; any
                other is corrected to the equivalent angle in [-pi, pi). With method="pyfai_full" the
                seam lies one bracket below the first centre, so rows differ
                slightly from a default pyFAI cake.
            errors: Whether to compute error estimates (default: False)
            azimuthal_range: Optional (min, max) azimuthal edges in radians, with min
                at any finite, resolvable angle and 0 < max - min <= 2*pi. Starts outside ±2*pi
                are corrected to [-pi, pi), including the returned axis. The n_azimuthal bins split
                it evenly (bin i covers [min + i*step, min + (i+1)*step)) and pixels
                outside are left out, so sectors work, also across the +-pi cut.
                result.azimuthal holds the bin centres. Radial bins are the same as
                without a range. Invalid ranges raise ValueError.
                With method="pyfai_full", individual bins depend on how the range
                is written (a range shifted by 2*pi, or a sector vs the same rows
                of a full circle, can differ, as in pyFAI itself); prefer
                "pixel_split" for sectors.
            num_workers: Number of parallel workers (optional)
            num_threads: Threads per worker (optional)

        Returns:
            BatchIterator that yields BatchResult objects as they complete.
            The iterator has `total`, `completed`, and `progress` properties.
            When using a generator, `total` is 0 and `progress` returns None.

        Raises:
            ValueError: If the integrator uses the "histogram" method, which has
                no azimuthal binning; if n_radial or n_azimuthal is 0; or if an
                image is empty, not C-contiguous, or does not match the mask shape
            TypeError: If an item is neither a path nor a 2D float64 array.
                Lists are checked up front; for a generator the error (or any
                exception the generator raises) is raised by the iterator at
                that item's position, after all earlier results

        Example:
            >>> # With a list
            >>> for result in integrator.batch2d_iter(files, n_radial=100, n_azimuthal=360):
            ...     if result.is_ok():
            ...         cake = result.result.intensity
            >>>
            >>> # With a generator
            >>> for result in integrator.batch2d_iter(frame_gen(), n_radial=100, n_azimuthal=360):
            ...     process(result.result)
        """
        ...


class MultiIntegrator:
    """
    Multi-detector integrator that merges results from multiple detectors.

    Merges integration results from multiple detectors into a single diffraction
    pattern by summing raw signal, normalization, and variance across detectors
    before dividing. The 2D absolute normalization mode supplies pyFAI
    MultiGeometry's solid-angle scale; splitting algorithms can still differ.

    All integrators must share the same radial_unit and radial_range, and must
    not use the "histogram" method.

    Example:
        >>> import dioptrin
        >>> int1 = dioptrin.Integrator("det1.poni", unit="q_A^-1", radial_range=(0.5, 25.0))
        >>> int2 = dioptrin.Integrator("det2.poni", unit="q_A^-1", radial_range=(0.5, 25.0))
        >>> multi = dioptrin.MultiIntegrator(integrators=[int1, int2])
        >>> result = multi.integrate1d(images=[img1, img2], n_points=2000)
    """

    def __init__(self, integrators: list[Integrator], *, pixel_split_mode: str = "fractional") -> None:
        """
        Create a new MultiIntegrator from a list of Integrator instances.

        All integrators must:
        - Have the same radial_unit
        - Have an explicit radial_range set
        - Have the same radial_range values
        - Not use the "histogram" method

        Args:
            integrators: List of Integrator instances, one per detector
            pixel_split_mode: "fractional" (default) or "pyfai_histogram" for
                approximate Qt/pyFAI full-histogram 2D splitting. This changes
                boundary clipping and weights; it has no effect on 1D.
                Singular beam pixels use 128x128 strata with four symmetric
                samples per stratum in both modes. Negative weights are avoided.

        Raises:
            ValueError: If validation fails (empty list, mismatched units/ranges,
                missing radial_range, or histogram method used)

        Example:
            >>> int1 = dioptrin.Integrator("det1.poni", unit="q_A^-1", radial_range=(0.5, 25.0))
            >>> int2 = dioptrin.Integrator("det2.poni", unit="q_A^-1", radial_range=(0.5, 25.0))
            >>> multi = dioptrin.MultiIntegrator(integrators=[int1, int2])
        """
        ...

    def integrate1d(
        self,
        images: list[Union[str, np.ndarray]],
        n_points: int,
        errors: bool = False,
    ) -> IntegrationResult:
        """
        Integrate one image per detector and merge into a single 1D pattern.

        Args:
            images: List of images (file paths or 2D float64 NumPy arrays),
                one per detector, in the same order as the integrators
            n_points: Number of bins for the 1D integration output
            errors: Whether to compute error estimates (default: False)

        Returns:
            IntegrationResult with merged radial values, intensities, and optional errors

        Raises:
            ValueError: If number of images doesn't match number of detectors,
                n_points is 0, a TIFF cannot be loaded, or an image is empty or
                does not match its detector's mask shape

        Example:
            >>> result = multi.integrate1d(images=[img1, img2], n_points=2000)
            >>> result = multi.integrate1d(images=[img1, img2], n_points=2000, errors=True)
            >>> print(result.radial, result.intensity)
        """
        ...

    def integrate2d(
        self,
        images: list[Union[str, np.ndarray]],
        n_radial: int,
        n_azimuthal: Optional[int] = None,
        errors: bool = False,
        azimuthal_range: Optional[tuple[float, float]] = None,
        *,
        normalization: str = "relative",
        normalization_factors: Optional[list[float]] = None,
        azimuthal_bracket: Optional[float] = None,
    ) -> IntegrationResult:
        """Combine raw fractional pixel contributions onto shared 2D edges.

        Give either n_azimuthal or azimuthal_bracket. azimuthal_bracket is the
        half-width in radians of contiguous rows given by their centres: the
        first row is centred on the start of azimuthal_range, which then
        defaults to (0, 2*pi), and the span must be a whole number of rows
        (else ValueError). E.g. np.radians(2.5) gives 72 rows centred on
        0, 5, ..., 355 degrees. The end of azimuthal_range is exclusive in this
        mode: it is where the next centre would be. A first centre in
        [-pi, 2*pi] is kept as given; any other is corrected to the equivalent
        angle in [-pi, pi).

        Radial edges use the common radial_range and unit. Without a bracket,
        azimuthal edges are radians (default -pi, pi), span at most 2*pi, and
        clip sectors without folding outliers. Supports counting, pixel_split and supersampled.
        Inputs are C-contiguous float64 arrays or TIFF paths; mixed lists work.
        normalization='absolute' additionally puts pixel1*pixel2/distance**2 in
        each detector's denominator for pyFAI MultiGeometry absolute scaling.
        normalization_factors contains one finite positive monitor per detector.
        errors=True propagates independent Poisson max(image, 0) variance.
        Coordinates contain bin centers; flat intensity/errors reshape to
        (n_azimuthal, n_radial). Empty bins have zero intensity/error.
        Invalid grids, shapes, masks, geometry or factors raise ValueError.
        """
        ...

    def batch1d(
        self,
        images: list[list[Union[str, np.ndarray]]],
        n_points: int,
        errors: bool = False,
    ) -> list[IntegrationResult]:
        """
        Batch integrate multiple sets of detector images.

        Each element in the outer list is one "frame set" containing one image
        per detector. The inner lists must have the same length as the number
        of integrators.

        Args:
            images: List of image-lists, e.g.
                ``[[det1_frame1, det2_frame1], [det1_frame2, det2_frame2], ...]``
            n_points: Number of bins for the 1D integration output
            errors: Whether to compute error estimates (default: False)

        Returns:
            List of IntegrationResult, one per frame set

        Raises:
            ValueError: If any image-list has wrong length, n_points is 0, a TIFF
                cannot be loaded, or an image is empty or does not match its
                detector's mask shape

        Example:
            >>> image_sets = [[img1_a, img2_a], [img1_b, img2_b]]
            >>> results = multi.batch1d(images=image_sets, n_points=2000)
            >>> for result in results:
            ...     print(result.intensity)
        """
        ...


class Dynamic1dPlan:
    """Thread-safe immutable prepared tables shared by lightweight frame workers."""
    @property
    def shape(self) -> Tuple[int, int]: ...
    @property
    def n_points(self) -> int: ...
    @property
    def shared_plan_id(self) -> int: ...
    def worker(self, num_threads: int = 1) -> Dynamic1dWorker: ...


class Dynamic1dWorker:
    """Own compute pool and per-call accumulation, sharing a Dynamic1dPlan."""
    @property
    def shared_plan_id(self) -> int: ...
    @property
    def num_threads(self) -> int: ...
    def integrate1d(self, image: np.ndarray, mask: Optional[np.ndarray] = None,
                    *, errors: bool = False, azimuthal_std: bool = False
                    ) -> IntegrationResult:
        """Borrow C-contiguous float64 image and optional uint8 mask; release GIL.

        Inputs must remain unchanged until the call returns. All masks and
        accumulators are independent between calls, including normalization.
        """
        ...


# ---------------------------------------------------------------------------
# Calibration API
# ---------------------------------------------------------------------------


class Calibrant:
    """
    Reference powder described by its list of d-spacings.

    Cannot be constructed directly; use ``Calibrant.from_file`` or
    ``Calibrant.from_d_spacings``.
    """

    @staticmethod
    def from_file(path: str) -> Calibrant:
        """
        Load a calibrant from a pyFAI-style ``.D`` file.

        Raises:
            IOError: If the file cannot be read or parsed
        """
        ...

    @staticmethod
    def from_d_spacings(name: str, d_spacings: list[float]) -> Calibrant:
        """Build from an explicit list of d-spacings (Å)."""
        ...

    @property
    def name(self) -> str: ...

    @property
    def d_spacings(self) -> list[float]: ...

    def n_rings(self) -> int: ...

    def two_theta_rad(self, wavelength_m: float) -> list[Optional[float]]:
        """
        2θ (radians) for each ring at the given wavelength (meters).

        Rings outside the Bragg-physical range yield None.
        """
        ...

    def __repr__(self) -> str: ...


class CalibrationPoints:
    """Collection of control points (pixel coordinates plus optional ring index)."""

    def __init__(self) -> None:
        """Create an empty set of control points."""
        ...

    def add(self, d1: float, d2: float, ring: Optional[int] = None) -> None:
        """
        Append one control point.

        Args:
            d1: Pixel coordinate along rows; sub-pixel allowed
            d2: Pixel coordinate along columns; sub-pixel allowed
            ring: Calibrant ring index (None for auto-assignment)
        """
        ...

    def extend(
        self,
        d1: list[float],
        d2: list[float],
        ring: list[Optional[int]],
    ) -> None:
        """
        Append from parallel arrays.

        Raises:
            ValueError: If d1, d2 and ring do not have the same length
        """
        ...

    def __len__(self) -> int: ...

    def clear(self) -> None: ...


class CalibrationResult:
    """Result of ``refine``. Not constructible from Python."""

    @property
    def n_iter(self) -> int: ...

    @property
    def chi2(self) -> float: ...

    @property
    def rms_rad(self) -> float: ...

    @property
    def assignments(self) -> list[Optional[int]]: ...

    @property
    def inliers(self) -> list[bool]: ...

    @property
    def poni(self) -> dict[str, float]:
        """
        Refined PONI as a dict with keys pixel1, pixel2, distance, poni1,
        poni2, rot1, rot2, rot3 and wavelength.
        """
        ...

    def __repr__(self) -> str: ...


def refine(
    start: Union[Integrator, dict[str, float]],
    points: CalibrationPoints,
    calibrant: Calibrant,
    fix: Optional[list[str]] = None,
    loss: Optional[Union[str, dict[str, Union[str, float]]]] = None,
    auto_assign: bool = False,
    assign_tol: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> CalibrationResult:
    """
    Refine detector geometry from picked control points and a known calibrant.

    Args:
        start: Starting guess: either an Integrator (its PONI is used) or a PONI dict
        points: A CalibrationPoints
        calibrant: A Calibrant
        fix: List of parameter names to lock (default: ["wavelength"]). Valid names:
            "distance" (or "dist"), "poni1", "poni2", "rot1", "rot2", "rot3",
            "wavelength" (or "wl"). Must be a list.
        loss: "l2" (default), "huber" (c = 1e-3 rad), or a dict like
            {"type": "huber", "c": <rad>}
        auto_assign: Assign unassigned points to the nearest predicted ring each iteration
        assign_tol: Auto-assignment tolerance in rad of 2θ (default 0.05)
        max_iter: Levenberg-Marquardt iterations (default 100)
        tol: Relative cost tolerance for convergence (default 1e-9)

    Returns:
        A CalibrationResult

    Raises:
        ValueError: If start, fix or loss is invalid, or the refinement fails
    """
    ...


def find_peaks_massif(
    image: np.ndarray,
    seed: Tuple[int, int],
    valley_size: Optional[float] = None,
    max_peaks: int = 200,
    dedup_dist: float = 1.0,
) -> list[Tuple[float, float]]:
    """
    Massif-based peak picking from a seed.

    Given a 2D image and an integer (d1, d2) seed, return a list of sub-pixel
    (d1, d2) peaks that belong to the same connected "above-local-background"
    region as the seed. This is the same algorithm Dioptas's
    find_peaks_automatic uses via pyFAI's Massif.

    Args:
        image: 2D C-contiguous float64 NumPy array
        seed: (d1, d2) pixel indices near a peak
        valley_size: Gaussian sigma for the background filter, in pixels. None (default)
            uses pyFAI's adaptive value max(5, max(dim1, dim2) / 50).
        max_peaks: Cap on returned sub-peaks (default 200)
        dedup_dist: Merge peaks closer than this in pixel space (default 1.0)

    Raises:
        ValueError: If the image is not C-contiguous
    """
    ...


class EllipseFit:
    """Ellipse fitted by ``ransac_ellipses``. Not constructible from Python."""

    @property
    def center(self) -> Tuple[float, float]: ...

    @property
    def semi_major(self) -> float: ...

    @property
    def semi_minor(self) -> float: ...

    @property
    def theta(self) -> float: ...

    @property
    def conic(self) -> list[float]:
        """
        Conic coefficients [A, B, C, D, E, F] with the convention
        A x² + B xy + C y² + D x + E y + F = 0.
        """
        ...

    def distance(self, point: Tuple[float, float]) -> float:
        """Sampson distance from the point to the ellipse."""
        ...

    def __repr__(self) -> str: ...


class RansacEllipseResult:
    """Result of ``ransac_ellipses``. Not constructible from Python."""

    @property
    def ellipses(self) -> list[EllipseFit]:
        """List of EllipseFit, sorted by semi-major axis (smallest first)."""
        ...

    @property
    def assignments(self) -> list[Optional[int]]:
        """
        Per-input-point assignment: k = inlier of ellipses[k], None = outlier
        rejected from every ellipse.
        """
        ...

    def inlier_mask(self) -> list[bool]:
        """True for inliers (any ellipse), False for outliers."""
        ...

    def n_ellipses(self) -> int: ...


def ransac_ellipses(
    points: list[Tuple[float, float]],
    inlier_threshold: float = 1.5,
    min_inliers: int = 10,
    max_iterations: int = 2000,
    max_ellipses: int = 12,
    max_axis_ratio: Optional[float] = None,
    dedupe_center_dist: float = 5.0,
    dedupe_size_rel: float = 0.02,
    rng_seed: int = 0xC0FFEEC0DE123456,
) -> RansacEllipseResult:
    """
    RANSAC ellipse clustering of a list of 2-D peak coordinates.

    Returns a RansacEllipseResult listing the detected ellipses (sorted by
    semi-major axis; for a powder diffraction pattern this matches ring index
    k = 0, 1, 2 ...) and a per-point assignment vector.

    Args:
        points: List of (d1, d2) peak coordinates
        inlier_threshold: Sampson distance (pixels) for inlier acceptance
        min_inliers: Minimum points required to accept an ellipse
        max_iterations: Random samples per ellipse search
        max_ellipses: Cap on detected ellipses
        max_axis_ratio: Optional axis-ratio limit (default None)
        dedupe_center_dist: Ellipse de-duplication center distance (default 5.0)
        dedupe_size_rel: Ellipse de-duplication relative size (default 0.02)
        rng_seed: Seed for reproducibility (unsigned 64-bit)
    """
    ...


class RingIdResult:
    """Result of ``identify_rings``. Not constructible from Python."""

    @property
    def ring_index(self) -> list[Optional[int]]:
        """
        Per-ellipse calibrant ring assignment. None = ellipse couldn't be
        matched to any calibrant ring within tolerance.
        """
        ...

    @property
    def anchor_ring(self) -> Optional[int]:
        """Calibrant ring of the smallest ellipse, or None if that ellipse is unmatched."""
        ...

    @property
    def n_matched(self) -> int:
        """Number of ellipses confidently matched to calibrant rings."""
        ...

    @property
    def confident(self) -> bool:
        """True iff n_matched >= min_matches."""
        ...

    def __repr__(self) -> str: ...


def identify_rings(
    ellipses: list[EllipseFit],
    calibrant: Calibrant,
    wavelength: float,
    ratio_tolerance: float = 0.02,
    max_anchor_ring: int = 8,
    min_matches: int = 3,
) -> RingIdResult:
    """
    Identify which calibrant ring each detected ellipse corresponds to,
    using the size-ratio invariant r_k / r_0 = tan(2θ_k) / tan(2θ_0).

    Iterates over every (anchor_ellipse, anchor_ring) hypothesis and picks the
    assignment that maximises matched ellipses subject to a distance-consistency
    check. Handles missing calibrant rings, spurious detected ellipses, and the
    case where the smallest detected ellipse is not the calibrant's ring 0.

    Args:
        ellipses: List of EllipseFit from ransac_ellipses. Must be sorted by
            semi-major axis (ascending).
        calibrant: A Calibrant
        wavelength: Beam wavelength in metres
        ratio_tolerance: Relative tolerance on size-ratio matching (default 0.02 = 2 %)
        max_anchor_ring: Highest calibrant ring index to try as the anchor (default 8)
        min_matches: Minimum matched ellipses for confident=True (default 3)
    """
    ...


def find_peaks_image(
    image: np.ndarray,
    valley_size: Optional[float] = None,
    threshold_factor: float = 3.0,
    max_peaks: int = 5000,
) -> list[Tuple[float, float]]:
    """
    Image-wide peak extraction without a seed.

    Returns sub-pixel (d1, d2) peak centers for every above-background blob.

    Args:
        image: 2D C-contiguous float64 NumPy array
        valley_size: Gaussian sigma for the background filter (pyFAI-adaptive default if None)
        threshold_factor: Pixels are flagged as foreground when the value exceeds
            mean + threshold_factor * std of the residual (default 3.0)
        max_peaks: Cap on returned peaks (default 5000)

    Raises:
        ValueError: If the image is not C-contiguous
    """
    ...


class AutoCalibrateResult:
    """Result of ``auto_calibrate``. Not constructible from Python."""

    @property
    def n_iter(self) -> int: ...

    @property
    def chi2(self) -> float: ...

    @property
    def rms_rad(self) -> float: ...

    @property
    def n_detected_ellipses(self) -> int: ...

    @property
    def n_identified_ellipses(self) -> int: ...

    @property
    def bootstrap_distance(self) -> float: ...

    @property
    def bootstrap_ring_center_d1(self) -> float:
        """
        Mean ring-center (d1, meters) used to initialise the bootstrap PONI.

        Caveat: equals PONI only for near-perpendicular detectors. For tilted
        detectors PONI and ring-center are distinct (see estimate_ring_center).
        """
        ...

    @property
    def bootstrap_ring_center_d2(self) -> float: ...

    @property
    def poni(self) -> dict[str, float]:
        """
        Refined PONI as a dict with keys pixel1, pixel2, distance, poni1,
        poni2, rot1, rot2, rot3 and wavelength.
        """
        ...

    def __repr__(self) -> str: ...


def auto_calibrate(
    peaks: list[Tuple[float, float]],
    calibrant: Calibrant,
    wavelength: float,
    pixel1: float,
    pixel2: float,
    method: str = "radial",
    ransac_inlier_threshold: float = 1.5,
    ransac_min_inliers: int = 10,
    ransac_max_iterations: int = 2000,
    ransac_max_ellipses: int = 12,
    ransac_max_axis_ratio: float = 1.2,
    ransac_dedupe_center_dist: float = 5.0,
    ransac_dedupe_size_rel: float = 0.02,
    ratio_tolerance: float = 0.03,
    min_matches: int = 3,
    center_consistency_px: float = 30.0,
    grid_distance_min: float = 0.05,
    grid_distance_max: float = 0.6,
    grid_distance_steps: int = 18,
    grid_rot_max_deg: float = 45.0,
    grid_rot_steps: int = 15,
    grid_poni_search_radius_px: float = 400.0,
    grid_poni_steps: int = 13,
    grid_tolerance_rad: float = 8.0e-3,
    grid_max_peaks: int = 400,
    grid_max_ring_index: int = 12,
    grid_top_k: int = 50,
    fix: Optional[list[str]] = None,
    huber_c: float = 2.0e-3,
    max_iter: int = 200,
    tol: float = 1e-9,
    rng_seed: int = 0xC0FFEEC0DE123456,
) -> AutoCalibrateResult:
    """
    End-to-end automatic calibration: peaks -> ellipses -> ring identity ->
    bootstrap PONI -> L-M refine. No manual seeds, no geometry hint.

    Args:
        peaks: List of (d1, d2) peak coordinates (e.g. from find_peaks_image)
        calibrant: Reference powder
        wavelength: Beam wavelength in metres
        pixel1: Detector pixel pitch along d1 in metres
        pixel2: Detector pixel pitch along d2 in metres
        method: "radial" (default), "ransac", "grid", or experimental "auto"
            (case-insensitive)
        fix: List of parameter names to lock during the final refine
            (default: ["wavelength"]); same names as for refine
        huber_c: Huber loss scale (rad) used for the refine (default 2e-3)
        max_iter: Refine iterations (default 200)
        tol: Relative cost tolerance for convergence (default 1e-9)
        rng_seed: Seed for reproducibility (unsigned 64-bit)

    The ransac_*, ratio_tolerance, min_matches, center_consistency_px and
    grid_* keyword arguments tune the RANSAC, ring-identification and
    grid-search sub-steps; defaults are sensible for typical lab/synchrotron
    data (near-perpendicular detector, Huber loss for the bootstrap).

    Raises:
        ValueError: If method or fix is invalid, or the calibration fails
    """
    ...


def estimate_ring_center(
    peaks: list[Tuple[float, float]],
    max_trials: int = 500,
    radial_bin_width: float = 4.0,
    top_k_bins: int = 10,
    max_radius_px: float = 5000.0,
    rng_seed: int = 0xBEAC0CEFF00DF00D,
) -> Optional[Tuple[float, float]]:
    """
    Estimate the ring center from a list of peak coordinates via 3-point
    circle RANSAC + radial-band consistency scoring.

    IMPORTANT: "ring center" is not the same as PONI (point of normal
    incidence) or beam center on a tilted detector. The three coincide only
    for a perpendicular detector. For tilted detectors each ring's projected
    ellipse has its own geometric center (depending on the ring's 2θ); this
    function returns a consensus from circle fits that is a useful starting
    guess only when rings are approximately circular (rot1, rot2 at most a
    few degrees).

    Robust to partial arcs and off-detector beams as long as the
    near-circular assumption holds.

    Args:
        peaks: List of (d1, d2) peak coordinates
        max_trials: Number of 3-point RANSAC samples (default 500)
        radial_bin_width: Width (px) of radial histogram bins for scoring
        top_k_bins: Score = sum of the top-K bin counts
        max_radius_px: Radii beyond this are ignored
        rng_seed: Seed for reproducibility (unsigned 64-bit)

    Returns:
        (d1, d2), or None if too few peaks / no consensus
    """
    ...
