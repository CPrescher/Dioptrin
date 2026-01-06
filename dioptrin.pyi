"""Type stubs for dioptrin - High-performance 1D/2D diffraction integration library."""

from typing import Iterator, Optional, Tuple, Union

import numpy as np

__all__ = [
    "Integrator",
    "IntegrationResult",
    "BatchResult",
    "BatchIterator",
    "rayon_num_threads",
    "set_rayon_threads",
]


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
        radial: Radial bin centers (present for 1D integration, None for 2D)
        azimuthal: Azimuthal bin centers (present for 2D integration, None for 1D)
        intensity: Integrated intensity values
        errors: Error estimates (present if errors=True was specified)
        n_radial: Number of radial bins (for reshaping 2D data)
        n_azimuthal: Number of azimuthal bins (for reshaping 2D data)
    """

    @property
    def radial(self) -> Optional[list[float]]:
        """Radial bin centers (1D only)."""
        ...

    @property
    def azimuthal(self) -> Optional[list[float]]:
        """Azimuthal bin centers (2D only)."""
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
    def progress(self) -> float:
        """Fraction complete (0.0 to 1.0)."""
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
    ) -> None:
        """
        Create a new Integrator instance.

        The integrator is configured with detector geometry from a PONI file.
        The dimensions will be determined automatically when the first image
        is integrated.

        Args:
            poni_path: Path to the PONI calibration file (v1 or v2 format)
            method: Integration method: "pixel_split" (default), "counting", "histogram", or "supersampled"
            polarization_factor: Optional polarization correction factor (0.0 to 1.0)
            n_ss: Number of subpixels per pixel for supersampled method (default: 5)
            unit: Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"

        Example:
            >>> import dioptrin
            >>> integrator = dioptrin.Integrator(
            ...     "calibration.poni",
            ...     method="pixel_split",
            ...     unit="2th_deg"
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
    ) -> "Integrator":
        """
        Create an Integrator from a PONI dictionary.

        The dictionary should contain PONI calibration parameters as keys:
        "pixel1", "pixel2", "distance", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength".

        Args:
            poni_dict: Dictionary containing PONI calibration parameters
            method: Integration method: "pixel_split" (default), "counting", "histogram", or "supersampled"
            polarization_factor: Optional polarization correction factor (0.0 to 1.0)
            n_ss: Number of subpixels per pixel for supersampled method (default: 5)
            unit: Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"

        Returns:
            New Integrator instance

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

    def integrate1d(
        self,
        image: Union[str, np.ndarray],
        n_points: int,
        errors: bool = False
    ) -> IntegrationResult:
        """
        Perform 1D radial integration.

        Args:
            image: Path to the TIFF image file or 2D NumPy array
            n_points: Number of bins for the 1D integration output
            errors: Whether to compute error estimates (default: False)

        Returns:
            IntegrationResult with radial values, intensities, and optional errors

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
        n_azimuthal: int,
        errors: bool = False,
        azimuthal_range: Optional[Tuple[float, float]] = None,
    ) -> IntegrationResult:
        """
        Perform 2D integration.

        Args:
            image: Path to the TIFF image file or 2D NumPy array
            n_radial: Number of radial bins
            n_azimuthal: Number of azimuthal bins
            errors: Whether to compute error estimates (default: False)
            azimuthal_range: Optional (min, max) tuple for azimuthal range in radians

        Returns:
            IntegrationResult with flat intensity (and optional errors) sized
            n_radial * n_azimuthal

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
        Available methods: "pixel_split", "counting", "histogram", or "supersampled".

        Args:
            method: Integration method name
            n_ss: Number of subpixels per pixel for supersampled method (required if method is "supersampled")

        Raises:
            ValueError: If the method name is invalid

        Example:
            >>> integrator.set_method("supersampled", n_ss=7)
        """
        ...

    def set_polarization_factor(self, polarization_factor: Optional[float]) -> None:
        """
        Set the polarization correction factor.

        Applies polarization correction to the integration. The factor should
        be between 0.0 and 1.0. Set to None to disable polarization correction.

        Args:
            polarization_factor: Polarization factor (0.0 to 1.0) or None to disable

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
            ValueError: If the unit name is invalid

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
        tiff_paths: list[str],
        n_points: int,
        errors: bool = False,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> list[BatchResult]:
        """
        Perform batch 1D integration.

        Returns a list of BatchResult objects with IntegrationResult containing
        optional error estimates.

        Args:
            tiff_paths: List of TIFF file paths to process
            n_points: Number of bins for the 1D integration output
            errors: Whether to compute error estimates (default: False)
            num_workers: Number of parallel workers (optional)
            num_threads: Threads per worker (optional)

        Returns:
            List of BatchResult objects, each containing filename and IntegrationResult.

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
        tiff_paths: list[str],
        n_points: int,
        errors: bool = False,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> BatchIterator:
        """
        Perform streaming batch 1D integration.

        Returns an iterator that yields results as each file completes.
        More memory-efficient for large batches.

        Args:
            tiff_paths: List of TIFF file paths to process
            n_points: Number of bins for the 1D integration output
            errors: Whether to compute error estimates (default: False)
            num_workers: Number of parallel workers (optional)
            num_threads: Threads per worker (optional)

        Returns:
            BatchIterator that yields BatchResult objects as they complete.
            The iterator has `total`, `completed`, and `progress` properties.

        Example:
            >>> it = integrator.batch1d_iter(files, n_points=1000)
            >>> for result in it:
            ...     print(f"Progress: {it.completed}/{it.total}")
            ...     if result.is_ok():
            ...         process(result.result)
        """
        ...

    def batch2d(
        self,
        tiff_paths: list[str],
        n_radial: int,
        n_azimuthal: int,
        errors: bool = False,
        azimuthal_range: Optional[Tuple[float, float]] = None,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> list[BatchResult]:
        """
        Perform batch 2D integration.

        Returns a list of BatchResult objects with IntegrationResult containing
        optional error estimates.

        Args:
            tiff_paths: List of TIFF file paths to process
            n_radial: Number of radial bins
            n_azimuthal: Number of azimuthal bins
            errors: Whether to compute error estimates (default: False)
            azimuthal_range: Optional (min, max) tuple for azimuthal range in radians
            num_workers: Number of parallel workers (optional)
            num_threads: Threads per worker (optional)

        Returns:
            List of BatchResult objects, each containing filename and IntegrationResult.

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
        tiff_paths: list[str],
        n_radial: int,
        n_azimuthal: int,
        errors: bool = False,
        azimuthal_range: Optional[Tuple[float, float]] = None,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> BatchIterator:
        """
        Perform streaming batch 2D integration.

        Returns an iterator that yields results as each file completes.
        More memory-efficient for large batches.

        Args:
            tiff_paths: List of TIFF file paths to process
            n_radial: Number of radial bins
            n_azimuthal: Number of azimuthal bins
            errors: Whether to compute error estimates (default: False)
            azimuthal_range: Optional (min, max) tuple for azimuthal range in radians
            num_workers: Number of parallel workers (optional)
            num_threads: Threads per worker (optional)

        Returns:
            BatchIterator that yields BatchResult objects as they complete.
            The iterator has `total`, `completed`, and `progress` properties.

        Example:
            >>> for result in integrator.batch2d_iter(files, n_radial=100, n_azimuthal=360):
            ...     if result.is_ok():
            ...         cake = result.result.intensity
        """
        ...