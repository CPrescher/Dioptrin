"""Type stubs for dioptrin - High-performance 1D/2D diffraction integration library."""

from typing import Tuple, Optional
import numpy as np

__all__ = [
    "Integrator",
    "integrate1d",
    "integrate1d_with_errors",
    "rayon_num_threads",
    "set_rayon_threads",
]


def integrate1d(
    poni_path: str,
    tiff_path: str,
    n_points: int,
    method: Optional[str] = None,
    polarization_factor: Optional[float] = None,
    n_ss: Optional[int] = None,
    unit: Optional[str] = None,
) -> Tuple[list[float], list[float]]:
    """
    Perform 1D radial integration of a diffraction pattern.

    This function loads a PONI calibration file and a TIFF image, then performs
    1D integration along the radial (scattering angle) axis.

    Args:
        poni_path: Path to the PONI calibration file (v1 or v2 format)
        tiff_path: Path to the TIFF image file containing the diffraction pattern
        n_points: Number of bins for the 1D integration output
        method: Integration method: "pixel_split" (default), "counting", "histogram", or "supersampled"
        polarization_factor: Optional polarization correction factor (0.0 to 1.0)
        n_ss: Number of subpixels per pixel for supersampled method (default: 5)
        unit: Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"

    Returns:
        Tuple of (radial_values, intensities) as lists of floats

    Example:
        >>> import dioptrin
        >>> radial, intensity = dioptrin.integrate1d(
        ...     "calibration.poni",
        ...     "pattern.tif",
        ...     n_points=1000,
        ...     method="pixel_split",
        ...     unit="2th_deg"
        ... )
    """
    ...


def integrate1d_with_errors(
    poni_path: str,
    tiff_path: str,
    n_points: int,
    method: Optional[str] = None,
    polarization_factor: Optional[float] = None,
    n_ss: Optional[int] = None,
    unit: Optional[str] = None,
) -> Tuple[list[float], list[float], list[float]]:
    """
    Perform 1D radial integration with error propagation.

    Similar to `integrate1d`, but also returns error estimates (standard deviations)
    for each bin based on Poisson statistics.

    Args:
        poni_path: Path to the PONI calibration file (v1 or v2 format)
        tiff_path: Path to the TIFF image file containing the diffraction pattern
        n_points: Number of bins for the 1D integration output
        method: Integration method: "pixel_split" (default), "counting", "histogram", or "supersampled"
        polarization_factor: Optional polarization correction factor (0.0 to 1.0)
        n_ss: Number of subpixels per pixel for supersampled method (default: 5)
        unit: Radial unit: "2th_rad" (default), "2th_deg", "q_A^-1", or "q_nm^-1"

    Returns:
        Tuple of (radial_values, intensities, errors) as lists of floats

    Example:
        >>> import dioptrin
        >>> radial, intensity, errors = dioptrin.integrate1d_with_errors(
        ...     "calibration.poni",
        ...     "pattern.tif",
        ...     n_points=1000
        ... )
    """
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


class Integrator:
    """
    Integrator for 1D radial integration of diffraction patterns.

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
        self, tiff_path: str, n_points: int
    ) -> Tuple[list[float], list[float]]:
        """
        Perform 1D radial integration from a TIFF file.

        Loads a TIFF image and integrates it along the radial axis using the
        configured integration method and corrections.

        Args:
            tiff_path: Path to the TIFF image file
            n_points: Number of bins for the 1D integration output

        Returns:
            Tuple of (radial_values, intensities) as lists of floats

        Example:
            >>> integrator = dioptrin.Integrator("calibration.poni")
            >>> radial, intensity = integrator.integrate1d("pattern.tif", n_points=1000)
        """
        ...

    def integrate1d_array(
        self, array: np.ndarray, n_points: int
    ) -> Tuple[list[float], list[float]]:
        """
        Perform 1D radial integration from a NumPy array.

        Integrates a 2D NumPy array (detector image) along the radial axis.
        The array must be C-contiguous (row-major) and of type float64.

        Args:
            array: 2D NumPy array of shape (dim1, dim2) containing the diffraction pattern
            n_points: Number of bins for the 1D integration output

        Returns:
            Tuple of (radial_values, intensities) as lists of floats

        Raises:
            ValueError: If the array is not C-contiguous or not float64

        Example:
            >>> import numpy as np
            >>> integrator = dioptrin.Integrator("calibration.poni")
            >>> data = np.array(image_data, dtype=np.float64)
            >>> radial, intensity = integrator.integrate1d_array(data, n_points=1000)
        """
        ...

    def integrate1d_array_with_errors(
        self, array: np.ndarray, n_points: int
    ) -> Tuple[list[float], list[float], list[float]]:
        """
        Perform 1D radial integration from a NumPy array with error propagation.

        Similar to `integrate1d_array`, but also returns error estimates
        (standard deviations) for each bin based on Poisson statistics.

        Args:
            array: 2D NumPy array of shape (dim1, dim2) containing the diffraction pattern
            n_points: Number of bins for the 1D integration output

        Returns:
            Tuple of (radial_values, intensities, errors) as lists of floats

        Raises:
            ValueError: If the array is not C-contiguous or not float64

        Example:
            >>> import numpy as np
            >>> integrator = dioptrin.Integrator("calibration.poni")
            >>> data = np.array(image_data, dtype=np.float64)
            >>> radial, intensity, errors = integrator.integrate1d_array_with_errors(data, n_points=1000)
        """
        ...

    def integrate1d_with_errors(
        self, tiff_path: str, n_points: int
    ) -> Tuple[list[float], list[float], list[float]]:
        """
        Perform 1D radial integration from a TIFF file with error propagation.

        Similar to `integrate1d`, but also returns error estimates
        (standard deviations) for each bin based on Poisson statistics.

        Args:
            tiff_path: Path to the TIFF image file
            n_points: Number of bins for the 1D integration output

        Returns:
            Tuple of (radial_values, intensities, errors) as lists of floats

        Example:
            >>> integrator = dioptrin.Integrator("calibration.poni")
            >>> radial, intensity, errors = integrator.integrate1d_with_errors("pattern.tif", n_points=1000)
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

    def batch_integrate1d(
        self,
        tiff_paths: list[str],
        n_points: int,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> list[tuple[str, Optional[Tuple[list[float], list[float]]], Optional[str]]]:
        """
        Perform batch 1D radial integration over multiple TIFF files.

        Returns:
            List of (filename, result, error) tuples. Result is None when error is set.
        """
        ...

    def batch_integrate1d_with_errors(
        self,
        tiff_paths: list[str],
        n_points: int,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> list[tuple[str, Optional[Tuple[list[float], list[float], list[float]]], Optional[str]]]:
        """
        Perform batch 1D radial integration with errors over multiple TIFF files.

        Returns:
            List of (filename, result, error) tuples. Result is None when error is set.
        """
        ...

    def batch_integrate2d(
        self,
        tiff_paths: list[str],
        n_radial: int,
        n_azimuthal: int,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> list[tuple[str, Optional[list[float]], Optional[str]]]:
        """
        Perform batch 2D radial integration over multiple TIFF files.

        Returns:
            List of (filename, result, error) tuples. Result is None when error is set.
        """
        ...

    def batch_integrate2d_with_errors(
        self,
        tiff_paths: list[str],
        n_radial: int,
        n_azimuthal: int,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> list[tuple[str, Optional[Tuple[list[float], list[float]]], Optional[str]]]:
        """
        Perform batch 2D radial integration with errors over multiple TIFF files.

        Returns:
            List of (filename, result, error) tuples. Result is None when error is set.
        """
        ...

    def batch_integrate2d_with_azimuthal_range(
        self,
        tiff_paths: list[str],
        n_radial: int,
        n_azimuthal: int,
        azimuthal_min: float,
        azimuthal_max: float,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> list[tuple[str, Optional[list[float]], Optional[str]]]:
        """
        Perform batch 2D integration with an azimuthal range over multiple TIFF files.

        Returns:
            List of (filename, result, error) tuples. Result is None when error is set.
        """
        ...

    def batch_integrate2d_with_azimuthal_range_and_errors(
        self,
        tiff_paths: list[str],
        n_radial: int,
        n_azimuthal: int,
        azimuthal_min: float,
        azimuthal_max: float,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
    ) -> list[tuple[str, Optional[Tuple[list[float], list[float]]], Optional[str]]]:
        """
        Perform batch 2D integration with azimuthal range and errors over multiple TIFF files.

        Returns:
            List of (filename, result, error) tuples. Result is None when error is set.
        """
        ...
