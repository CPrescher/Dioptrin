import argparse
import os
import sys

if not os.path.isdir("test_data/test_output"):
    print("Skipping: test_data/test_output/ not found (run examples locally first)")
    sys.exit(77)

import numpy as np
import fabio
import matplotlib.pyplot as plt
from pyFAI.integrator.azimuthal import AzimuthalIntegrator


def main(plot: bool = False) -> None:
    cake_rust = np.loadtxt("test_data/test_output/cake_lut.txt")
    cake_rust_array = cake_rust.reshape((360, 850))

    cake_rust_ss = np.loadtxt("test_data/test_output/cake_lut_ss.txt")
    cake_rust_ss_array = cake_rust_ss.reshape((360, 850))

    azimuthal_integrator = AzimuthalIntegrator()
    azimuthal_integrator.load("test_data/v1.poni")
    fabio_data = fabio.open("test_data/CeO2_Pilatus1M.tif").data[::-1]

    cake_pyfai, _, _ = azimuthal_integrator.integrate2d(
        fabio_data, 850, unit="2th_rad", polarization_factor=0.99
    )

    if plot:
        plt.figure(figsize=(9.5, 3.5))
        plt.subplot(131)
        plt.title("Dioptrin")
        plt.imshow(cake_rust_ss_array, origin="lower", vmin=100, vmax=300, aspect="auto")
        plt.subplot(132)
        plt.title("pyFAI")
        plt.imshow(cake_pyfai, origin="lower", vmin=100, vmax=300, aspect="auto")
        plt.subplot(133)
        plt.title("Relative Difference")
        plt.imshow(
            cake_pyfai / cake_rust_ss_array,
            origin="lower",
            vmin=0.9,
            vmax=1.1,
            aspect="auto",
        )
        plt.tight_layout()
        plt.show()
        plt.savefig("test_data/test_output/cake_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D integration comparison between Dioptrin and pyFAI")
    parser.add_argument("--plot", action="store_true", help="Generate and display plots")
    args = parser.parse_args()
    main(plot=args.plot)
