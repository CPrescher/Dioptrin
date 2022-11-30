# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %matplotlib widget
from pylab import *
import numpy as np
import fabio
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

# %%
tth_rust = np.loadtxt("../test_data/test_output/pattern_tth.txt")
int_rust = np.loadtxt("../test_data/test_output/pattern_int.txt")
tth_rust_csr = np.loadtxt("../test_data/test_output/pattern_tth_csr.txt")
int_rust_csr = np.loadtxt("../test_data/test_output/pattern_int_csr.txt")
tth_rust_csr_ss = np.loadtxt("../test_data/test_output/pattern_tth_csr_ss.txt")
int_rust_csr_ss = np.loadtxt("../test_data/test_output/pattern_int_csr_ss.txt")

# %%
azi = AzimuthalIntegrator()
azi.load("../test_data/v1.poni")
fabio_data = fabio.open("../test_data/CeO2_Pilatus1M.tif").data[::-1]

tth_pyFAI, int_pyFAI = azi.integrate1d(fabio_data, len(int_rust), unit="2th_rad", method=("no", "histogram", "cython"), polarization_factor=0.99)
tth_pyFAI_csr, int_pyFAI_csr = azi.integrate1d(fabio_data, len(int_rust), unit="2th_rad", method=("no", "csr", "cython"), polarization_factor=0.99)

# %%
figure()
plot(tth_rust, int_rust, label="rust")
plot(tth_pyFAI, int_pyFAI, label="pyFAI")
plot(tth_pyFAI_csr, int_pyFAI_csr, label="pyFAI csr")
plot(tth_rust_csr_ss, int_rust_csr_ss, label="rust_csr_ss")
legend()
print(len(tth_rust_csr_ss))

# %%
np.savetxt("../test_data/test_output/pattern_tth_pyFAI.txt", tth_pyFAI)
np.savetxt("../test_data/test_output/pattern_int_pyFAI.txt", int_pyFAI)

# %%
cake_rust = np.loadtxt("../test_data/test_output/cake_csr.txt")
cake_rust_array = cake_rust.reshape((360, 850))
cake_pyfai, _, _ = azi.integrate2d(fabio_data, 850, unit="2th_rad", polarization_factor=0.99)

# %%
figure(figsize=(10, 3.5))
subplot(131)
title('Dioptrin')
imshow(cake_rust_array, origin='lower', vmin=0, vmax=5300, aspect='auto')
subplot(132)
title('pyFAI')
imshow(cake_pyfai, origin='lower', vmin=0, vmax=5300, aspect='auto')
subplot(133)
title('Difference')
imshow(cake_pyfai/cake_rust_array, origin='lower', vmin=0.9, vmax=1.1, aspect='auto')
tight_layout()


# %%
# %%timeit
cake_pyfai, _, _ = azi.integrate2d(fabio_data, 850, unit="2th_rad", polarization_factor=0.99)
