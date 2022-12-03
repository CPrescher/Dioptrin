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
# %matplotlib notebook
from pylab import *
import numpy as np
import fabio
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

# %%
tth_rust = np.loadtxt("../test_data/test_output/pattern_tth.txt")
int_rust = np.loadtxt("../test_data/test_output/pattern_int.txt")
tth_rust_lut = np.loadtxt("../test_data/test_output/pattern_tth_lut.txt")
int_rust_lut = np.loadtxt("../test_data/test_output/pattern_int_lut.txt")
tth_rust_lut_ss = np.loadtxt("../test_data/test_output/pattern_tth_lut_ss.txt")
int_rust_lut_ss = np.loadtxt("../test_data/test_output/pattern_int_lut_ss.txt")

# %%
azi = AzimuthalIntegrator()
azi.load("../test_data/v1.poni")
fabio_data = fabio.open("../test_data/CeO2_Pilatus1M.tif").data[::-1]

tth_pyFAI, int_pyFAI = azi.integrate1d(fabio_data, len(int_rust), unit="2th_rad", method=("no", "histogram", "cython"), polarization_factor=0.99)
tth_pyFAI_lut, int_pyFAI_lut = azi.integrate1d(fabio_data, len(int_rust), unit="2th_rad", method=("no", "lut", "cython"), polarization_factor=0.99)

# %%
figure()
plot(tth_rust, int_rust, label="rust")
plot(tth_pyFAI, int_pyFAI, label="pyFAI")
plot(tth_pyFAI_lut, int_pyFAI_lut, label="pyFAI lut")
plot(tth_rust_lut_ss, int_rust_lut_ss, label="rust_lut_ss")
legend()
print(len(tth_rust_lut_ss))

# %%
np.savetxt("../test_data/test_output/pattern_tth_pyFAI.txt", tth_pyFAI)
np.savetxt("../test_data/test_output/pattern_int_pyFAI.txt", int_pyFAI)

# %%
cake_rust = np.loadtxt("../test_data/test_output/cake_lut.txt")
cake_rust_array = cake_rust.reshape((360, 850))

cake_rust_ss =  np.loadtxt("../test_data/test_output/cake_lut_ss.txt")
cake_rust_ss_array = cake_rust_ss.reshape((360, 850))

cake_pyfai, _, _ = azi.integrate2d(fabio_data, 850, unit="2th_rad", polarization_factor=0.99)

# %%
figure(figsize=(9.5, 3.5))
subplot(131)
title('Dioptrin')
imshow(cake_rust_ss_array, origin='lower', vmin=100, vmax=300, aspect='auto')
subplot(132)
title('pyFAI')
imshow(cake_pyfai, origin='lower', vmin=100, vmax=300, aspect='auto')
subplot(133)
title('Difference')
imshow(cake_pyfai/cake_rust_ss_array, origin='lower', vmin=0.9, vmax=1.1, aspect='auto')
tight_layout()


# %%
# %%timeit
cake_pyfai, _, _ = azi.integrate2d(fabio_data, 850, unit="2th_rad", polarization_factor=0.99)

# %%
