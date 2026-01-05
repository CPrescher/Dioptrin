# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.9.12 ('GlassAnalysis')
#     language: python
#     name: python3
# ---

# %%
# %matplotlib widget
from pylab import *
import numpy as np
import fabio
from pyFAI.geometry import Geometry

# %%
fabio_data = fabio.open("../test_data/CeO2_Pilatus1M.tif").data[::-1]
frame = np.loadtxt("test_output/frame.txt")

# %%
figure(figsize=(8, 3))
subplot(1, 3, 1)
imshow(fabio_data, origin="lower", vmin=0, vmax=1000)
subplot(1, 3, 2)
imshow(frame.reshape(fabio_data.shape), origin="lower", vmin=0, vmax=1000)
subplot(1, 3, 3)
imshow(fabio_data - frame.reshape(fabio_data.shape), origin="lower", vmin=-100, vmax=100)
tight_layout()
