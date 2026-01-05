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
from pyFAI.geometry import Geometry

# %%
geometry = Geometry()
geometry.load("../test_data/v1.poni")

shape = (10, 20)
test_index = 65

# %% [markdown]
# # Checking test Outputs

# %% [markdown]
# ## Pixel Positions with detector as reference

# %%
d1_rust = np.loadtxt("../test_data/test_output/d1_cart.txt").reshape(shape)
d2_rust = np.loadtxt("../test_data/test_output/d2_cart.txt").reshape(shape)
figure(figsize=(6, 3))
subplot(1, 2, 1)
title("d1 Cartesian"); axis("off")
imshow(d1_rust, origin="lower", aspect="auto")
colorbar(location="bottom")
subplot(1, 2, 2)
title("d2 Cartesian"); axis("off")
imshow(d2_rust, origin="lower", aspect="auto")
colorbar(location="bottom")
tight_layout()

# %% [markdown]
# ## Pixel positions with PONI as reference

# %%
d1_rust = np.loadtxt("../test_data/test_output/d1_cart_poni.txt").reshape(shape)
d2_rust = np.loadtxt("../test_data/test_output/d2_cart_poni.txt").reshape(shape)
figure(figsize=(6, 3))
subplot(1, 2, 1)
title("d1 poni reference"); axis("off")
imshow(d1_rust, origin="lower", aspect="auto")
colorbar(location="bottom")
subplot(1, 2, 2)
title("d2 poni reference"); axis("off")
imshow(d2_rust, origin="lower", aspect="auto")
colorbar(location="bottom")
tight_layout()

# %% [markdown]
# ## Pixel position in x, y, z with sample as reference

# %%
x_rust = np.loadtxt("../test_data/test_output/x.txt").reshape(shape)
y_rust = np.loadtxt("../test_data/test_output/y.txt").reshape(shape)
z_rust = np.loadtxt("../test_data/test_output/z.txt").reshape(shape)

figure(figsize=(8, 3))
subplot(1, 3, 1)
title("x"); axis("off")
imshow(x_rust, origin="lower", aspect="auto")
colorbar(location="bottom")
subplot(1, 3, 2)
title("y"); axis("off")
imshow(y_rust, origin="lower", aspect="auto")
colorbar(location="bottom")
subplot(1, 3, 3)
title("z"); axis("off")
imshow(z_rust, origin="lower", aspect="auto")
colorbar(location="bottom")
tight_layout()

# %%
d1 = np.arange(0, shape[0])
d2 = np.arange(0, shape[1])
d2, d1 = np.meshgrid(d2, d1)
figure(figsize=(8, 3))
subplot(1, 2, 1)
title("d1"); axis("off")
imshow(d1, origin="lower", aspect="auto")
subplot(1, 2, 2)
title("d2"); axis("off")
imshow(d2, origin="lower", aspect="auto")
tight_layout

# %%
z, x, y = geometry.calc_pos_zyx(None, d1, d2) # does actually return zxy and not zyx
figure(figsize=(8, 3))
subplot(1, 3, 1)
title("x pyFAI"); axis("off")
imshow(x, origin="lower", aspect="auto")
subplot(1, 3, 2)
title("y pyFAI"); axis("off")
imshow(y, origin="lower", aspect="auto")
subplot(1, 3, 3)
title("z pyFAI"); axis("off")
imshow(z, origin="lower", aspect="auto")
tight_layout()

# %%
figure(figsize=(8, 3))
suptitle("Difference between pyFAI and Rust")
subplot(1, 3, 1)
title("x diff"); axis("off")
imshow(x-x_rust, origin="lower", aspect="auto")
subplot(1, 3, 2)
title("y diff"); axis("off")
imshow(y-y_rust, origin="lower", aspect="auto")
subplot(1, 3, 3)
title("z diff"); axis("off")
imshow(z-z_rust, origin="lower", aspect="auto")
tight_layout()

# %%
np.savetxt("../test_data/test_output/x_pyFAI.txt", x.flatten())
np.savetxt("../test_data/test_output/y_pyFAI.txt", y.flatten())
np.savetxt("../test_data/test_output/z_pyFAI.txt", z.flatten())

# %% [markdown]
# ## Two-Theta

# %%
tth_rust = np.loadtxt("../test_data/test_output/tth.txt").reshape(shape)
tth = geometry.twoThetaArray()
figure(figsize=(8, 3))
suptitle("Two Theta")
subplot(1, 3, 1)
title("pyFAI"); axis("off")
imshow(tth, origin="lower", aspect="auto")
subplot(1, 3, 2)
title("rust"); axis("off")
imshow(tth_rust, origin="lower", aspect="auto")
subplot(1, 3, 3)
title("diff"); axis("off")
imshow(tth_rust-tth, origin="lower", aspect="auto")
tight_layout()

# %%
np.savetxt("../test_data/test_output/tth_pyFAI.txt", tth.flatten())

# %% [markdown]
# ## Azimuthal angle

# %%
azimuth_rust = np.loadtxt("../test_data/test_output/azimuth.txt").reshape(shape)
azimuth = geometry.chiArray()
figure(figsize=(8, 3))
suptitle("Azimuthal Angle")
subplot(1, 3, 1)
title("pyFAI"); axis("off")
imshow(azimuth, origin="lower", aspect="auto")
subplot(1, 3, 2)
title("rust"); axis("off")
imshow(azimuth_rust, origin="lower", aspect="auto")
subplot(1, 3, 3)
title("diff"); axis("off")
imshow(azimuth_rust-azimuth, origin="lower", aspect="auto")
tight_layout()

# %%
np.savetxt("../test_data/test_output/azimuth_pyFAI.txt", azimuth.flatten())

# %% [markdown]
# # Solid Angle

# %%
sa = geometry.solidAngleArray()
sa_rust = np.loadtxt("../test_data/test_output/solid_angle.txt").reshape(shape)

figure(figsize=(8, 3))
suptitle("Solid Angle")
subplot(1, 3, 1)
title("pyFAI"); axis("off")
imshow(sa, origin="lower", aspect="auto")
subplot(1, 3, 2)
title("rust"); axis("off")
imshow(sa_rust, origin="lower", aspect="auto")
subplot(1, 3, 3)
title("diff"); axis("off")
imshow(sa_rust-sa, origin="lower", aspect="auto")
tight_layout()

# %%
np.savetxt("../test_data/test_output/solid_angle_pyFAI.txt", sa.flatten())

# %% [markdown]
# ## Polarization

# %%
pol = geometry.polarization(factor=0.99)
pol_rust = np.loadtxt("../test_data/test_output/polarization.txt").reshape(shape)

figure(figsize=(8, 3))
suptitle("Polarization")
subplot(1, 3, 1)
title("pyFAI"); axis("off")
imshow(pol, origin="lower", aspect="auto")
subplot(1, 3, 2)
title("rust"); axis("off")
imshow(pol_rust, origin="lower", aspect="auto")
subplot(1, 3, 3)
title("diff"); axis("off")
imshow(pol_rust-pol, origin="lower", aspect="auto")
tight_layout()

# %%
np.savetxt("../test_data/test_output/polarization_pyFAI.txt", pol.flatten())
