import fabio
import numpy as np
import pyFAI
import dioptrin

dioptrin.set_rayon_threads(12)


def time_function(lambda_func, repetition, name=""):
    import time

    start = time.time()
    for _ in range(repetition):
        lambda_func()
    end = time.time()
    print(f"Time for {name}: {(end - start) / repetition:.6f} seconds")


# Load the image using fabio
image_path = "../test_data/CeO2_Pilatus1M.tif"
image_path = "../test_data/LaB6_OffCenter_PE.tif"
image_path = "../test_data/CeO2_120923_central_off_Y_Soller_out_CeO2_120923_central_off_Y_Soller_out_0001_scan0001_sum.tiff"
poni_path = "../test_data/v1.poni"
image = fabio.open(image_path).data[::-1]
image_c = np.ascontiguousarray(image, dtype=np.float64)

ai = pyFAI.load(poni_path)

time_function(
    lambda: ai.integrate1d(image, 1000, unit="2th_deg"),  # Warm-up
    1,
    "pyFAI integrate1d warm-up",
)

integrator = dioptrin.Integrator.from_poni_file(poni_path, "ss", 0.99, 5)
time_function(
    lambda: integrator.integrate1d_array(image_c, 1000),  # Warm-up
    1,
    "dioptrin integrate1d_array warm-up",
)

time_function(
    lambda: ai.integrate1d(image, 1000, unit="2th_deg"),
    100,
    "pyFAI integrate1d",
)

time_function(
    # lambda: integrator.integrated1d_array_with_errors(np.ascontiguousarray(image, np.float64), 1000),
    lambda: integrator.integrate1d_with_errors(image_path, 1000),
    100,
    "dioptrin integrate1d_with_errors from file:",
)

time_function(
    lambda: integrator.integrate1d_array_with_errors(
        np.ascontiguousarray(fabio.open(image_path).data[::-1], np.float64), 1000
    ),
    100,
    "dioptrin integrate1d_array_with_errors from array:",
)
