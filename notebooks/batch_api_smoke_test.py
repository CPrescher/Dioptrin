import dioptrin


def main() -> None:
    poni_path = "test_data/v1.poni"
    tiff_path = "test_data/CeO2_Pilatus1M.tif"
    files = [tiff_path, tiff_path]

    integrator = dioptrin.Integrator(poni_path, dim1=1043, dim2=981, method="pixel_split")

    results_1d = integrator.batch_integrate1d(files, n_points=1000)
    ok_1d = sum(1 for _, result, err in results_1d if result is not None and err is None)
    print(f"1D batch ok: {ok_1d}/{len(results_1d)}")

    results_1d_err = integrator.batch_integrate1d_with_errors(files, n_points=1000)
    ok_1d_err = sum(1 for _, result, err in results_1d_err if result is not None and err is None)
    print(f"1D+errors batch ok: {ok_1d_err}/{len(results_1d_err)}")

    results_2d = integrator.batch_integrate2d(files, n_radial=200, n_azimuthal=180)
    ok_2d = sum(1 for _, result, err in results_2d if result is not None and err is None)
    print(f"2D batch ok: {ok_2d}/{len(results_2d)}")

    results_2d_err = integrator.batch_integrate2d_with_errors(files, n_radial=200, n_azimuthal=180)
    ok_2d_err = sum(
        1 for _, result, err in results_2d_err if result is not None and err is None
    )
    print(f"2D+errors batch ok: {ok_2d_err}/{len(results_2d_err)}")

    results_2d_range = integrator.batch_integrate2d_with_azimuthal_range(
        files, n_radial=200, n_azimuthal=180, azimuthal_min=0.0, azimuthal_max=3.14
    )
    ok_2d_range = sum(
        1 for _, result, err in results_2d_range if result is not None and err is None
    )
    print(f"2D range batch ok: {ok_2d_range}/{len(results_2d_range)}")

    results_2d_range_err = integrator.batch_integrate2d_with_azimuthal_range_and_errors(
        files, n_radial=200, n_azimuthal=180, azimuthal_min=0.0, azimuthal_max=3.14
    )
    ok_2d_range_err = sum(
        1 for _, result, err in results_2d_range_err if result is not None and err is None
    )
    print(f"2D range+errors batch ok: {ok_2d_range_err}/{len(results_2d_range_err)}")


if __name__ == "__main__":
    main()
