import dioptrin


def main() -> None:
    poni_path = "test_data/v1.poni"
    tiff_path = "test_data/CeO2_Pilatus1M.tif"
    files = [tiff_path, tiff_path]

    integrator = dioptrin.Integrator(poni_path, method="pixel_split")

    # 1D batch integration
    results_1d = integrator.batch1d(files, n_points=1000)
    ok_1d = sum(1 for r in results_1d if r.is_ok())
    print(f"1D batch ok: {ok_1d}/{len(results_1d)}")

    # 1D batch with errors
    results_1d_err = integrator.batch1d(files, n_points=1000, errors=True)
    ok_1d_err = sum(1 for r in results_1d_err if r.is_ok())
    print(f"1D+errors batch ok: {ok_1d_err}/{len(results_1d_err)}")

    # 2D batch integration
    results_2d = integrator.batch2d(files, n_radial=200, n_azimuthal=180)
    ok_2d = sum(1 for r in results_2d if r.is_ok())
    print(f"2D batch ok: {ok_2d}/{len(results_2d)}")

    # 2D batch with errors
    results_2d_err = integrator.batch2d(files, n_radial=200, n_azimuthal=180, errors=True)
    ok_2d_err = sum(1 for r in results_2d_err if r.is_ok())
    print(f"2D+errors batch ok: {ok_2d_err}/{len(results_2d_err)}")

    # 2D batch with azimuthal range
    results_2d_range = integrator.batch2d(
        files, n_radial=200, n_azimuthal=180, azimuthal_range=(0.0, 3.14)
    )
    ok_2d_range = sum(1 for r in results_2d_range if r.is_ok())
    print(f"2D range batch ok: {ok_2d_range}/{len(results_2d_range)}")

    # 2D batch with azimuthal range and errors
    results_2d_range_err = integrator.batch2d(
        files, n_radial=200, n_azimuthal=180, azimuthal_range=(0.0, 3.14), errors=True
    )
    ok_2d_range_err = sum(1 for r in results_2d_range_err if r.is_ok())
    print(f"2D range+errors batch ok: {ok_2d_range_err}/{len(results_2d_range_err)}")

    # Test streaming iterator
    print("\nStreaming iterator test:")
    for r in integrator.batch1d_iter(files, n_points=1000):
        status = "ok" if r.is_ok() else f"error: {r.error}"
        print(f"  {r.filename}: {status}")


if __name__ == "__main__":
    main()
