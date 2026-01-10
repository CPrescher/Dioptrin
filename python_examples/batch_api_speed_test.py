import argparse
import time

import dioptrin


def main() -> None:
    parser = argparse.ArgumentParser(description="Dioptrin Batch API speed test")
    parser.add_argument("--poni", default="test_data/v1.poni")
    parser.add_argument("--tiff", default="test_data/CeO2_Pilatus1M.tif")
    parser.add_argument("tiff_path", nargs="?", default=None)
    parser.add_argument("--files", type=int, default=20)
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--n-radial", type=int, default=200)
    parser.add_argument("--n-azimuthal", type=int, default=180)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument(
        "--method",
        default="pixel_split",
        choices=["counting", "pixel_split", "supersampled"],
        help="Integration method",
    )
    args = parser.parse_args()

    tiff_path = args.tiff_path or args.tiff
    files = [tiff_path for _ in range(args.files)]

    print(f"Loaded PONI: {args.poni}")
    print(f"Integration method: {args.method}")
    print(f"Files: {args.files}")
    if args.num_workers:
        print(f"Workers: {args.num_workers}")
    if args.num_threads:
        print(f"Threads: {args.num_threads}")
    print()

    integrator = dioptrin.Integrator(args.poni, method=args.method)
    # Warmup - build LUT
    integrator.integrate1d(files[0], args.n_points)

    # 1D batch integration
    start = time.perf_counter()
    results_1d = integrator.batch1d(
        files, args.n_points, num_workers=args.num_workers, num_threads=args.num_threads
    )
    elapsed = time.perf_counter() - start
    ok_1d = sum(1 for r in results_1d if r.is_ok())
    print(f"1D batch ok: {ok_1d}/{len(results_1d)}")
    print(f"1D batch time: {elapsed:.2f}s")
    if elapsed > 0:
        print(f"1D throughput: {len(results_1d) / elapsed:.2f} files/s")
    print()

    # 2D batch integration
    start = time.perf_counter()
    results_2d = integrator.batch2d(
        files,
        args.n_radial,
        args.n_azimuthal,
        num_workers=args.num_workers,
        num_threads=args.num_threads,
    )
    elapsed = time.perf_counter() - start
    ok_2d = sum(1 for r in results_2d if r.is_ok())
    print(f"2D batch ok: {ok_2d}/{len(results_2d)}")
    print(f"2D batch time: {elapsed:.2f}s")
    if elapsed > 0:
        print(f"2D throughput: {len(results_2d) / elapsed:.2f} files/s")
    print()

    # 1D streaming batch with progress
    print("1D streaming batch:")
    start = time.perf_counter()
    for r in integrator.batch1d_iter(
        files, args.n_points, num_workers=args.num_workers, num_threads=args.num_threads
    ):
        pass  # Just consume the iterator
    elapsed = time.perf_counter() - start
    print(f"1D streaming time: {elapsed:.2f}s")
    if elapsed > 0:
        print(f"1D streaming throughput: {len(files) / elapsed:.2f} files/s")


if __name__ == "__main__":
    main()
