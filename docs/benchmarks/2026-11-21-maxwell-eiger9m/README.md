# Eiger 9M integration on Maxwell

A prepared Dioptrin service published all 625 intensity and Poisson-error patterns from a completed 3,000-bin scan in a median **1.960 s** on one Maxwell CPU node. Six arrivals ranged from 1.925 to 1.987 s. This measures latency for one file arriving at an idle watcher. Sustained acquisition with a growing queue is a separate workload.

## From a completed scan to saved patterns

The input was a 5.69 GB Bitshuffle/LZ4 HDF5 scan of 625 distinct 3,262 × 3,108 uint32 frames. The node had two AMD EPYC 7402 processors, each with 24 physical cores and two hardware threads per core. Four processes each used 12 physical cores and both hardware threads. Each process allocated its working memory on its socket and ran four native readers, for 16 readers in total.

| Radial bins | Output | All patterns available in RAM | Result published on GPFS |
| ---: | --- | ---: | ---: |
| 2,000 | Intensity | 1.798 s | 1.817 s |
| 2,000 | Intensity and Poisson sigma | 1.819 s | 1.866 s |
| 3,000 | Intensity | 1.892 s | 1.918 s |
| 3,000 | Intensity and Poisson sigma | 1.901 s | 1.960 s |

Each reader built its lookup table (LUT) before any scan arrived. At 3,000 bins, one-time service startup took a median 5.05 s. A producer on the same GPFS client atomically renamed each completed scan into the inbox. The service polled every 50 ms and used `set_source` to open and index the file while retaining its LUT. Timing started immediately before the rename. The last column includes writing and closing an uncompressed NumPy archive and renaming it into place. It measures visibility of the result to consumers. The test used no explicit fsync.

Three randomized blocks covered seven layouts, both bin counts and both error modes. Each table cell summarizes six arrivals. A copy of the original scan and a second file containing its compressed frames in reverse order exercised source replacement. All 168 saved archives passed frame-by-frame intensity and requested sigma checks against a reference integrated from independently decoded pixels. The largest absolute difference was 5.36 × 10⁻⁹. A successful per-file GPFS cache-clear request, zero Linux-resident input pages and about 5.743 GB received over InfiniBand per arrival verified a cold payload on the compute client. Storage-controller cache state was unobserved.

**Memory placement affected file latency.** With two processes and 16 readers fixed, changing allocations from interleaved to socket-local reduced the 3,000-bin Poisson median from 2.657 s to 2.091 s saved. Two further controls give context for the prepared service:

| 3,000 bins with Poisson errors | Median to patterns in RAM | Timed work |
| --- | ---: | --- |
| Compressed file cached, 16 prepared readers | 1.20 s | Decode, integrate and assemble all patterns |
| Client-cold file, four processes and four newly constructed readers | 4.19 s | Open, build LUTs, decode and integrate |

A separate single-reader control took 36 ms to open and index the dataset. Reading the compressed bytes sequentially from a cold client, without decoding, took 1.525 s. These controls have different access patterns and cannot be added to decompose the watcher latency.

## In-memory library comparison

The CPU sweeps started with all 625 decoded float64 frames in RAM and warmed integration engines. They used 1, 2, 4, 8, 12, 24, 48 and 96 workers with 2,000 and 3,000 bins. The table gives medians of three complete scans at 3,000 bins. Each cell shows intensity-only / Poisson frames per second.

| Library and splitting method | 24 workers | 48 workers | 96 workers |
| --- | ---: | ---: | ---: |
| Dioptrin 0.14.0, full splitting | 479.9 / 405.9 | 462.6 / 479.9 | 470.9 / 396.0 |
| pyFAI 2026.9.0, full splitting | 74.6 / 73.5 | 103.5 / 101.6 | 116.7 / 115.8 |
| Dioptrin 0.14.0, 5 × 5 subpixels | 525.9 / 438.7 | 449.5 / 519.7 | 505.8 / 284.9 |
| azint 1.2.6.dev6+g4f78cc3b1, 5 × 5 subpixels | 140.6 / 99.4 | 119.6 / 76.8 | 133.3 / 95.4 |
| azint, fixed-mask normalization cached | 241.6 / 115.6 | 357.6 / 174.9 | 365.7 / 181.2 |

A separate seven-repeat **Dioptrin-only** placement experiment used full splitting. With plan and frame storage local to each socket, going from one to two 24-core processes raised the 3,000-bin rates from 446 to 850 frames/s for intensity and from 398 to 754 with Poisson errors. In the subsequent hardware-thread test, four socket-local processes using all 96 threads reached 1,105 / 1,015 frames/s at 3,000 bins and 1,398 / 1,229 at 2,000 bins. The other libraries were measured in the original sweep. The four-process results quantify Dioptrin tuning and do not establish a like-for-like tuned library speedup.

The RAM library comparison used 951,372 excluded pixels, solid-angle correction, no polarization correction and a shared radial interval of 0–44.232582° in two-theta. The file watcher used the archived Mac protocol's 887,588-pixel mask, polarization factor 0.965 and full splitting. Its dioptrin-io 0.4.1 fused adapter links Dioptrin 0.10.0. Timing differences between the RAM and file protocols do not isolate decompression cost. [Native adapter revision](https://github.com/CPrescher/dioptrin-io/tree/97f5ba73ca1f8b169ce3037ff79b0a51b40f6098/python-fused)

### Numerical agreement

For Dioptrin and pyFAI full splitting, the maximum frame-wise relative L2 difference in intensity was 0.102% at 3,000 bins and 0.426% at 2,000 bins. The corresponding maxima for reported Poisson sigma were 20.1% and 67.6%. The second radial bin accounted for 97.4% and 99.7% of the total squared sigma discrepancy, respectively. The saved pyFAI arrays had negative sigma entries in that bin for all 625 frames at both bin counts. The cause remains open. For Dioptrin and azint with 5 × 5 subpixels, the maximum intensity-profile difference at 3,000 bins was 2.30%. Timing comparisons should retain these qualifications.

## GPU and Mac context

A separate 625-frame file harness streamed the input before each fresh trial and included process startup. Native CUDA took a median 9.01 s on the A100 PCIe 40 GB node and 4.64 s on the H200 NVL node. The corresponding CPU CSR controls took 7.81 s and 4.18 s on their respective hosts. These figures include host work, and the GPU nodes and file workflow differ from the prepared CPU watcher. The earlier M4 Pro measurements used the archived 2,000-bin protocol with an external decoder. [Archived Mac protocol](../2026-11-05-full-file-mask-modes/bench_full_file.py)

## Running the watcher

The measured watcher is included with this record. On a Linux allocation with two NUMA sockets, install the native adapter at the revision above, NumPy, h5py and FabIO, and make `numactl` available. The following example uses the calibration and mask from the repository. Set the three paths to a completed matching template scan, an inbox and a dedicated output directory.

```sh
python docs/benchmarks/2026-11-21-maxwell-eiger9m/watch_eiger.py \
  --template "$EIGER_TEMPLATE" --inbox "$EIGER_INBOX" --output "$EIGER_OUTPUT" \
  --poni docs/benchmarks/2026-11-05-bslz4/calibration.poni \
  --mask test_data/eiger_0000.mask --polarization 0.965 \
  --bins 3000 --poisson --processes 4 --readers 16 --policy local
```

Publish closed files under unique final `.h5` names. A new calibration, mask or binning configuration needs fresh LUTs and its own output directory. The template must use compatible detector geometry and stored row orientation. The measured source scan's SHA-256 was `2b749693d7a942b35d89806d99b9a1025ee119850be0ade27428f1dc11d0a1f3`. [Watcher source](watch_eiger.py) · [Calibration](../2026-11-05-bslz4/calibration.poni) · [Mask](../../../test_data/eiger_0000.mask)
