# Offline license enforcement

Licensed Python wheels keep the signed license payload and signature inside each
native `Integrator`. Construction loads this evidence directly; it does not use
the result of the public diagnostic `validate_license()` function. The evidence
is private native state, not a Python attribute or a cached `licensed` boolean.
Acquisition uses strict Ed25519 verification in `ed25519-dalek`; operation
admission uses the independent Ed25519 verifier in `ring`. Forcing one signature
backend to return success therefore does not authorize both stages.

At the start of each 1D, 2D, batch, and streaming-batch call, the binding verifies
the retained signature against the compiled public key and checks expiration.
`MultiIntegrator` validates and retains evidence for every constituent detector,
and checks that evidence before each integration or batch call. High-level grant
checks are requested inline at these separate entry points. Numerical kernels,
lookup tables, normalization, and returned scientific values are unchanged.

This is defense in depth for the distributed Python wheel. A fabricated success
return from the diagnostic verifier cannot provide a usable entitlement. Skipping
only the constructor's verification also leaves operation checks in place. These
checks do not make a user-controlled process tamper-proof: a determined attacker
can modify other native code, replace a public key, or patch multiple entry
points. Inlining and stripping are not cryptographic security boundaries. The
Rust calculation API and explicitly unlicensed development builds are unchanged.

## Offline and lifetime behavior

- No network, activation server, periodic online lease, or machine binding.
- Existing normally signed licenses keep their format and public verification key.
- Diagnostics/acquisition use strict Ed25519 verification; operation checks use
  independent Ed25519 verification against the same compiled public key.
- An admitted object retains its original signed evidence. Removing or replacing
  the license file/environment variable does not change that object's evidence.
  Construct a new integrator to pick up a renewed license.
- Expiration is checked for each new operation, using the existing UTC-day policy
  (valid through the specified date). Local clock rollback is not prevented.
- A batch is authorized when submitted. An already admitted batch/iterator can
  finish after midnight, including subsequently consumed generator frames; checks
  are not performed inside pixel loops or on every batch frame. This avoids
  interrupting an acquisition and amortizes signature verification over the batch.
- Invalid evidence fails with the existing license exception family, before
  numerical work is dispatched. No altered or silently degraded numerical output
  is used as an enforcement mechanism.

## Regression checks

`cargo test --features licensed` exercises the Rust calculations and entitlement
unit tests. The unit tests use their own test signing keys and explicit times to
check fabricated proof, payload tampering, and expiration of retained evidence.

`scripts/check_licensed_install.py` tests an installed production wheel in child
processes with an empty home/config, absent or invalid `DIOPTRIN_LICENSE`, and
Python isolation. Both constructors must reject even after monkeypatching the
Python validators. This script is run by the release-wheel matrix.

Run `python scripts/test_license_hardening.py` in a test venv containing `maturin`,
`numpy`, and `cryptography`, with Cargo available. It builds wheels in temporary
source copies using a newly generated test public key. The private test key stays
in memory. No production license or signing key is needed or read, and no test
key, bypass feature, or special acceptance path is added to the release code.
The temporary wheels are never uploaded by this test workflow.

The test covers:

1. Authorized offline numerical integration, an independent NumPy counting
   reference, scaling, method changes, batches, and multiple detectors.
2. A native diagnostic stub returning empty successful metadata, modeling the
   effect of the previously demonstrated binary patch, plus a forced-success
   operation verifier. Constructors still reject missing, malformed, and
   structurally valid but incorrectly signed licenses.
3. Deliberately bypassed constructor checks with fabricated or expired evidence.
   All eight integration/batch entry points must still reject.

Source fault injection makes those regressions portable across architectures. It
is not a claim that arbitrary binary edits are prevented; release-binary attack
reviews remain useful alongside these tests. A failure to apply a fixture edit
is an error, not a skipped test.

## Local validation snapshot (macOS ARM64, CPython 3.13)

The initial implementation check passed 158 Rust tests and 16 doctests, licensed
release-wheel checks, and all temporary-wheel fault-injection cases. Both
license-file and environment activation were exercised with generated test keys.
A separate binary-level check forced the diagnostic to return success; both
constructors still rejected missing licenses. Another binary modification forced
the constructor signature backend to accept a zero signature: construction then
succeeded, but all six single-integrator operation entry points and construction
of a multi-integrator still rejected that evidence. These are targeted regression
checks, not an independent adversarial assessment of the completed hardening.

A small warmed-cache microbenchmark used counting integration, 64 output bins,
one Rayon thread, and the median of five groups of 100 calls. The unlicensed and
licensed builds respectively took 12.51 / 38.03 microseconds for 64x64 frames and
181.76 / 215.95 microseconds for 512x512 frames. This is a local observational
measurement, not a portable performance guarantee. Signature checks add tens of
microseconds per direct call; batches amortize one check across their frames.
