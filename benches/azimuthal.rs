//! 2D integration on azimuthal ranges, warm and cold, through the three
//! public APIs. Needs the 0.13 range APIs, so it does not compile against
//! v0.12.0 (the start/step API alone is also in the older `integrator`
//! target).
//!
//! Input: the committed Pilatus 1M CeO2 image (1043 x 981) with `v1.poni`,
//! 1000 radial bins. IDs:
//!
//! - `azimuthal/range_sector/<method>`: `integrate2d_on_azimuthal_range` on a
//!   90 degree sector (-45 to 45 degrees, 90 rows), warm.
//! - `azimuthal/range_full/<method>`: the same API on the full circle (360
//!   rows), warm.
//! - `azimuthal/start_step/<method>`: `integrate2d_with_azimuthal_range`, the
//!   pyFAI-style first row centre and step, on the same sector, warm.
//! - `azimuthal/brackets/<method>`: `integrate2d_on_azimuthal_brackets`, rows
//!   given by their centres and half-width, on the same sector, warm.
//! - `azimuthal/build_sector/<method>`: cold, a new `Integrator` and its first
//!   sector integration; for `supersampled4` this includes the radial
//!   clamping of subpixels to the range.
//!
//! `<method>` is `pixel_split`, `supersampled4` (n_ss = 4) or `pyfai_full`.

mod common;

use std::f64::consts::PI;
use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use dioptrin::integration::{IntegrationMethod, Integrator};

const N_RADIAL: usize = 1000;
const N_AZIMUTHAL: usize = 360;

fn azimuthal(c: &mut Criterion) {
    let (geometry, frame) = common::ceo2();
    // A 90 degree sector across the +-pi/4 band, 90 one-degree rows.
    let sector = (-PI / 4.0, PI / 4.0);
    let n_sector = 90;
    let step = (sector.1 - sector.0) / n_sector as f64;
    let half_width = step / 2.0;
    let ranged = [
        ("pixel_split", IntegrationMethod::PixelSplit),
        ("supersampled4", IntegrationMethod::Supersampled { n_ss: 4 }),
        ("pyfai_full", IntegrationMethod::PyfaiFull),
    ];

    let mut group = c.benchmark_group("azimuthal");
    for (name, method) in ranged {
        let ai = Integrator::new(geometry.clone()).with_method(method);
        let run = || {
            ai.integrate2d_on_azimuthal_range(&frame, N_RADIAL, n_sector, sector, false)
                .unwrap()
        };
        black_box(run());
        group.bench_function(format!("range_sector/{name}"), |b| b.iter(run));

        let ai = Integrator::new(geometry.clone()).with_method(method);
        let run = || {
            ai.integrate2d_on_azimuthal_range(&frame, N_RADIAL, N_AZIMUTHAL, (-PI, PI), false)
                .unwrap()
        };
        black_box(run());
        group.bench_function(format!("range_full/{name}"), |b| b.iter(run));

        let ai = Integrator::new(geometry.clone()).with_method(method);
        let centres = (sector.0 + half_width, sector.1 + half_width);
        let run = || {
            ai.integrate2d_on_azimuthal_brackets(&frame, N_RADIAL, centres, half_width, false)
                .unwrap()
        };
        black_box(run());
        group.bench_function(format!("brackets/{name}"), |b| b.iter(run));
    }
    group.finish();

    let mut group = common::cold_group(c, "azimuthal/build_sector");
    for (name, method) in ranged {
        group.bench_function(name, |b| {
            b.iter_batched(
                || Integrator::new(geometry.clone()).with_method(method),
                |ai| {
                    let out = ai
                        .integrate2d_on_azimuthal_range(&frame, N_RADIAL, n_sector, sector, false)
                        .unwrap();
                    (out, ai)
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::config();
    targets = azimuthal
}
criterion_main!(benches);
