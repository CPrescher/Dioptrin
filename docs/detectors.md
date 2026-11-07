# Physical detector geometry

`dioptrin::detector` owns numerical detector definitions independently of file
decoding. `dioptrin-io` retains original image data and acquisition metadata;
consumers construct the physical model before integration or fitting.

```rust
use dioptrin::{detector,geometry::Geometry,poni::Poni};
use std::sync::Arc;
let d=detector::named("Pilatus1M")?;
let geometry=Geometry::new(1043,981,Poni {
    pixel1:d.pixel1,pixel2:d.pixel2,distance:0.2,
    poni1:0.08,poni2:0.09,rot1:0.,rot2:0.,rot3:0.,
    wavelength:0.4e-10,spline_file:String::new(),
}).with_detector(Arc::new(d))?;
# Ok::<(),String>(())
```

Grid-boundary coordinates are used throughout: `(row+0.5,column+0.5)` is a pixel
centre. `Detector::position` returns `[z,slow,fast]` metres; it does not touch
image samples. `Geometry::angular_corners`/`corner_indices` retain independent
corners at module gaps and share grid vertices for contiguous maps. Cached
sampling transforms serve supersampling. Refinement uses the same physical
mapping in every residual and Jacobian evaluation.

`Detector` supports regular pixels, separable physical edge maps, independent
quadrilateral NeXus-style corner data, Dioptas permutations, pyFAI orientations
0–4, integer binning, manufacturer pitch tables and built-in masks. Attach a
validated immutable `Arc<Detector>` through `Geometry::with_detector`; use the
constructor again to replace it. `Geometry::new` alone retains the existing
uniform-pixel/scalar-PONI contract: the legacy `Poni::spline_file` string is not
a distortion evaluator. File-owning applications parse/load resources, construct
`detector::spline::Spline`, and attach the resulting detector. Named/oriented PONI
import must retain that descriptor, rather than round-tripping through scalar Poni.

Fit2D splines use native cubic tensor-product B-spline evaluation. Fixed-width
headers, region, knot counts/order, finite coefficients and pixel sizes are
validated. Knots/coefficients retain pyFAI's float32 import rounding; evaluation
uses float64 arithmetic. Solid-angle area differences and nonplanar normal
indexing follow pyFAI's image-index conventions. `fit2d::Fit2d` implements the
existing PONI conversion in mm/µm/degrees; wavelength stays SI. Arbitrary rot3
does not survive the Fit2D representation.

The compressed catalogue is a frozen **pyFAI 2026.9.0** data snapshot: 109
identities, 105 quadrilateral models, four unavailable six-corner Pixirad models.
It retains aliases, pitches, dimensions, masks, nonuniform/curved coordinates,
Perkin binning and official Rayonix binning pitches. Catalogue availability is
not a claim of arbitrary detector-configuration/API parity. Nondefault geometry,
unknown crops/binning, hexagonal splitting and parallax need explicit support.
See [pyFAI's detector documentation](https://www.silx.org/doc/pyFAI/latest/api/detectors/index.html).

`scripts/freeze_detectors.py` refreshes the data; `freeze_detector_reference.py`
produces independent reference samples and mask hashes. These development tools
require pyFAI/NumPy. Builds, catalogue lookup, spline evaluation, fitting and
integration do not. The MIT pyFAI notice is retained in
`../licenses/pyFAI-detectors.txt`; the real spline fixture comes unchanged from
Dioptas' `dioptas/tests/data/distortion/f4mnew.spline`.

Validation uses 105 model coordinate/mask fixtures (1e-7 m for float32 maps),
SHA-256 of every mask, real spline samples (1e-5 pixel against the reference's
float32 evaluator), Fit2D fixtures, detector permutations, and a nonuniform
synthetic refinement. Existing unit tests retain uniform-geometry and
supersampling regression coverage. This is scoped regression evidence, not
experimental detector accuracy certification. NeXus/PONI/project transport is
qualified separately by the Dioptas native consumer.
