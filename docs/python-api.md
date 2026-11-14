# Python API

## Usage and settings

```{include} ../README.md
:start-after: "## Python Bindings"
:end-before: "<!-- manual-python-end -->"
```

## Typed reference

The complete Python signatures are also available in the
{download}`type stub <../dioptrin.pyi>` and in the
[source repository](https://github.com/CPrescher/Dioptrin/blob/main/dioptrin.pyi).

## Combined detector cakes

`MultiIntegrator.integrate2d(images, n_radial, n_azimuthal, errors=False,
azimuthal_range=None, *, normalization="relative", normalization_factors=None)`
combines raw fractional contributions on common radial and azimuthal edges.
Both coordinate arrays are populated in its `IntegrationResult`; flat outputs
reshape to `(n_azimuthal, n_radial)`. Its `azimuthal_range` is a pair of edges
in radians, as for `Integrator.integrate2d`. For the pyFAI MultiGeometry absolute
solid-angle scale, set `normalization="absolute"`. See the
[multi-detector guide](MULTI_DETECTOR.md) for units, masks, uncertainties and
important splitting/legacy normalization differences.
