# Trackformer 1.1

Trackformer 1.1 is a causal tropical-cyclone route and intensity research
model. It predicts position, maximum sustained wind, central pressure, radius
of maximum wind, and R34/R50/R64 wind radii at six-hour leads through +120 h.

This is not an operational warning system. Do not use it for evacuation,
aviation, maritime, emergency-management, or other safety-critical decisions.

## Availability

- [GitHub release: Trackformer 1.1](https://github.com/yu314-coder/typhoon-predict/releases/tag/trackformer-1.1)
- [Hugging Face model](https://huggingface.co/euler314/typhoon-predict)
- [Live interactive demo](https://yu314-coder.github.io/typhoon-tracks.html)

The public repository intentionally contains the stable 1.1 release only.

## Causal Input Boundary

The model may use only information available at the forecast issue time or
earlier:

- observed IBTrACS track and intensity history;
- atmospheric analysis fields available at or before the issue time;
- causal nearby-storm and basin-context summaries;
- static geography and land/sea features when present in the prepared packet.

It does not use JMA, JTWC, ECMWF, GFS, GEFS, DeepMind, or another agency's
forecast track or positive-lead weather field as an input. Later observations
and best-track labels are used only for verification after the forecast is
frozen.

## Outputs

For 20 six-hour leads, the model returns:

| Output | Shape or unit |
| --- | --- |
| Track latitude and longitude | `20 x 2`, degrees |
| Maximum sustained wind | knots, 1-minute estimate |
| Central pressure | hPa |
| Radius of maximum wind | kilometres |
| R34, R50, and R64 | four quadrant radii in kilometres |

Radius outputs are non-negative and preserve the expected R34 >= R50 >= R64
ordering when the structure head is enabled.

## Hugging Face Files

The model repository keeps the 1.1 files under `models/trackformer_1_1/` and
includes the matching Python modules, calibration, normalization statistics,
and seed checkpoints. The released archive is the authoritative weight
package.

## Running It

Download the GitHub release archive or clone the Hugging Face model repository,
install the pinned requirements, and prepare an issue packet containing only
the causal inputs above. Use the included 1.1 prediction wrapper and preserve
the validity mask for missing radius or analysis values. Never substitute a
future observation or an official forecast for a missing input.

The live demo is a visualization and research interface, not a warning
service. Compare predictions with the later realized IBTrACS track only after
the issue-time forecast has been saved.
