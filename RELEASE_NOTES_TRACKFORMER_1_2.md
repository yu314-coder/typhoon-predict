# Trackformer 1.2 — pressure-field research model

**[Download weights + complete inference code](https://github.com/yu314-coder/typhoon-predict/releases/download/trackformer-1.2/trackformer_1_2_field_pressure_export_v2.tar.gz)** · **[Hugging Face](https://huggingface.co/euler314/typhoon-predict)** · **[Technical paper](https://github.com/yu314-coder/typhoon-predict/blob/main/paper/trackformer.pdf)**

Trackformer 1.2 forecasts Western Pacific storm tracks, central pressure and evolving sea-level-pressure fields through +120 hours. A multiscale environmental-attention network conditions the moving pressure core; track and central pressure are read from that evolving field.

## Direct detailed pressure-map output

The current package exports all twenty six-hour leads with:

- Whole-WP basin pressure, the original fixed regional composite, and the actual **65×65 moving-core pressure field in physical hPa**.
- Basin, regional and per-lead core latitude/longitude grids, coverage masks, issue time and exact valid times.
- Original track/central-pressure outputs and an explicit **one-member** count.
- Frozen weight/checkpoint identity and an optional PNG renderer: blue low pressure, red high pressure, labelled isobars.

```bash
python models/trackformer_1_2_field/predict.py causal_issue_packet.npz forecast.npz --device cpu --pressure-map pressure_120h.png --map-lead 120
python models/trackformer_1_2_field/plot_pressure.py forecast.npz pressure_24h.png --lead 24 --interval 2
```

NumPy and PyTorch are required for inference; Matplotlib is optional for images. Use `--device mps` or `--device cuda` on a compatible system. Prepare the nine-analysis causal input packet using the [published schema](https://github.com/yu314-coder/typhoon-predict/blob/main/models/trackformer_1_2_field/README.md); this package does not fetch live weather automatically.

The core's 20-km spacing is a learned computational reconstruction, not new native observations. Invalid coverage remains masked. Fields are not shifted onto a route, and central pressure is not inserted as a display vortex. Ensemble members require geographic registration before physical-field averaging; this command is not the separate 50-member benchmark policy.

**The learned modules, inference weights and forecast equations are unchanged.** The added export captures fields already produced by the released model. The original September 29 package remains available separately; use the pressure-export-v2 archive for the complete default output.

Inference weight SHA-256: `db49f36e85a3766defc4c172746897a1f783705d1ce8e6f9dfb8e87ae1d902cb`.

## Matched development results

| Measure | 1.1 | 1.2 mean of 50 | Matched coverage |
| --- | ---: | ---: | --- |
| Mean track error | 798.4 km | 471.2 km | 1,473 daily starts / 270 storms |
| Direction error | 51.58° | 34.96° | Same daily starts |
| Central-pressure MAE against JMA | 13.53 hPa | 12.84 hPa | 134 common starts / 40 storms |

Storms receive equal weight after valid leads and daily starts are averaged. The mean pressure reduction is small and its paired whole-storm uncertainty includes no improvement. These repeatedly inspected results are development evidence, not a certified untouched holdout. [Verified common-support metrics](https://github.com/yu314-coder/typhoon-predict/blob/main/evaluation/released_daily/released_daily_benchmark.json).

The [model announcement](https://github.com/yu314-coder/typhoon-predict) includes the selected Mangkhut pressure-map animation and architecture. Selected examples are not representative skill. Auxiliary wind and pressure-derived radius diagnostics remain unvalidated; there is no native wind-radius forecast head in 1.2.

This is a research model, not an operational warning service or a safety-critical forecast. Trackformer 1.1 remains a separate release. No historical archive, media, benchmark predictions or training run is modified by this exporter update.
