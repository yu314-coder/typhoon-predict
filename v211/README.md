# Trackformer 1.2.11

Trackformer 1.2.11 is an intensity-only upgrade over the corrected Trackformer
1.2.10 causal environment model. The SST and 200-850 hPa shear forecast is
frozen; the new head predicts wind and central pressure residuals with
per-lead strengthening, mature, and weakening experts.

## Data policy

- Inference uses only the issue-time and past track/weather analysis features.
- Future analysis rows are training labels only.
- JMA, JTWC, GFS, ECMWF, and other official forecast products are not model
  inputs.
- TIP data is not used for training.

## Quality gate

The selected head was chosen on chronological validation and evaluated on an
untouched post-2019 test split:

| Split | Frozen 1.2.10 wind MAE | 1.2.11 wind MAE | Frozen 1.2.10 pressure MAE | 1.2.11 pressure MAE |
|---|---:|---:|---:|---:|
| Validation | 17.6747 kt | 17.5467 kt | 13.4229 hPa | 13.2331 hPa |
| Test | 17.0321 kt | 16.8931 kt | 13.0413 hPa | 12.6824 hPa |

These are research metrics, not operational warning guidance.

## Files

- `intensity_only/trackformer_1_2_11_manifest.json`: model contract,
  selection, metrics, and data policy.
- `intensity_only/trackformer_1_2_11_intensity_weights.npz`: serialized
  intensity head.
- `../v210/corrected_environment_expert/trackformer_1_2_10_corrected_environment_weights.npz`:
  frozen environment/intensity base required by the head.
- `../v211_intensity_only_train.py`: training and quality-gate implementation.
- `../scripts/test_trackformer_1211_intensity_dolphin.py`: case evaluator and
  graph/map renderer.

The 1.2.11 head does not change route geometry or radius outputs; those remain
the frozen route model outputs until a separate structure quality gate passes.
