# Forecast examples and diagnostics

The [model announcement](../README.md) features the geographically registered moving-core Mangkhut (2018) forecast. Its 50-member arrays and replay/playback receipts are linked there. These are development illustrations, not representative or untouched-test evidence.

## Additional films

These earlier 50-member films retain a fixed issue-relative regional pressure patch. When the centre leaves it, only the saved coarse basin field covers that position. They use 4 hPa isobars and 23-second playback with a held ending; do not confuse them with the current moving-core Mangkhut rendering. Their original model outputs remain preserved.

- [Fung-wong (2025), 7 November 00 UTC](https://huggingface.co/euler314/typhoon-predict/resolve/87a6e366b42bb4cc0edc95d2c50c55fca21a2c93/docs/trackformer_1_2_fung_wong.mp4) · [provenance](../evaluation/release_data/fung_wong_video.json)
- [Soudelor (2015), 5 August 00 UTC](https://huggingface.co/euler314/typhoon-predict/resolve/87a6e366b42bb4cc0edc95d2c50c55fca21a2c93/docs/trackformer_1_2_soudelor.mp4) · [provenance](../evaluation/release_data/soudelor_video.json)
- [Meranti (2016), 10 September 00 UTC](https://huggingface.co/euler314/typhoon-predict/resolve/87a6e366b42bb4cc0edc95d2c50c55fca21a2c93/docs/trackformer_1_2_meranti.mp4) · [provenance](../evaluation/release_data/meranti_video.json)

## Wind and structure

[Four-storm wind comparison](../evaluation/storm_structure/four_storm_wind_candidate.png) · [audit index](../evaluation/storm_structure/index.json) · [experimental structure method](../release_tools/STORM_STRUCTURE_DIAGNOSTICS.md)

USA one-minute and JMA ten-minute winds are separate references. Native 1.1 RMW and quadrant R34/R50/R64 are scored against matching USA definitions. Trackformer 1.2 has an auxiliary maximum-wind scalar but no native radius head. Its pressure-derived radius candidate is not an official-equivalent prediction; masked/persistent components are not learned radius forecasts.

| Storm | Exact-time comparisons | Data and assumptions |
| --- | --- | --- |
| Fung-wong | [Plot](../evaluation/storm_structure/fung_wong_wind_radius.png) | [Record](../evaluation/storm_structure/fung_wong_structure.json) |
| Soudelor | [Plot](../evaluation/storm_structure/soudelor_wind_radius.png) | [Record](../evaluation/storm_structure/soudelor_structure.json) |
| Mangkhut | [Plot](../evaluation/storm_structure/mangkhut_wind_radius.png) | [Record](../evaluation/storm_structure/mangkhut_structure.json) |
| Meranti | [Plot](../evaluation/storm_structure/meranti_wind_radius.png) | [Record](../evaluation/storm_structure/meranti_structure.json) |

## Full evaluation

[Daily protocol](daily_storm_benchmark.md) · [all daily scores](../evaluation/daily_storm_final.json) · [intensity/radius protocol](intensity_benchmark.md) · [native intensity results](../evaluation/intensity/intensity_final.json) · [matched pressure curves](../evaluation/released_daily/pressure_comparison.png)

Route uses 1,473 starts / 270 storms; the matched native pressure comparison uses 134 starts / 40 storms. JMA pressure MAE falls from 13.53 to 12.84 hPa, with paired uncertainty including zero. JMA curve similarity is 0.7074 / 0.7118 (1.1 / 1.2), while USA curve similarity is 0.7237 / 0.6637 on 131 eligible days—a regression. A centred shape score discards level and amplitude and must be read with physical hPa timelines, not used to conceal pressure errors.
