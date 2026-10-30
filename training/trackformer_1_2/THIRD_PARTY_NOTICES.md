# Data sources and attribution

The MIT license covers project code and project model artifacts; it does not replace upstream data terms.

- **NOAA PSL NCEP/NCAR Reanalysis 1**: basin mean sea-level pressure, 500 hPa height and winds at 850/500/200 hPa. Data provided by NOAA PSL, Boulder, Colorado, USA. https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
- **NOAA/NCEI IBTrACS**: derived storm observations, labels and masks; credit the original contributing meteorological agencies and NOAA/NCEI. Knapp et al. (2010), doi:10.1175/2009BAMS2755.1. https://www.ncei.noaa.gov/products/international-best-track-archive . Exact derived archive SHA is in assets.json; its original raw snapshot ID was not recorded.
- **Copernicus Climate Change Service (C3S), ERA5, accessed through Google's ARCO-ERA5 mirror**: native mean-sea-level-pressure patches and static land fraction/elevation. Contains modified Copernicus Climate Change Service information (2026). ERA5 hourly single-level data, doi:10.24381/cds.adbb2d47, https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels . The current dataset lists a CC-BY licence; attribution and upstream terms apply. ECMWF or the European Commission does not endorse this model. ARCO mirror and recipes: https://github.com/google-research/arco-era5 . The mirror's code licence is separate from the underlying meteorological data licence.

The published arrays are transformed subsets: quantized/coarse basin grids, float16 pressure, issue-centred patches, normalized derived track features and masks. These transformations do not create new native observational resolution. No claim of endorsement by NOAA, ECMWF, C3S or Google is made.
