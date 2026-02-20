# Batan Bay Hydroclimate and Water Quality Analyses

This repository contains the analysis scripts for a manuscript currently under review:

Hydroclimate redistribution, nutrient hotspots, and tide--depth oxygen structure in a tropical estuary with recurrent HAB impacts (Batan Bay, Philippines)

Eladawy et al.

## Contents
- `chirps_seasonal_shift_points.py` - CHIRPS rainfall seasonal redistribution analysis and figure.
- `nutrients_maps_stoichiometry.py` - Nutrient maps and DIN:PO4 stoichiometry.
- `do_flood_ebb_maps_diagnostics.py` - Flood/ebb DO maps and spatial diagnostics.
- `aaq_nutrients_multivariate_analysis.py` - Multivariate AAQ and nutrient analysis.
- `mur_ghrsst_mhw_inlet.py` - GHRSST MUR inlet extraction and MHW diagnostics.

## Data layout
Place data under `data/` as follows:

```
data/
  AAQ/EXTRACTED2024/
    extracted_temp_data.csv
    extracted_sal_data.csv
    extracted_chla_data.csv
    extracted_turb_data.csv
    final2024DOprofiles.csv
    (other AAQ CSVs as needed)
  basemap/
    L15-1720E-1090N.tif
  chirps_geotiff/
    *.tif
  ghrsst/
    *JPL-L4_GHRSST-SSTfnd-MUR-GLOB*.nc4
outputs/
  (auto-generated)
```

Notes:
- The repository does not include large external datasets (CHIRPS GeoTIFFs, GHRSST MUR NetCDFs, basemap TIFF). Please obtain them separately.
- Some data may be subject to permissions. Ensure you have appropriate rights to use and share them.

## Setup
Python 3.9+ is recommended. Install dependencies:

```
pip install -r requirements.txt
```

Geospatial packages (e.g., `cartopy`, `geopandas`, `rasterio`) may require system libraries.
If you run into installation issues, consider using conda/miniforge.

## Usage
Run the scripts directly, for example:

```
python chirps_seasonal_shift_points.py
python nutrients_maps_stoichiometry.py
python do_flood_ebb_maps_diagnostics.py
python aaq_nutrients_multivariate_analysis.py
python mur_ghrsst_mhw_inlet.py
```

Outputs are written to `outputs/` by default.

## Citation
If you use this code, please cite the manuscript noted above. A `CITATION.cff` file is included.

## License
MIT License. See `LICENSE`.
