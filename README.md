# cold_pool_morphology

## Overview
Code to reproduce the results and plots of the study on observed morphology and growth of convective cold pools by B. Kirsch, C. Hohenegger, and F. Ament. Data sources are the observations of a dense surface-based network of custom-designed APOLLO and WXT weather stations as well as an X-band rain radar during the FESSTVaL 2021 field experiment (Hohenegger et al., 2023) and the precursor FESST@HH 2020 (Kirsch et al., 2022). The respective data sets are openly available for download.

## Description of files
### Code
- **fesstval_interpolation.py**: Reads APOLLO and WXT level 2 data, calculates temperature perturbations for cold pool events, performs spatial interpolation of station network data onto regular cartesian grid via kriging, and writes level 3 data (daily netCDF files of interpolated temperature and pressure perturbation)
- **fesstval_cp_morph.py** (main analysis script): Reads level 3 data of station network and X-band rain radar, performs cluster analysis, and generates all plots (except for station map and event overview of distributions of temperature perturbations)
- **cp_spatial_analysis.py**: Routines needed for spatial interpolation and cluster analysis used by fesstval_interpolation.py and fesstval_cp_morph.py
- **fesstval_plots.py**: Plot routines used by fesstval_cp_morph.py
- **fesstval_map.py**: Plots map of FESSTVaL measurement locations
- **fesstval_cp_overview.py**: Reads APOLLO and WXT level 2 data, calculates temperature perturbations for cold pool events, and plots event overview of distributions of temperature perturbations
- **radar_level2_to_level3.py**: Reads level 2 data of X-band radar measurements, performs nearest-neighbor interpolation from native polar grid onto cartesian grid, and writes level 3 data (corresponding to format of interpolated station data) 
- **fesstval_routines.py**: Routines specific to FESSTVaL data (e. g., read functions)
- **standard_routines.py**: Generic routines used by most of the other scripts (not all routines are actually used)

### Meta data
- **stations_fessthh.txt**: Meta data of APOLLO and WXT stations for FESST@HH 2020
- **stations_fesstval.txt**: Meta data of APOLLO and WXT stations for FESSTVaL 2020
- **cold_pools_fessthh.txt**: List of cold pool events during FESST@HH 2020
- **cold_pools_fesstval.txt**: List of cold pool events during FESSTVaL 2020

### Example data
- **fval_uhh_apollowxt_l3_ta_v00_20210629.nc**: Level 3 data of interpolated temperature perturbations of APOLLO and WXT stations
- **fval_uhh_apollowxt_l3_pa_v00_20210629.nc**: Level 3 data of interpolated pressure perturbations of APOLLO and WXT stations
- **fval_uhh_wrx_l3_rr_v00_20210629.nc**: Level 3 data of interpolated rainfall rates of X-band rain radar

## Requirements
- cartopy 0.17.0
- matplotlib 3.3.4
- netcdf4 1.5.3
- numpy 1.20.1
- pandas 1.2.4
- pykrige 1.6.1
- scipy 1.6.2
- xarray 0.18.0

## References
- Hohenegger, C. and the FESSTVaL team (2023): *FESSTVaL: the Field Experiment on Submesoscale Spatio-Temporal Variability in Lindenberg*, Bull. Am. Meteorol. Soc. (in review)
- Kirsch, B., Hohenegger, C., Klocke, D., Senke, R., Offermann, M., and Ament, F. (2022): *Sub-mesoscale observations of convective cold pools with a dense station network in Hamburg, Germany*, Earth Syst. Sci. Data, 14, 3531–3548, https://doi.org/10.5194/essd-14-3531-2022. 

## Data sets and code
- APOLLO and WXT level 2 data of FESST@HH 2020: https://doi.org/10.25592/UHHFDM.10172
- APOLLO and WXT level 2 data of FESSTVaL 2021: https://doi.org/10.25592/UHHFDM.10179
- X-band radar level 2 data of FESST@HH 2020: https://doi.org/10.26050/WDCC/LAWR_UHH_HHG
- X-band radar level 2 data of FESSTVaL 2021: https://doi.org/10.25592/UHHFDM.10090
- Description of FESSTVaL data format: https://doi.org/10.25592/UHHFDM.10416
- cold_pool_detection code: https://doi.org/10.5281/ZENODO.4321260

## Data policy
The data policy for FESSTVaL campaign data applies (https://doi.org/10.25592/UHHFDM.10181).

## Contact
Bastian Kirsch (bastian.kirsch@uni-hamburg.de)<br>
Felix Ament (felix.ament@uni-hamburg.de)<br>
Meteorologisches Institut, Universität Hamburg, Germany

Last updated: 15 May 2023
