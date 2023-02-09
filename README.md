# cold_pool_morphology

## Overview
Code to reproduce the results and plots of the study on observed morphology and growth of convective cold pools by B. Kirsch, C. Hohenegger, and F. Ament. Data sources are the observations of a dense surface-based network of custom-designed APOLLO and WXT weather stations as well as an X-band rain radar during the FESSTVaL 2021 field experiment (Hohenegger et al., 2023) and the precursor FESST@HH 2020 (Kirsch et al., 2022). The respective data sets are openly available for download.

## Description of files
- fesstval_interpolation.py: Reads APOLLO and WXT level 2 data, calculates temperature perturbations for cold pool events, performs spatial interpolation of station network data (temperature and pressure) onto regular grid via kriging, and writes level 3 data (netCDF format)
- fesstval_cp_morph.py (main analysis script): Reads level 3 data of station network and X-band rain radar, performs cluster analysis, and generates all plots (except for station map and event overview of distributions of temperature perturbations)
- cp_spatial_analysis.py: Contains routines needed for spatial interpolation and cluster analysis used by fesstval_interpolation.py and fesstval_cp_morph.py
- fesstval_plots.py: 
- fesstval_map.py:
- fesstval_cp_overview.py:
- radar_level2_to_level3.py:
- fesstval_routines.py:
- standard_routines.py:
- stations_fessthh.txt:
- stations_fesstval.txt:
- cold_pools_fessthh.txt:
- cold_pools_fesstval.txt: 

## Requirements
(List of python package versions)

## References
- Hohenegger, C. and the FESSTVaL team (2023): *FESSTVaL: the Field Experiment on Submesoscale Spatio-Temporal Variability in Lindenberg*, submitted to Bull. Am. Meteorol. Soc.
- Kirsch, B., Hohenegger, C., Klocke, D., Senke, R., Offermann, M., and Ament, F. (2022): *Sub-mesoscale observations of convective cold pools with a dense station network in Hamburg, Germany*, Earth Syst. Sci. Data, 14, 3531–3548, https://doi.org/10.5194/essd-14-3531-2022. 

## Data sets and code
- APOLLO and WXT level 2 data of FESST@HH 2020: https://doi.org/10.25592/UHHFDM.10172
- APOLLO and WXT level 2 data of FESSTVaL 2021: https://doi.org/10.25592/UHHFDM.10179
- X-band radar data of FESST@HH 2020: https://doi.org/10.26050/WDCC/LAWR_UHH_HHG
- X-band radar data of FESSTVaL 2021: https://doi.org/10.25592/UHHFDM.10090
- cold_pool_detection code: https://doi.org/10.5281/ZENODO.4321260

## Contact
Bastian Kirsch (bastian.kirsch@uni-hamburg.de)<br>
Felix Ament (felix.ament@uni-hamburg.de)<br>
Meteorologisches Institut, Universität Hamburg, Germany

9 February 2023
