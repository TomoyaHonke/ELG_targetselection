# ELG Target Selection (Uchuu $\times$ DESI)
This repository contains code to compare ELG target selection between
mock catalogs (Uchuu) and DESI observations.
## Overview
- Compute number density n(z) for mock catalogs (box / lightcone)
- Convert DESI angular density to comoving number density
- Compare mock and DESI results
- Compare results without target selection
## Thesis
- ELG_target_selection_thesis.pdf
## Requirements
- Python 3.10
- numpy
- pandas
- h5py
- matplotlib
- scipy
- mpmath
- colossus
## Data
This repository does NOT include data.
### DESI Data
The following file is required:

- main-800coaddefftime1200-nz-zenodo.ecsv
 (https://arxiv.org/pdf/2208.08513)

- ELG_LOPnotqso_NGC_nz.txt
- ELG_LOPnotqso_SGC_nz.txt
 (https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5)

### Uchuu mock
Stored locally (not included)

## Notes
### Data path
Please replace "****" with your local data directory.
