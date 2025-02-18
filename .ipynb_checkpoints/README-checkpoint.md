# Contents:
This repository contains the notebooks and data used to produce the figures and values in the article called "Observational constraints applied to the AMOC projections indicates substantial weakening before 2100".

The repository named "notebooks" contains four notebooks:
- main.ipynb: Display the performances of different observational constrain methods. It produces the figures 3, 4, 5, S1 and the Table S2 of the corresponding article.
- Sensitivity_test_scenario.ipynb: Compares the performances for different choices of scenarios.
- Sensitivity_test_X_Y_definition.ipynb: Compares the performances for different choices of X and Y definition.
- TableS1_Fig1_Fig2.ipynb: Represents the CMIP6 data used through the table S1, the figures 1 and 2 of the corresponding article 

The repository named "functions" contains the different functions used in the notebooks.

The repository named "data" contains the different dataset used in the notebooks. These data repository contains different repositories:
- last_obs_AMOC: Contains the observation of AMOC. It is dowloaded directly from https://rapid.ac.uk/
- obs_SST_SSS: Contains the observations of sea surface temperature (SST) and salinity (SSS). It comes from EN4 (file EN.4.2.2.analyses.g10) that has been preprocessed by us to have annual values in a 1° spatial grid. The spatial resampling is made using the nearest neighbor method.
- multiruns: Contains the values of CMIP6 models of AMOC, SST and SSS for different SSP scenarios. These values are already preprocessed by us to have annual values and values averaged in each of the 9 regions defined in the article, otherwise the datasets are too heavy.
- area_r360x180.nc: Contains the value of area for each grid-cell of the 1° spatial grid.


# Required packages:
The needed packages may have conflictual versions. We recommand the use of a virtual environment where the packages versions are fixed. Using these versions in a virtual environment ensure to avoid conflictual versions, this is only a recommandation as it could work using a different environment. To create a virtual environment, you can install anaconda (https://docs.anaconda.com/anaconda/install/), open Anaconda Navigator, create a new environment in the tab "Environments", install on this environment Jupyter Notebook, and open it to access the notebooks of this packages. Then install the following packages under the corresponding versions, in that order:
- numpy 1.26.4
- pandas 2.2.3
- matplotlib 3.9.4
- scikit-learn 1.6.1
- cartopy 0.23.0
- netCDF4 1.7.2

Python 3.9.21 has been used. These packages can be installed in the Anaconda Navigator of the corresponding virtual environement, or directly in a notebook running for example: !pip install numpy==1.26.4

Once these packages has been installed, you are free to run and modify the notebooks as you wish to experiment them.





