# patchOTDA
## What is this?
The aim of this package is to facilitate the integration of patch clamp electrophysiology datasets. Due to the sensitivity of patch clamp electrophysiological recordings to a variety of different extraneous variables, for example, temperature, solution, region, etc.
![](assets/dataset_drift.PNG)  

patchOTDA is a small python package that wraps several optimal transport based domain adaptation packages. The package aims to help intermediate users integrate two datasets by following simple OOP conventions.  
 
End users are encouraged to check out the streamlit app: https://patchotda.streamlit.app/. This app allows you to integrate your dataset with a reference dataset from the Allen Institute.

## Quickstart
### Install

The package is not currently available on pip but can be installed by pulling the git repo
```
pip install git+https://github.com/smestern/pypatchOTDA.git
```
Should install the packages and their dependencies.
To use SKADA, and FUGW transporters, the user will need to install these additional dependencies manually
```
pip install git+https://github.com/scikit-adaptation/skada
pip install unbalancedgw
```

### Basic usage
