# FTpyR

Python program for analysing Open Path Fourier Transform InfraRed (OP-FTIR) spectra of volcanic gases.

## Overview

This programis designed to fit absorbance FTIR spectra of volcanic plumes to retrieve the column amounts of various volcanic gases of interest along the line of sight. Included is a Graphical User Iterface (GUI) designed to allow easy analysis of FTIR spectra in multiple windows simultaneously, while the FTpyR library is designed to allow for more flexible or batch analysis if required.

## Instalation

Currently there is no executable file for the FTpyR GUI, though this is planned. Follow these steps to install the required Python libraries. There are two options: using the Anaconda scientific Python distribution or using Python virtual environments.

### Clone git repository

You will need to clone this repository to your computer by either using `git clone https://github.com/benjaminesse/FTpyR.git` or by downloading as a .zip file. This contains all the Python scripts for FTpyR, but not the RFM model.

### Anaconda

You will first need to install [Anaconda ](https://www.anaconda.com/download)or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/). Once install, you can create a new conda environment using the following commands:

```
conda create -n ftir
```

This will create an environment called "ftir". Feel free to change this name to whatever you want, but make sure to be consistent. Activate the environment:

```
conda activate ftir
```

The required libaries for basic functionality can be installed using:

```
conda install -c conda-forge numpy scipy pandas xarray tqdm
```

If you want to also use the GUI, then the following are also required:

```
conda install -c conda-forge pyyaml PySide2 pyqtgraph
```

as well as this library, which is not available on conda and so must be installed with pip:

```
pip install pyqtdarktheme
```

This should be everything you need to run FTpyR! To do this, enter the following command in the command line:

```
python FTpyR_interface.py
```

### Python Virtual Environment

You will first require Python, which can be downloaded from https://www.python.org/. Once you have Python installed on your system, follow these steps. Navigate to your local FTpyR directory and create a virtual environment:

```
python python -m venv venv
```

This will create a local directory holding the virtual environment. To activate it run:

```
venv\Scripts\activate
```

on windows or

```
source venv/bin/activate
```

on Unix or MacOS. To test the install type `python` and hit enter. This should open an interactive command prompt for python. To exit this type `exit()` or `ctrl+Z` and hit enter.

Next we need to install the libraries required for FTpyR. For basic functionality you only need the following:

```
pip install numpy scipy pandas xarray tqdm
```

To use the full GUI you will need the following:

```
pip install pyyaml pyqtdarktheme PySide2 pyqtgraph
```

You can then create your own analysis scripts, or to run the GUI program use:

```
python FTpyR_interface.py
```

## Reference Forward Model

FTpyR uses the Reference Forward Model (RFM) developed at AOPP, University of Oxford. The current version of FTpyR uses version RFM v5.12. More details can be found in [this paper](https://doi.org/10.1016/j.jqsrt.2016.06.018) and online at [http://eodg.atm.ox.ac.uk/RFM/](http://eodg.atm.ox.ac.uk/RFM/) (including how to obtain the source code).

## HITRAN

In order to compute otical depths, the RFM requires information on the absorption features of different gases. FTpyR is designed to use the [HITRAN2020 database](https://hitran.org/). The website enables you to download the line-by-line database to a single .par file (defining the wavenumber range and which species to include). This can be read by the RFM directly, or first converted to a .bin format using the [HITBIN](http://eodg.atm.ox.ac.uk/RFM/hitbin.html) program, which will speed up running RFM.

## To Do

The following features still require implementation:

- Read in functions for various spectra types, currently only .spc and spectacle files are supported
- Simple example script for batch analysis
- Example script for parallel processing of batch analysis
- Solar occultaion fitting
