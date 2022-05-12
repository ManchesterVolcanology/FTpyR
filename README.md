# FTpyR
Python library for analysing Open Path Fourier Transform InfraRed (OP-FTIR) spectra of volcanic gases.

## Overview
This library is designed to fit absorbance FTIR spectra of volcanic plumes to retrieve the column amounts of various volcanic gases of interest along the line of sight. Included is a Graphical User Iterface (GUI) designed to allow easy analysis of FTIR spectra in multiple windows simultaneously, while the FTpyR library is designed to allow for more flexible or batch analysis if required.

## Instalation
Currently there is no executable file for the FTpyR GUI, though this is planned. These steps assume using a Python virtual environment, but feel free to use whatever method you are familiar with. Note that at the time of writing the Anaconda version of PySide6 and pyqtgraph are not up to date with the requirements for the GUI, so using pip is prefered. 

### Python
To install FTpyR, you will first require Python, which can be downloaded from https://www.python.org/. Once you have this, clone this repoistory to your computer either using `git clone https://github.com/benjaminesse/FTpyR.git` or by downloading as a .zip file.

### Virtual Environment
Next, navigate to your local FTpyR directory and create a virtual environment:
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

on Unix or MacOS. To test the install type `python` and hit enter. This should open an interactive command prompt for python. To exit this type `exit()` and hit enter.

### Libraries
Next we need to install the libraries required for FTpyR.

For basic functionality you only need the following:
```
pip install numpy scipy pandas xarray 
```

To use the full GUI you will need the following:
```
pip install pyyaml pyqtdarktheme PySide6 pyqtgraph
```

You can then create your own analysis scripts, or to run the GUI program use:
```
python FTpyR_interface.py
```

## To Do
The following features still require implementation:
- Read in functions for various spectra types, currently only .spc and spectacle files are supported
- Simple example script for batch analysis
- Example script for parallel processing of batch analysis
- Solar occultaion fitting
- Release executable files
