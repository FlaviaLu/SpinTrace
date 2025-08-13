# SpinTrace

This Python library was develop to download ZTF PSF photometry measurements of a given target from IRSA-IPAC database. 
After that it will automaticaly, based in user inputs, select the right measurements, build the phase curves, and search for periodicity within the interval given by the user. 

ZTF survey looks for targets up to ~21 mag, so results were tested just for such bright targets. 

## Features

- 🔭 ZTF public data download and analysis for periodicity
- 🪐 Small bodies rotational period detection
- 📈 Long-term photometry analysis

## Installation and usage steps

- Create a conda environment by conda create -name spintrace
- Activate the environment and install all required libraries using pip.
- Install jupyterlab using conda conda install -c conda-forge jupyterlab
- Download the zip file into a new directory.
- In a Jupyter notebook, import the SpinTrace.py library;
- To start the analysis, provide the csv table with ZTF image's metadata as obtained from IRSA/IPAC webpage. 
- To obtain such tables for your target(s) do the following:
    - Open the webpage https://irsa.ipac.caltech.edu/applications/ztf/?__action=layout.showDropDown&view=Search
    - Write the name of your target under 'Object Name or ID' field, wait for websites' suggestions and click on the right one. 
    - Optionally, you can set the time frame for the search.
    - Click on the search button.
    - It takes some minutes until the search is completed.
    - Once you see the table in the screen click on the save button and select CSV format. 
- Example Jupyter notebooks are provided in the example directory for Haumea and Hektor. 
