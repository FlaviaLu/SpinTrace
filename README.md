# ZTFrc

This Python library was develop to download ZTF PSF photometry measurements of a given target from IRSA-IPAC database. 
After that it will automaticaly, based in user inputs, select the rigth measurements, build the phase curve, and search for periodicity in between the interval given by the user. 

ZTF survey looks for targets up to ~21 mag, so results were tested just for such bright targets. 

## Features

- 🔭 ZTF public data download and analysis for periodicity
- 🪐 Trojans and other small bodies rotational period detection
- 📈 Long-term photometry analysis

## Installation and usage steps

- Download the zip file into a new directory.
- Unzip it and enter in ZTFrc/input_files/
- Open and edit the input_parameters.csv, it must have at least two lines to work properly. 
- Already in the input_files/ directory, are two examples of images metadata tables as obtained manually from IRSA/IPAC website. 
- To obtain such tables for your target(s) do the following:
    - Open the webpage https://irsa.ipac.caltech.edu/applications/ztf/?__action=layout.showDropDown&view=Search
    - Write the name of your target under 'Object Name or ID' field, wait for websites' suggestions and click on the right one. 
    - Optionally, you can set the dates interval and image size (10 arcsecons is more than enough).
    - Click on search button.
    - It takes some minutes until the search is completed.
    - Once you see the table in the screen click on the save button (upper right corner).
    - Save as csv file in the same imput_files/ directory as mentioned above.
-  Open a terminal in the ZTFrc directory and type: python3 ZTFrc.py
- Press enter and let it work!
