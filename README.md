# ZTFrc

This Python library was develop to download ZTF PSF photometry measurements of a given target from IRSA-IPAC database. 
After that it will automaticaly, based in user inputs, select the rigth measurements, build the phase curve, and search for periodicity in between the interval given by the user. 

ZTF survey looks for targets up to 21 mag, so results were tested just for such bright targets. 

## Features

- 🔭 ZTF public data download and analysis for periodicity
- 🪐 Trojans and other small bodies rotational period detection
- 📈 Long term photometry analysis

## Installation

```bash
pip install ZTFrc
