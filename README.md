# Spectrometer Calibration

This repository contains a Python script for calibrating Avantes and NIRQuest spectrometers. The code aligns measured spectra with reference spectra and generates calibration curves to correct intensity variations.

⸻

# Features
	•	Reads Avantes .RAW8 files and NIRQuest .txt spectra.
	•	Interpolates measured and reference spectra to a common wavelength grid.
	•	Smooths NIRQuest spectra to reduce fluctuations (NIRQuest is more prone to noise).
	•	Generates calibration factors and plots for visualization.

⸻

# How to Use
## 1.	Clone the repository
## 2.  Install dependencies (numpy pandas matplotlib scipy openpyxl)
## 3.  Edit the script:

	•	Set the spec_path variable to the folder containing your measured spectra.
	•	Provide a reference spectrum as a 2-column array (wavelength, intensity).

## 4.	Run the script:

For Avantes, set kind="Avantes" and choose num="single" or num="scan" depending on your files.
	•	For NIRQuest, set smooth=True to reduce fluctuations.

## 5.	The script will generate:
	•	Calibration curve (I_calib)
	•	Comparison plots of measured vs reference spectra
	•	Optional .TXT file and plot if you choose to save them

# Notes
	•	NIRQuest spectra often have significant noise, which is why smoothing is applied.
	•	The code works without additional data files—just provide your reference and measured spectra paths.
	•	Make sure your spectra are properly labeled to distinguish different fibers and spectrometers.
