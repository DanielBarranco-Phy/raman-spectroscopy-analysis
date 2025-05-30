# raman-spectroscopy-analysis

This project processes Raman spectroscopy data from Renishaw structures (Raman Microscope files). It extracts peak information and generates heatmaps representing the spatial distribution of these peaks across the sample.

## Features
- Reads and parses Raman02 data files
- Detects and extracts peak positions and intensities
- Generates heatmaps to visualize peak distribution
- Provides summary statistics of identified peaks
- Interactive GUI for file selection and input parameters

## Tech Stack
- Python
- Tkinter (GUI dialogs)
- NumPy
- Pandas
- Matplotlib / Seaborn (for heatmaps and plots)
- SciPy (peak detection and curve fitting)

## How to run

Run the main script:

```bash
python analyze_peaks.py
