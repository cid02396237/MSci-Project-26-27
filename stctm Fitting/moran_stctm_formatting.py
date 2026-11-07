# First test module for reformatting Moran data into stctm csv
from scipy.ndimage import median_filter
from astropy.table import Table
from pathlib import Path
import numpy as np

### S. Moran Data ###
m_path = "Z:/jameskirk/MSci Project 26-27/GJ 486/NIRSpec/S. Moran Work/GJ486b_Zenodo_2023/StellarSpectra/GJ486_StellarSpectraAndModels.txt"
m_data = np.loadtxt(m_path, skiprows = 28, max_rows = 3203, usecols=(1, 2, 3))
m_wave_clean, m_flux_clean, m_flux_err = np.array(m_data).T

# Convert to Jy baseline and clip between JWST limits
mask = (m_wave_clean >= 2.8806259031070622) & (m_wave_clean <= 3.7164257476067553)
m_wave_clean = m_wave_clean[mask]
m_flux_clean = m_flux_clean[mask] * 1e3
m_flux_err = m_flux_err[mask] * 1e3

### Formatting for stctm ###

# Convert flux to 10^-10 erg s^-1 cm^-2 micron^-1
wave = m_wave_clean
conversion = 29.9792458 / wave**2
yval = m_flux_clean * conversion
yerrLow = m_flux_err * conversion
yerrUpp = yerrLow.copy()

# Estimate the wavelength bin edges
edges = np.empty(len(wave) + 1)
edges[1:-1] = 0.5 * (wave[:-1] + wave[1:])
edges[0] = wave[0] - 0.5 * (wave[1] - wave[0])
edges[-1] = wave[-1] + 0.5 * (wave[-1] - wave[-2])
waveMin = edges[:-1]
waveMax = edges[1:]

# Preparing table 
stctm_table = Table({"wave": wave, "waveMin": waveMin, "waveMax": waveMax, "yval": yval, "yerrLow": yerrLow, "yerrUpp": yerrUpp})
output_path = Path("Z:/jameskirk/MSci Project 26-27/stctm Fitting/Moran_stctm.csv")

stctm_table.write(output_path, format = "ascii.csv", overwrite = True)
print("Saved")