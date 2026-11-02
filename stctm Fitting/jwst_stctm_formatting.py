# First test module for reformatting JWST data into stctm csv st
from scipy.ndimage import median_filter
from astropy.table import Table
from astropy.io import fits
from pathlib import Path
import numpy as np

# region Loading JWST GJ 486 NIRSpec data

j_path = "Z:/jameskirk/MSci Project 26-27/GJ 486/NIRSpec/Calibrated Products/jw01981-o011_t001_nirspec_f290lp-g395h-s1600a1-sub2048_x1dints.fits"

with fits.open(j_path) as hdul:
    # Extension 2 is EXTRACT1D segment
    j_data = hdul[2].data

    # Read all integrations
    j_wavelength_all = np.asarray(j_data["WAVELENGTH"], dtype = float)
    j_flux_all = np.asarray(j_data["FLUX"], dtype = float)
    j_flux_err_all = np.asarray(j_data["FLUX_ERROR"], dtype = float)
    j_dq_all = np.asarray(j_data["DQ"], dtype = np.uint32)

# Mask NaNs, non-positive fluxes, and JWST DO_NOT_USE pixels
j_valid = (np.isfinite(j_flux_all) & np.isfinite(j_wavelength_all) & (j_flux_all > 0) & (j_wavelength_all > 0) & (j_dq_all == 0))

j_flux_all[~j_valid] = np.nan
j_wavelength_all[~j_valid] = np.nan
j_flux_err_all[~j_valid] = np.nan

# Median-combine the remaining integrations into one stellar spectrum
j_wavelength = np.nanmedian(j_wavelength_all, axis = 0)
j_flux = np.nanmedian(j_flux_all, axis = 0)
j_flux_err = np.nanmedian(j_flux_err_all, axis = 0)

# Identify narrow spikes that remain after time-averaging
j_flux_local = median_filter(j_flux, size = 7, mode = "nearest")
j_flux_residual = j_flux - j_flux_local
j_flux_scatter = np.nanmedian(np.abs(j_flux_residual))

valid = (np.isfinite(j_flux) & np.isfinite(j_wavelength) & (j_flux > 0) & (j_wavelength > 0) & (np.abs(j_flux_residual) < 6 * 1.4826 * j_flux_scatter))

j_wave_clean = j_wavelength[valid]
j_flux_clean = j_flux[valid]
j_flux_err = j_flux_err[valid] 

# endregion

# region Preparing JWST data in table format for stctm

# Convert flux to 10^-10 erg s^-1 cm^-2 micron^-1
wave = j_wave_clean
print(wave)
yval = 2.99792458e-19 * j_flux_clean / j_wave_clean ** 2
# Define uncertainties
yerrLow = 2.99792458e-19 * j_flux_err / j_wave_clean**2
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
output_path = Path("Z:/jameskirk/MSci Project 26-27/stctm Fitting/GJ486_NIRSpec_stctm.csv")

stctm_table.write(output_path, format = "ascii.csv", overwrite = True)
print("Saved")
# endregion