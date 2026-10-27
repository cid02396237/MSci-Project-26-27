# Module to compare JWST level 3 products to Eureka! calibrated S. Moran Work
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import median_filter
import numpy as np

# JWST level 3 file
j_path = "Z:/jameskirk/MSci Project 26-27/GJ 486/NIRSpec/Calibrated Products/jw01981-o011_t001_nirspec_f290lp-g395h-s1600a1-sub2048_x1dints.fits"

### JWST Data ###
with fits.open(j_path) as hdul:
    # Extension 2 is EXTRACT1D segment
    j_data = hdul[2].data

    # Read all integrations
    j_wavelength_all = np.asarray(j_data["WAVELENGTH"], dtype = float)
    j_flux_all = np.asarray(j_data["FLUX"], dtype = float)
    j_dq_all = np.asarray(j_data["DQ"], dtype = np.uint32)

# Mask NaNs, non-positive fluxes, and JWST DO_NOT_USE pixels
j_valid = (
    np.isfinite(j_flux_all)
    & np.isfinite(j_wavelength_all)
    & (j_flux_all > 0)
    & (j_wavelength_all > 0)
    & (j_dq_all == 0)
)

j_flux_all[~j_valid] = np.nan
j_wavelength_all[~j_valid] = np.nan

# Median-combine the remaining integrations into one stellar spectrum
j_wavelength = np.nanmedian(j_wavelength_all, axis = 0)
j_flux = np.nanmedian(j_flux_all, axis = 0)

# Identify narrow spikes that remain after time-averaging
j_flux_local = median_filter(j_flux, size = 7, mode = "nearest")
j_flux_residual = j_flux - j_flux_local
j_flux_scatter = np.nanmedian(np.abs(j_flux_residual))

valid = (
    np.isfinite(j_flux)
    & np.isfinite(j_wavelength)
    & (j_flux > 0)
    & (j_wavelength > 0)
    & (np.abs(j_flux_residual) < 6 * 1.4826 * j_flux_scatter)
)

j_wave_clean = j_wavelength[valid]
j_flux_clean = j_flux[valid]
j_flux_clean = j_flux_clean * 1e3  # Jy -> mJy

# Plotting
plt.figure()
plt.plot(j_wave_clean, j_flux_clean, color = "b")
plt.xlabel(r"Wavelength ($\mu$m)")
plt.ylabel("Flux (mJy)")
plt.grid(True, alpha = 0.3)
plt.title("GJ 486 Spectrum NIRSpec")
plt.show()