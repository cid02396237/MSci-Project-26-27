# Module for first attempt at single component PHOENIX fitting
from scipy.optimize import minimize_scalar
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
import numpy as np

# region χ² minimisation function for scaling factor
def calc_chi2(scale, model_flux, data_flux, data_err):
    scaled_model = scale * model_flux
    chi2 = np.sum(((data_flux - scaled_model) / data_err) ** 2)
    return chi2
# endregion

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
j_flux_clean = j_flux[valid] * 1e3
j_flux_err = j_flux_err[valid] * 1e3

print(np.min(j_wave_clean))
print(np.max(j_wave_clean))
# endregion

# region Loading and trying PHOENIX models 
p_folder = "Z:/jameskirk/MSci Project 26-27/PHOENIX High Resolution Z0 Models/PHOENIX-ACES-AGSS-COND-2011/Z-0.0"
p_path = Path(p_folder)

# Prescribe the temperature range for relevant models (K)
minimum_temperature = 2300
maximum_temperature = 4300

# Load wavelength grid for PHOENIX
wave_path = Path("Z:/jameskirk/MSci Project 26-27/PHOENIX High Resolution Z0 Models/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
with fits.open(wave_path) as hdul_wave:
    p_wavelength = hdul_wave[0].data

# Correct for units Å --> μm
p_wave_scaled = p_wavelength / 1e4 

# Set flag values for smallest χ²
small_chi = np.inf
small_flux = None
small_scale = None
small_model = None

# Iterate through sorted model files in the prescribed temperature range
model_files = sorted(p_path.rglob("*.fits"))

# Iterating through all files in the folder and load flux data
for model in model_files:

    # Exlcude wavelength grid!
    if model.name == wave_path.name:
        continue

    # Temperature from filename
    model_temperature = int(model.name[3:8])
    print(model_temperature)
    
    # Only use temperature models in reasonable range
    if model_temperature < minimum_temperature:
        continue
    if model_temperature > maximum_temperature:
        break

    with fits.open(model) as hdul:
        p_flux = hdul[0].data          

    # Correct for units ergs/s/cm^2/cm^-1 --> mJy
    p_flux_mJy = 0.333564 * p_flux * (p_wavelength ** 2)

    # Scale by R2∗/dist^2
    r_star_m = 0.33 * 6.597e8
    dist_m = 8.07 * 3.0857e16
    p_flux_scaled = p_flux_mJy * ((r_star_m / dist_m) ** 2)

    # Smooth the flux with a SavGol filter
    window_length = 501 
    polyorder = 2
    p_flux_smoothed = savgol_filter(p_flux_scaled, window_length = window_length, polyorder = polyorder)

    # Interpolate onto same grid as JWST data
    p_flux_clean = np.interp(j_wave_clean, p_wave_scaled, p_flux_smoothed)

    # Fitting scaling factor and determining χ² using bounds from Moran et al. (2023)
    lower_bound = 0.9
    upper_bound = 1.1
    result = minimize_scalar(calc_chi2, bounds = (lower_bound, upper_bound), method = 'bounded', args = (p_flux_clean, j_flux_clean, j_flux_err))

    # Finding smallest χ²
    if result.fun < small_chi:
        small_chi = result.fun
        print(small_chi)
        small_scale = result.x
        small_flux = p_flux_clean
        small_model = model

# endregion

# Plot best comparison
plt.figure()
plt.plot(j_wave_clean, j_flux_clean, color = 'blue', label = 'JWST NIRSpec')
plt.plot(j_wave_clean, small_scale * small_flux, color = 'red', label = f"PHOENIX Best Model R = {small_scale}")
plt.xlabel(r"Wavelength ($\mu$m)")
plt.ylabel("Flux (mJy)")
plt.legend()
plt.show()

print(small_model)
