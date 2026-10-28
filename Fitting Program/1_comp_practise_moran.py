# Module for first attempt at single component PHOENIX fitting
from scipy.optimize import minimize_scalar
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

# region Loading Moran data
m_path = "Z:/jameskirk/MSci Project 26-27/GJ 486/NIRSpec/S. Moran Work/GJ486b_Zenodo_2023/StellarSpectra/GJ486_StellarSpectraAndModels.txt"
m_data = np.loadtxt(m_path, skiprows = 28, max_rows = 3203, usecols=(1, 2, 3))
m_wave_clean, m_flux_clean, m_flux_e = np.array(m_data).T

# For consistency, clip data between JWST limits
mask = (m_wave_clean >= 2.8806259031070622) & (m_wave_clean <= 3.7164257476067553)
m_wave_clean = m_wave_clean[mask]
m_flux_clean = m_flux_clean[mask]
m_flux_e = m_flux_e[mask]

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

    # Exlcude wavelength grid
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
    p_flux_clean = np.interp(m_wave_clean, p_wave_scaled, p_flux_smoothed)

    # Fitting scaling factor and determining χ² using bounds from Moran et al. (2023)
    lower_bound = 0.9
    upper_bound = 1.1
    result = minimize_scalar(calc_chi2, bounds = (lower_bound, upper_bound), method = 'bounded', args = (p_flux_clean, m_flux_clean, m_flux_e))

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
plt.plot(m_wave_clean, m_flux_clean, color = 'blue', label = 'Eureka! Pipeline')
plt.plot(m_wave_clean, small_scale * small_flux, color = 'red', label = f"PHOENIX Best Model R = {small_scale}")
plt.xlabel(r"Wavelength ($\mu$m)")
plt.ylabel("Flux (mJy)")
plt.legend()
plt.show()

print(small_model)