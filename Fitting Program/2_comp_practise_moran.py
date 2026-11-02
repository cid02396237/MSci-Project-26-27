# Module for first attempt at two component PHOENIX fitting
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
import numpy as np

# region Constants
radial_vel = 19.20
c_kms = 299792.458
beta = radial_vel / c_kms
r_star_m = 0.33 * 6.597e8
dist_m = 8.07 * 3.0857e16
# endregion

# region χ² minimisation function for scaling factors and coverage fractions
def objective(x, model_base_flux, model_second_flux, data_flux, data_err):

    # x[0] = overall scale
    # x[1] = coverage fraction of the second component
    overall_scale, f_second = x

    # Combined fluxes
    comb_flux = overall_scale * ((1.0 - f_second) * model_base_flux + f_second * model_second_flux)

    chi2 = np.sum(((data_flux - comb_flux) / data_err) ** 2)
    return chi2
# endregion

# region Flux calibration function for PHOENIX data
def calibrate_flux(data_wave, model_flux, model_wave):

    # Correct for units ergs/s/cm^2/cm^-1 --> mJy
    flux_mJy = 0.333564 * model_flux * (p_wavelength ** 2)

    # Scale by R2∗/dist^2
    flux_scaled = flux_mJy * ((r_star_m / dist_m) ** 2)

    # Smooth the flux with a SavGol filter
    window_length = 501 
    polyorder = 2
    flux_smoothed = savgol_filter(flux_scaled, window_length = window_length, polyorder = polyorder)

    # Interpolate onto same grid as JWST data
    flux_clean = np.interp(data_wave, model_wave, flux_smoothed)

    return flux_clean
#endregion

# region Loading Moran data
m_path = "Z:/jameskirk/MSci Project 26-27/GJ 486/NIRSpec/S. Moran Work/GJ486b_Zenodo_2023/StellarSpectra/GJ486_StellarSpectraAndModels.txt"
m_data = np.loadtxt(m_path, skiprows = 28, max_rows = 3203, usecols=(1, 2, 3))
m_wave_clean, m_flux_clean, m_flux_err = np.array(m_data).T

# For consistency, clip data between JWST limits
mask = (m_wave_clean >= 2.8806259031070622) & (m_wave_clean <= 3.7164257476067553)
m_wave_clean = m_wave_clean[mask]
m_flux_clean = m_flux_clean[mask]
m_flux_err = m_flux_err[mask]

# endregion

# region Loading and trying PHOENIX models 
p_folder = "Z:/jameskirk/MSci Project 26-27/PHOENIX High Resolution Z0 Models/PHOENIX-ACES-AGSS-COND-2011/Z-0.0"
p_path = Path(p_folder)

# Prescribe the temperature range for relevant models (K)
minimum_temperature = 3000
maximum_temperature = 4000

# Prescribe appropriate log(g) range
min_logg = 4.00
max_logg = 5.50

# Load wavelength grid for PHOENIX
wave_path = Path("Z:/jameskirk/MSci Project 26-27/PHOENIX High Resolution Z0 Models/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
with fits.open(wave_path) as hdul_wave:
    p_wavelength = hdul_wave[0].data

# Correct for units Å --> μm
p_wave_scaled = p_wavelength / 1e4 

# Apply doppler correction for radial movements
p_wave_clean = p_wave_scaled * np.sqrt((1 + beta) / (1 - beta))

# Set flag values for smallest χ²
small_chi = np.inf
small_temp = None
small_scale = None
small_logg = None
small_flux = None
small_second_temp = None
small_base_scale = None
small_second_scale = None
small_fraction = None
small_combined_flux = None

# Iterate through sorted model files in the prescribed temperature range
model_files = sorted(p_path.rglob("*.fits"))

# Iterating through all files in the folder and load flux data
for model in model_files:

    # Exlcude wavelength grid!
    if model.name == wave_path.name:
        continue

    # Temperature and log(g) from filename
    model_temperature = int(model.name[3:8])
    model_logg = float(model.name[9:13])
    
    # Only use temperature and log(g) models in reasonable range
    if (model_temperature < minimum_temperature) or (model_logg < min_logg) or (model_logg > max_logg):
        continue
    if model_temperature > maximum_temperature:
        break

    print(f"Loading Base Model T = {model_temperature}K, log(g) = {model_logg}.")

    # Create list of models at same log(g) and T <= Teff - 100K
    second_list = [model2 for model2 in model_files if (model2.name != wave_path.name and float(model2.name[9:13]) == model_logg and int(model2.name[3:8]) <= model_temperature - 100)]

    # Open base photosphere model
    with fits.open(model) as hdul1:
        p_base_flux = hdul1[0].data

    # Calibrate the flux
    p_base_flux_clean = calibrate_flux(m_wave_clean, p_base_flux, p_wave_clean)

    # Open second model
    for model2 in second_list:
        with fits.open(model2) as hdul2:
            p_second_flux = hdul2[0].data

        model2_temperature = int(model2.name[3:8])
        model2_logg = float(model2.name[9:13])

        print(f"Loading Second Model T = {model2_temperature}K.")

        # Calibrate the flux
        p_second_flux_clean = calibrate_flux(m_wave_clean, p_second_flux, p_wave_clean)

        # Fit for the scaling parameter and coverage fraction of spots to minimise χ²
        result = minimize(objective, x0 = [1.0, 0.1], args = (p_base_flux_clean, p_second_flux_clean, m_flux_clean, m_flux_err), bounds=[(0.9, 1.1), (0.0, 0.5)], method="Nelder-Mead")
        overall_scale, f_second = result.x
        combined_flux = overall_scale * ((1.0 - f_second) * p_base_flux_clean + f_second * p_second_flux_clean)

        # Finding smallest χ²
        if result.fun < small_chi:
            small_chi = result.fun
            small_scale = overall_scale
            small_temp = model_temperature
            small_logg = model_logg
            small_second_temp = model2_temperature
            small_fraction = f_second
            small_flux = combined_flux
            small_combined_flux = combined_flux

            print(f"New smallest χ²: {small_chi}.")

# endregion

print(f"Optimised Model T = {small_temp} K")
print(f"Optimised second Model T = {small_second_temp} K")
print(f"Optimised log(g) = {small_logg}")
print(f"Optimised scale = {small_scale}")
print(f"Optimised second-component coverage fraction = {small_fraction}")

# Plot best comparison
plt.figure()
plt.plot(m_wave_clean, m_flux_clean, color = "blue", label = "Moran et al. (2023)")
plt.plot(m_wave_clean, small_combined_flux, color = "red", label = "PHOENIX two-component model")
plt.xlabel(r"Wavelength ($\mu$m)")
plt.ylabel("Flux (mJy)")
plt.legend()
plt.show()