import matplotlib.pyplot as plt
import numpy as np

### S. Moran Data ###
m_path = "Z:/jameskirk/MSci Project 26-27/GJ 486/NIRSpec/S. Moran Work/GJ486b_Zenodo_2023/StellarSpectra/GJ486_StellarSpectraAndModels.txt"
m_data = np.loadtxt(m_path, skiprows=28, max_rows=3203, usecols=(1, 2, 3))
m_wave_clean, m_flux_clean, m_flux_e = np.array(m_data).T

# Plotting
plt.figure()
plt.plot(m_wave_clean, m_flux_clean, color = "r")
plt.xlabel(r"Wavelength ($\mu$m)")
plt.ylabel("Flux (mJy)")
plt.grid(True, alpha = 0.3)
plt.title("GJ 486 Spectrum Eureka! Pipeline")
plt.show()