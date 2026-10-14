"""Edit the paths/coordinates, run this file, then use the lists in your own code."""

import os
import sys
from pathlib import Path

import numpy as np

# Allow running this file directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from GRIM_Backend.io.loaders import load_dataset


# EDIT THESE VALUES. Angles are degrees; frequency is GHz.
FOLDER = r"C:\data\my_datasets"
AZIMUTH = 90.0
ELEVATION = 0.0
FREQUENCY = 2.0
POLARIZATION = "VV"

paths = [
    os.path.join(FOLDER, name)
    for name in sorted(os.listdir(FOLDER))
    if name.lower().endswith(".grim") and os.path.isfile(os.path.join(FOLDER, name))
]
# For one file (any supported format), REPLACE the paths assignment above with:
# paths = [r"C:\data\my_dataset.grim"]
# For several specific files: paths = [r"C:\data\first.csv", r"C:\data\second.ss"]

# Normalize coordinates so the hard-coded values work with radian/Hz files too.
datasets = [
    load_dataset(path).convert_axis_units(azimuth="deg", elevation="deg", frequency="GHz")
    for path in paths
]

point_values = []          # One linear-power number per file.
azimuth_cuts = []          # One list of power vs. azimuth per file.
frequency_cuts = []        # One list of power vs. frequency per file.
selected_coordinates = []

for path, dataset in zip(paths, datasets):
    # Select the nearest stored coordinate; these lines do not interpolate.
    az = int(np.argmin(np.abs(dataset.azimuths - AZIMUTH)))
    el = int(np.argmin(np.abs(dataset.elevations - ELEVATION)))
    freq = int(np.argmin(np.abs(dataset.frequencies - FREQUENCY)))
    pol = dataset.polarizations.tolist().index(POLARIZATION.upper())

    # Array order: [azimuth, elevation, frequency, polarization].
    # A colon keeps every sample along that axis.
    point_values.append(float(dataset.power[az, el, freq, pol]))
    azimuth_cuts.append(dataset.power[:, el, freq, pol].tolist())
    frequency_cuts.append(dataset.power[az, el, :, pol].tolist())

    coordinates = {
        "azimuth_deg": float(dataset.azimuths[az]),
        "elevation_deg": float(dataset.elevations[el]),
        "frequency_ghz": float(dataset.frequencies[freq]),
        "polarization": str(dataset.polarizations[pol]),
    }
    selected_coordinates.append(coordinates)
    print(os.path.basename(path), coordinates, "linear power:", point_values[-1])

    # Other values you can extract and append to your own lists:
    # phase_deg = np.rad2deg(dataset.phase[:, el, freq, pol]).tolist()
    # iq = dataset.rcs_slice((slice(None), el, freq, pol)).tolist()
    # Unknown phase stays NaN; use dataset.power to retain magnitude-only data.

# Use any plotting library below. For example, uncomment to plot the first cut:
# import matplotlib.pyplot as plt
# first = datasets[0]
# y = first.linear_to_default_db(azimuth_cuts[0], selected_coordinates[0]["frequency_ghz"], eps=0)
# plt.plot(first.azimuths, y)
# plt.xlabel("Azimuth (deg)")
# plt.ylabel(first.default_log_unit())  # dBsm, dBke, or dB, as appropriate.
# plt.show()
# For a frequency plot, use datasets[0].frequencies and frequency_cuts[0].
