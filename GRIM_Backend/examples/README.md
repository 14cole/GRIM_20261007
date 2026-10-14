# GRIM Python examples

These are edit-and-run examples for common headless dataset workflows. There
are no command-line options. Open the desired script, change the clearly marked
`EDIT THESE SETTINGS` block near the top, save it, and run the file:

```powershell
python GRIM_Backend/examples/simple_data_access.py
python GRIM_Backend/examples/join_folder.py
python GRIM_Backend/examples/plot_folder_azimuth_sweeps.py
python GRIM_Backend/examples/plot_folder_frequency_sweeps.py
python GRIM_Backend/examples/query_dataset.py
```

Start with `simple_data_access.py` for a short, straight-line script you can
adapt freely. The other examples also provide importable functions for larger
Python workflows.

## Simple lists for your own plots

Edit [simple_data_access.py](simple_data_access.py). Set `FOLDER`, `AZIMUTH`,
`ELEVATION`, `FREQUENCY`, and `POLARIZATION`, then run it. The script uses an
`os.listdir()` list comprehension to load `.grim` files. To load one file or
several named files in any supported format, replace `paths` with a list of
their filenames as shown in the script.

Coordinates are converted to degrees and GHz. The script selects the nearest
stored coordinate and prints the actual selection; it does not interpolate.
Each dataset is selected independently, so check `selected_coordinates` when
comparing files with different sampling grids.

The lists remain available to the rest of your script:

```python
point_values[0]             # One linear-power value from the first file.
azimuth_cuts[0]             # Power vs. azimuth from the first file.
datasets[0].azimuths        # Matching azimuth coordinates in degrees.
frequency_cuts[0]           # Power vs. frequency from the first file.
datasets[0].frequencies     # Matching frequency coordinates in GHz.
```

The same index selects the same file in `paths`, `datasets`, and each result
list. Cuts may have different lengths across files. Linear power retains the
dataset's physical quantity (RCS, scattering width, or ratio). Optional lines
show phase, I/Q, and a plain Matplotlib plot with the correct dB label.
Magnitude-only data works; unavailable phase remains `NaN`.

The other folder examples recognize the formats supported by `GRIM_Backend.scripting.api`:
`.grim`, `.csv`, `.cst_data`, `.txt`, `.out`, `.pio`, `.cmplx_di`, `.ptm`, and
`.ss`. Set `FILE_PATTERN` or `INPUT_PATTERN` to `"*.grim"` (or another glob)
when only one file type should participate.

## Join every dataset in a folder

Edit [join_folder.py](join_folder.py):

```python
INPUT_FOLDER = Path(r"C:\data\cuts")
OUTPUT_FILE = Path(r"C:\data\joined.grim")
FILE_PATTERN = "*.grim"
SEARCH_SUBFOLDERS = False
PARALLEL_LOADERS = 1
OVERLAP_POLICY = "error"
COORDINATE_TOLERANCE = 1.0e-6
MAX_OUTPUT_GIB = None
OVERWRITE_OUTPUT = False
```

The script sorts paths deterministically, excludes its own prior output, and
uses strict overlap checking by default. Complementary data and equivalent
duplicate samples join normally; conflicting finite samples stop the operation.
Only set `OVERLAP_POLICY` to `"first"` or `"last"` when sorted-path priority is
an intentional data decision. Existing output is preserved unless
`OVERWRITE_OUTPUT` is `True`.

Join combines complementary coordinates into one grid. It is different from
Stitch, which deliberately resolves conflicting overlaps using a selected
policy.

## Cartesian azimuth sweeps from a folder

Edit [plot_folder_azimuth_sweeps.py](plot_folder_azimuth_sweeps.py):

```python
INPUT_FOLDER = Path(r"C:\data\trade_study")
INPUT_PATTERN = "*.grim"
OUTPUT_FOLDER = Path(r"C:\data\azimuth_plots")

# None means every common frequency. Values use the stored frequency unit.
FREQUENCIES = (8.0, 10.0)
ELEVATION = 0.0
POLARIZATIONS = ("VV",)

QUANTITY = "magnitude"
ANGLE_DISPLAY_UNIT = "deg"
FREQUENCY_DISPLAY_UNIT = "GHz"
```

The output is Cartesian (`azimuth_rect`), not polar. Multiple frequencies and
polarizations create separate PNGs, with all compatible folder datasets
overlaid in each plot. Set `QUANTITY = "phase"` for phase. `FREQUENCIES` and
`ELEVATION` use the first dataset's stored/native units; display-unit settings
only change axes and labels. Fixed-axis matching is exact within
`AXIS_MATCH_TOLERANCE`; the script never interpolates.

Leave `OUTPUT_FOLDER = None` to create `grim_azimuth_plots` under the input
folder. Existing PNGs receive a numeric suffix unless `OVERWRITE_EXISTING` is
`True`.

## Frequency sweeps with an optional azimuth percentile band

For an exact azimuth cut, edit
[plot_folder_frequency_sweeps.py](plot_folder_frequency_sweeps.py) like this:

```python
AZIMUTH = 0.0
AZIMUTH_BAND = None
AZIMUTH_PERCENTILE = None
ELEVATION = 0.0
POLARIZATIONS = ("VV",)
```

For P90 across an inclusive azimuth band at every frequency:

```python
AZIMUTH = None
AZIMUTH_BAND = (-10.0, 10.0)
AZIMUTH_PERCENTILE = 90.0
ELEVATION = 0.0
POLARIZATIONS = ("VV",)
```

When a band is active, `AZIMUTH_PERCENTILE = None` uses P50. A descending band,
such as `(170.0, -170.0)` on a signed degree axis, crosses the periodic seam.
The percentile is calculated independently at each frequency in the displayed
logarithmic RCS domain, using only common stored azimuth samples. Periodic
endpoint aliases are counted once. Ordinary percentiles are not meaningful for
wrapped phase, so phase frequency sweeps require an exact `AZIMUTH`.

Coordinate selections use the first dataset's stored units. There is no hidden
interpolation or extrapolation.

## Query one sample by coordinate value

Edit [query_dataset.py](query_dataset.py):

```python
DATASET_PATH = Path(r"C:\data\result.grim")

QUERY_AZIMUTH = 45.0
QUERY_ELEVATION = 0.0
QUERY_FREQUENCY = 10.0
QUERY_POLARIZATION = "VV"
QUERY_ANGLE_UNIT = "deg"
QUERY_FREQUENCY_UNIT = "GHz"

NEAREST_MATCH = False
ANGLE_TOLERANCE = None
FREQUENCY_TOLERANCE = None
JSON_OUTPUT_PATH = None
OVERWRITE_JSON = False
```

Query units may differ from the stored axes; the script converts them before
searching. The printed JSON shows the matched coordinates and all four indices.
GRIM's array order is always:

```text
[azimuth, elevation, frequency, polarization]
```

The result also includes stored linear power, field magnitude, the dataset's
normal dB display value/unit, phase with its declared wrapping interval, and
complex real and imaginary parts when phase is known. Exact-with-tolerance
matching is the default. Set `NEAREST_MATCH = True` only when deliberately
snapping to the closest stored coordinate.

Set `JSON_OUTPUT_PATH` to a `Path` to save the printed result. An existing JSON
file is preserved unless `OVERWRITE_JSON` is `True`.

The essential direct-access pattern is:

```python
indices = (azimuth_index, elevation_index, frequency_index, polarization_index)
power = dataset.power[indices]
phase_radians = dataset.phase[indices]
display_db = dataset.linear_to_default_db(
    power,
    frequency_value=dataset.frequencies[frequency_index],
)
```

Reading power and phase separately is important: a power-only dataset may have
valid power with unknown (`NaN`) phase, so reconstructing a complex value alone
would hide that usable magnitude sample.
