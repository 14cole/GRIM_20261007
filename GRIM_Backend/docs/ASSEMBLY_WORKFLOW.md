# Vehicle assembly

The Assembly tab builds a coherent vehicle approximation from a body response,
localized point responses, and line responses expanded along paths. Its primary
workflow is **Body → add features → Calculate & save**. The four pages organize
the inputs; visiting every page is not a prerequisite for a calculation.

## Main layout

- **Vehicle summary** stays visible: selected body, included point count, and
  included path count. A next-action message identifies the blocking input.
- **Coordinates** declares the shared placement units. There is no guessed
  default. The first Add dialog can establish these units for the vehicle.
- **Body / Points / Line features / Build** keep common tasks directly accessible.
- **3-D placement** stays beside the controls. Geometry preview updates after
  placement edits; response comparison becomes available after a build.
- **Save result** and **Calculate & save** stay at the bottom of every page.
  Selecting a body suggests an output filename that the user can change.
- **Recipe** opens naming, variants, save, and load. **More tools**
  contains optional manual placement editors and wing/fin section expansion.
- **Clear all** starts an empty assembly, resetting inputs, mappings, settings,
  preview, and analysis results. Source files and saved outputs are kept. Finish
  an active calculation or close the placement editor before clearing.

Both feature pages start with CSV import: **Import point CSV** or **Import line
CSV**. Coordinates and orientations come from the file; GRIM discovers the
feature types and presents their response mappings automatically. No coordinate
or vertex table is required. Optional manual editors are under **More tools →
Manual placement (optional)**.

## Body

Choose a clean-body `.grim` file or an already loaded, saved response.

| Input | What happens |
| --- | --- |
| BoR response with an embedded profile | Uses the embedded geometry for preview and placement checks. No separate mesh selection is required. |
| Normal coherent 3-D GRIM response | Uses its existing frequency and angle grid. The geometry section opens to request the matching STL/facet mesh for feature placement checks or shadowing. |
| Body without enabled features | Can calculate a body-only baseline. |

Mesh units are independent of placement units. A surface mesh supplies geometry;
it is not a replacement for the body's electromagnetic response. Existing
metadata, registration, and optional strict-library validation remain active.

The Body page contains only the body response and its geometry. Import point
placements on **Points** and line placements on **Line features**.

## Point features

**Import point CSV** is the primary action. Choose the vehicle's placement CSV,
declare its coordinate units, and assign a response file to each `dataset_id`.
One CSV can contain many feature types and placements; repeated fasteners share
the same `dataset_id` and response. Import reads positions and orientations
directly, so no manual coordinate entry is required. The CSV details section
opens after selection and includes a guide and a blank template. Choosing a
different point CSV replaces the current point list; line placements are kept.

The existing point CSV header is:

```csv
placement_id,dataset_id,x,y,z,nx,ny,nz,roll_x,roll_y,roll_z
```

**More tools → Manual placement → Add point feature manually** opens an optional
dialog containing:

1. A naming shortcut: Fastener, Inlet, Antenna, or Other point feature.
2. A descriptive group name and a response file, optionally reused from an
   existing feature group.
3. Position X/Y/Z, outward normal, and roll reference.
4. A count and X/Y/Z step for a straight row of identical features.

Selecting a type does not invent a response or change the physics. The response
file provides the coherent installed-feature-minus-clean-skin delta with the
required polarization channels. Position is the phase location. The normal
defines local +Z, and the projected roll reference defines local +X.

**Add to vehicle** validates the geometry syntax, assigns unique stable IDs,
preserves existing features, and maps the selected response automatically.
Repeating a name creates a separate group without overwriting the earlier one.

For example, one fastener response can be placed 20 times along a straight row
by entering its first position, count 20, and the spacing vector. Rings,
distribution along a path, and surface projection remain available in
the optional **Edit selected point placements** tool.

## Line features

**Import line CSV** reads the paths, ordered segment endpoints, and outward
normals from a file. Choose its coordinate units and map each `dataset_id` to a
line response. No vertices need to be typed or recreated in the GUI. Choosing
another line CSV replaces the line list while preserving point placements.

The line CSV guide and template are under **CSV details / response mappings**.
Rows for each `line_id` stay together, with `segment_index` beginning at 1 and
consecutive segments meeting head-to-tail. A closed perimeter is represented by
its final segment ending at the first segment's start.

For occasional manual work, **More tools → Manual placement → Add line path
manually** pairs a response with vertices entered in a table. Its **Close loop**
option adds the final segment back to the first vertex.

The line response supplies coherent feature-minus-skin TE/TM coefficients. Its
expansion follows the entered path; choosing a full-body or point response does
not make it a valid line response. Backend validation still checks compatibility.

Vertex positions use the vehicle's shared placement units. Normal vectors are
unitless. Duplicate consecutive vertices, invalid directions, and degenerate
frames are rejected before the feature is added. Curved surfaces need enough
vertices and appropriate normals to represent the skin within the existing
placement tolerances.

## Managing the vehicle

Each feature page lists **Use / Feature / Placed / Response**. This is a summary
of imported groups and their responses, not a coordinate-entry table. The count
shows included versus total placements when a group is partially excluded.

- **Use** includes or excludes the complete group from the calculation.
- Optional **More tools → Manual placement → Edit selected … placements** opens
  the geometry editor for detailed row changes, patterns, surface helpers, and undo.
- **Change response** changes the selected group's response without re-entering
  its geometry. Double-clicking a feature opens the same response selection.
- **Remove** removes that group's placements and mapping from the current
  vehicle. It leaves imported placement source files intact.
- **Build → Feature selection** provides individual-instance membership for
  finer comparisons.

The 3-D **Layers → Show** control affects display only. It never silently removes
a response from the calculation. Changing membership, geometry, mappings, or
physical settings invalidates previous validation.

## Build and review

**Calculate & save** performs the existing validation and then writes the
assembled response. Geometry preview and a separate validation pass are
available on Build, but are not mandatory extra clicks in the normal workflow.
Required release warnings stop automatic publication and remain visible for
review. Informational advisories are recorded without adding a required step.
Cancellation preserves the existing guarantee against publishing partial data.

The Build page contains actionable readiness checks, per-instance placement
results, optional study subsets, membership, and advanced tolerances. Frequency
and angle subsets use stored body samples; they do not introduce interpolation.
The existing coherent summation, response comparison, and feature-only output
remain the calculation path.

This is an approximation assembled from independent responses. It does not add
a full-vehicle solver for mutual coupling, multiple scattering, diffraction, or
creeping waves. The GUI improvements do not relax the response contracts or
change the electromagnetic calculations.

## Files and repeatability

Authored features are stored in managed placement CSVs using the same schemas
as imported and headless workflows. Each add/remove operation stages a new
file, runs the authoritative parser, and adopts the result only on success.
Imported CSVs are never rewritten by those operations. Adding or removing
groups preserves other groups' mappings and included/excluded membership.

Saving a recipe copies managed placement files beside the recipe so it can
survive removal of transient drafts. Input response and imported placement files
remain referenced sources; copy those referenced files too when moving the
assembly to another machine. Distinct saved variants keep independent placement
copies and output names.

## Verification targets

The relevant regression coverage includes body-only and external-body behavior,
point and line authoring, repeated names, connected and closed paths, failed
edits, membership preservation, recipe persistence, busy/editor locks, and
compact-window access to the persistent output and build controls. Numerical
backend tests remain responsible for the coherent assembly physics.
