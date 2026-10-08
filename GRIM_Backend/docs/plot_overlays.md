# Editable plot overlays

Use **Overlays** in the Plotting or ISAR toolbar to load coordinates or draw connected points. Available on ISAR images, Az. vs D.R., and Range–Freq maps. Create a plot first; controls become available when an axis shows a distance unit.

## Load an outline

1. Open **Overlays**. If the figure has several panels, choose the target **Panel**.
2. Click **Load overlay…** and choose an XY or XYZ text file.
3. For XYZ, choose **XY**, **XZ**, or **YZ**. The first chosen column becomes the horizontal plot coordinate and the second the vertical coordinate. This selects a plane; it does not rotate geometry using the ISAR viewing angle.
4. Select the **file length unit**: metres, centimetres, millimetres, inches, or feet. This describes the file values, which may differ from the plot units. On plots with an angle or frequency axis, that coordinate uses the units printed on its axis; the length-unit choice affects only distance coordinates.

Accepted files include `.xy`, `.xyz`, `.csv`, `.txt`, `.dat`, and `.pts`. Use two or three numeric columns separated by commas, semicolons, spaces, or tabs. An optional `x,y` or `x,y,z` header is supported. Lines beginning with `#` are comments. Blank lines separate disconnected outlines. All coordinate values must be finite; malformed or inconsistent rows are reported with their line number. Files are limited to 32 MiB and 200,000 points.

Example XY file:

```text
x,y
-0.3,-0.2
0.3,-0.2
0.3,0.4
-0.3,0.4
-0.3,-0.2
```

Points connect in file order. Repeating the first coordinate at the end closes an outline. Use **Points only** to show a point cloud without connecting segments.

## Draw and edit

- Click **Draw points**, then click positions on the plot. Each point connects to the previous one. Use **Finish drawing** or Escape to finish. Each drawing belongs to one panel; start another drawing for a different panel.
- Drag a point with the left mouse button to move it freely. This also works for points loaded from a file. Pan, Zoom Box, and data-marker modes take precedence; turn them off to drag overlay points. Starting a drawing turns those modes off.
- Right-click a point and choose **Edit coordinates…** to enter its X and Y values in the currently displayed units. Coordinates display five decimal places; scientific notation is supported when editing. Accepting the dialog without editing a coordinate preserves its full stored precision. **Remove point** deletes that point.
- Choose an overlay in the **Overlay** list, or click/right-click one of its points, to select it. Set its color, solid/dashed/dotted/dash-dot line style, width, and point visibility. Color, line type, and width are also available in the right-click menu.
- **Show** hides or reveals the selected overlay. **Remove** deletes the selected overlay. The plot's existing **Clear** button clears the plot and that tab's overlays.

Overlays survive ordinary redraws and length-unit changes while the application remains open. Plotting and ISAR keep separate overlays. They return only on a compatible axis type and their original panel number, preventing a distance outline from appearing on an unrelated frequency/power plot. Their coordinates stay in the chosen plot frame; changing the ISAR look angle does not rotate them as a 3D object.

## Measure

- Click an overlay line segment to show its length. This measures the segment between its two neighboring points, not the entire connected outline. You can also right-click the line and choose **Measure segment**.
- Click **Measure**, then click two overlay points. The points can belong to different overlays in the same panel. Right-click **Measure from this point** and **Measure to this point** are also available.
- The plot shows a dashed measurement line and a label; the controls show the length and changes in X and Y. Values use five decimal places. Escape or **Finish measuring** returns to normal point dragging, while the measurement remains visible. **Clear measurement** removes it.
- Measurements follow the selected points when you drag or edit them, and convert when the plot's length units change. Removing a referenced point or overlay clears its measurement. Only the latest measurement is shown for each plot tab.
- On ISAR and other plots with two length axes, the length is the straight-line distance in the displayed plane. If the axes use different length units, the total uses the X-axis unit. For XYZ imports, the unselected coordinate does not contribute to this planar measurement.
- On plots that mix distance with angle or frequency, only the separate changes in X and Y are shown. Those quantities cannot form a physical straight-line length. Blank lines in an imported outline are never treated as connecting segments.

Drawing, measuring, Pan, Zoom Box and data-marker modes are mutually exclusive. Measurements do not change the image data, its color scale or the plot limits.

## Save and export

**Save XY…** saves the selected overlay's edited, projected coordinates as a CSV file in the displayed axis units, retaining blank-line breaks. Comments identify both axes and their units. Choose those units when loading the saved file. XYZ's unselected third coordinate is not exported by Save XY. Coordinate files do not store line styling.

Five decimal places is a display format. Calculations and saved coordinate files retain full floating-point precision. Shorter labels improve readability but do not reduce the memory per coordinate or materially accelerate loading or reconstruction.

**Export Plot** includes visible overlays in the PNG or PDF. Overlays are visual annotations: they do not alter datasets, image intensity, reconstruction, or image color limits. The Python recorder explicitly notes that annotated exports cannot be replayed from the underlying plot recipe alone.

Use Save XY to keep drawings for another session. Overlays are not automatically saved into datasets or application preferences. No additional package installation is required.
