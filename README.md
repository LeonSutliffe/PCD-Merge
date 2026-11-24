# PCD Merge

A desktop program for LiDAR point clouds: drag in two or more scans (PCD, PLY,
LAS/LAZ or E57), it aligns them automatically, shows you the result in 3D, and on your OK merges
them into a single file (PCD, PLY, LAS/LAZ or E57).

![PCD Merge after aligning three scans](docs/screenshot.png)

## How it works

* **Only rigid transforms** (rotation + translation) are estimated, because all
  scans come from the same sensor and have the same scale.
* **Alignment runs on thinned copies.** Each cloud is voxel-downsampled, normals
  and FPFH features are computed, and a RANSAC search over feature matches finds
  the alignment without needing any starting guess. Each candidate is refined
  with point-to-plane ICP (coarse, then finer), and the best one is kept.
* **The ground is not used to match features.** The dominant plane (usually the
  ground) is set aside when matching and scoring, because flat ground looks the
  same everywhere and otherwise produces convincing but wrong alignments. It is
  still used by ICP to get height and tilt exactly right.
* **Three or more clouds:** pairs are registered, the most reliable pairs
  form a pose graph, and the graph is optimised so all clouds agree with each
  other. The first file is the reference frame. A pair only counts if its
  walls, poles and objects line up, not just its ground.
* **Whole sites (more than 6 scans):** instead of trying every pair, each scan
  is matched with the next one in file order (scanners usually number files in
  capture order), then scans that are still on their own are matched against
  the rest until everything is connected, and finally every pair that now sits
  side by side gets an extra quick ICP check (a loop closure) so small errors
  do not pile up along a long chain. Files that match nothing, such as a scan
  from another site, are reported and can be left out of the merge.
* **Merging uses every original point.** The transforms are applied to the
  full-resolution files, and all per-point fields (intensity, ring, timestamp,
  normals, colour, ...) are kept. Normals are rotated along with the points.
  Points with NaN coordinates are dropped. The output is an unorganised binary
  PCD (optionally `binary_compressed`).

The app reports, for every pair, how much of the clouds overlap after
alignment, and warns when a cloud overlaps the others by less than 30% or could
not be matched at all. In that case press **Retry**. Each press tries something
different: a coarser voxel size (1.5×, then 2×, then 3× the first try) with
more search attempts, which helps with repetitive or low-detail scenes. Every
try is graded on the same scale as the first, and the best result so far is
the one shown and merged: best grade first, then the smallest offset (see the
quality report below), so moved objects can't make a worse alignment win. Press **Align** to start over.

## Install on Windows (no Python needed)

1. Download the latest `PCD-Merge-…-windows-x64.zip` from
   [Releases](https://github.com/LeonSutliffe/PCD-Merge/releases/latest).
2. Unzip it anywhere and double-click **PCD Merge.exe** inside.
3. The first time, it offers to install itself for your user account. Say yes:
   it copies itself to `%LOCALAPPDATA%\PCD Merge`, puts a **PCD Merge**
   shortcut on your Desktop and in the Start menu, and starts. You can delete
   the unzipped folder afterwards.

From then on, start it from the shortcut. Each start checks for a newer
release; if there is one it asks, downloads it, swaps it in and starts the new
version. "Skip this version" stops it asking about that release. Offline, it
just starts.

Windows may show a SmartScreen warning the first time because the program is
not code-signed: choose *More info*, then *Run anyway*.

To check an installation works: `"PCD Merge.exe" --selftest` (writes
`%TEMP%\pcd-merge-selftest.log`).

## Install from source

Python 3.10 or newer.

```bash
git clone https://github.com/LeonSutliffe/PCD-Merge.git
cd PCD-Merge
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[fast]"
```

`[fast]` adds `python-lzf` for quick `binary_compressed` files. If it fails to
install on your machine, use `pip install -e .`; compressed files are then
still readable, just more slowly.

On Linux, Open3D needs `libegl1 libgl1 libgomp1` from your package manager.

## Use

### Desktop app

```bash
pcd-merge            # or: python -m pcd_merge
pcd-merge "Office.pcdproj"   # open a project
```

The window has three parts and a row of mode tabs at the top right:

* **Left, the project:** the scans (tick to show or hide, double-click a
  missing one to point to where it is now), and the **merged clouds** made
  from the project. After a merge the result is listed there and shown in the
  view instead of the scans; tick and untick scans and merged clouds to
  compare them. **Parts** are made by splitting in Edit, and **Measurements**
  with the Measure button (tick to show, double-click to rename, right-click to
  delete).
* **Middle, the 3D view:** orbit, top / front / side, an orthographic option,
  and **Walk through** (first person) inside the same view. **Hide panels**
  (F11, or *View > Hide side panels*) hides both side panels so the view fills
  the window; press it again to bring them back.
* **Right, the settings of the current mode.** **Align** has everything below.
  **Edit** cleans the scans up and splits them (see *Edit mode*). **Inspect** is
  for looking and measuring (see *Measuring*).

**Edit mode.** Choose a tool and select points in the view: **Brush** (click
or drag; the size is in metres), **Rectangle** and **Lasso** (everything inside
the shape, through the whole depth of the view: in a Top view that is a column
from the ground up), and **Object** (click a chair, a person or a car to take
all of it, but not the floor or the wall it touches). Shift adds to the
selection, Alt + drag rotates the view. **Delete** (the Del key) hides the
selected points; **Restore** brings deleted ones back (tick *Show deleted
points* to see them). *Edit > Undo / Redo* (Ctrl+Z / Ctrl+Y) step through the
edits. An edit is a shape in space saved in the project, so it applies to
every full-resolution point inside it, not just to the points drawn: merging,
**Save cleaned cloud…** and the walk-through all leave them out. The scan
files are never changed. If some scans are hidden, edits apply to the visible
ones only.

**Parts.** *Split into parts* cuts the site in two at a height (*Split here*,
for floors) or along a **Cut line** you drag across the view (in a Top view, a
vertical wall along the line). The parts are listed on the left: tick to show or
hide them, double-click to rename, right-click > *Save this part…* to save one
on its own. A part can be split again; Undo joins them back. **Colour: Height**
in the view makes floors easy to tell apart. The **automatic clean-up**
(duplicates, noise, moving objects) is set in Edit mode too and is applied when
saving.

**Projects.** *File > Save project as…* makes a folder `Name.pcdproj` that
remembers the scans, the alignment, the quality report, the settings and the
exports. From then on it saves itself after every change. *File > Close project*
closes it (saved) and leaves an empty window. Opening it again
(*File > Open project…* or *Open recent*) shows the aligned view straight away
from its cache, without re-reading the scans; they are read only when
something needs them (Align, the walk-through, clean-up). The scan files are
never changed. If they were moved, the app says so; point to one and the others
in the same folder are found too. Move the project and the scans together and
it finds them by itself. *Copy scans into the project* makes a project you can
hand over on its own. *Open a merged cloud as a new project* starts from a
cloud merged earlier, to add new scans to it.

1. Drag scans (`.pcd`, `.ply`, `.las`, `.laz`, `.e57`), or a whole folder
   (subfolders included), onto the window. An E57 file holding several scans
   adds each scan on its own, with its scanner position.
   *Original positions* shows the scans as they are in the files.
2. Press **Align**. The view switches to *Aligned*, each cloud in its own colour.
3. Check the result. Untick a file in the list to hide it. The quality report
   (see below) grades the alignment Good, Fair or Poor.
4. **Accept and merge...** writes the merged file, or **Retry** to try again.
   Pick the format in the save dialog (see *Output formats* below).
   **Save transforms...** writes the 4x4 matrices and the quality report as JSON.

If some clouds could not be aligned, merging asks whether to leave them out or
merge them in their original position.

**Align by hand.** When Align and Retry can't place a scan, use *Align by
hand* in the Align panel. Choose the scan to **Move** and a scan **Onto**
which it goes (one that is already right), press **Pick points**, and click
the same spot on each in turn: a corner, a pole, a door frame. The fixed
scan's picks are marked in yellow and the moving scan's in magenta, and only
the two scans are drawn while you pick. Three pairs are needed, or two with
*Scans are level* ticked (only the heading is found, for levelled tripod
scans). The panel says how well the pairs agree and which one fits worst,
which catches a mis-click. **Apply** places the scan; *Tighten automatically
afterwards* then fine-tunes it on the scans themselves, so the clicks only
need to be roughly right. The quality report and heat map update as after
Align.

**Adding new scans later.** Press **Add new scans to an existing merged
cloud…** and pick a merged cloud you saved earlier, then add the new scans (or
the whole site folder again) and press **Align**. The merged cloud stays
exactly where it is; the new scans are aligned to it and appended when you
merge. Every merge writes `merged.pcd.json` next to the file with the list of
scans in it and their transforms, so scans that are already in it are skipped
automatically. If the merged cloud has a `source` field, the new scans get the
next numbers.

To make the output smaller, set **Min. point spacing** (in the scans' units,
e.g. `0.02` for 2 cm). The merged cloud is then thinned so no two points are
closer than that. Original points are kept, not averaged, so intensity and the
other fields stay exactly as measured. The default, *Full resolution*, saves
every point.

### Measuring

Press **Measure** above the 3D view (or M), in any mode, and click two points
for the distance between them, or one point for its coordinates. Clicks snap
to the nearest point drawn under the mouse; Esc starts again. A distance also
gives the horizontal distance, the height difference (second point minus
first) and the slope. Measurements are drawn in the view with their value,
saved in the project, and listed on the left under *Measurements*.

**Inspect** mode opens with measuring on. Its panel chooses *Distance* or
*Point*, lists the measurements with their details, deletes them, exports them
all as a CSV spreadsheet, and saves a **screenshot** of the view, as shown or
two or four times larger for printing.

**Make viewing file…** (in Inspect) saves the site as a COPC file (`.copc.laz`,
Cloud Optimized Point Cloud): every point, with deleted points and the Edit
clean-up left out as in a merge, stored in the cubes of an octree with an
index. Viewers read only the cubes in view, at the detail they need, so even a
building-sized site opens at once in CloudCompare, QGIS, Potree or
viewer.copc.io, and it stays an ordinary LAZ file for any other LAS software.
COPC is also offered in the save dialog of every merge. The file is written
in two passes through a temporary file next to it, so allow about twice its
uncompressed size (36 bytes a point) of free disk space while it is made.

**Make browser walk-through…** (in Inspect) saves the site as **one `.html`
file that opens in any web browser** (Chrome, Edge, Firefox, Safari) by
double-clicking it, straight into a walk-through. It needs no app, no
internet and no installing, so it can be emailed, put on a USB stick or
shared with a link to download. Clients can walk around (the same controls as
the app's walk-through), switch to an **Overview** of the whole site, change
the colours and the detail, jump to the scan positions and **measure**
distances (click two points). The measurements shown in the app are included.
It shows the ticked scans, aligned, without deleted points or hidden parts,
plus ticked merged clouds, thinned evenly to 2, 5 or 10 million points
(about 20, 50 or 100 MB; positions kept to under a millimetre). Choose
*Small* for phones and tablets. Browsers don't let a page opened from a file
read other files, which is why everything is inside the one file.

### Walk-through (first person)

Press **Walk through** above the 3D view to walk or fly around inside the scan,
in the same view (Esc or **Leave walk-through** goes back to orbiting). It shows what the 3D view shows: the
aligned clouds after **Align** (or their original positions), and only the
files that are ticked. To walk through a merged file you saved earlier, load
just that file and press **Walk through**. Leaving the walk-through keeps it:
pressing **Walk through** again goes straight back to where you were, unless
what is shown has changed.

| | |
|---|---|
| W A S D or arrow keys | move (level, like walking) |
| Mouse | look around: click into the view first (Esc gives the mouse back), or drag |
| Space / E, C / Q | up, down |
| Shift, Ctrl | 4x faster, 4x slower while held |
| Mouse wheel | change the walking speed |
| 1 to 9 | jump to where scan 1 to 9 was taken |
| Home | back to the start |

The walk starts where the first scan was taken (from its PCD `VIEWPOINT`), so
you begin inside the building at tripod height. **Fly where I look** makes W
move in the direction you are looking instead of level. **Colour** offers the
scan's RGB colours if it has them, intensity, height, or one colour per scan.

Large scans stay smooth because not every point is drawn every frame. Up to
16 million points are kept on the graphics card, sorted into small cubes.
Nearby cubes are drawn in full; further away fewer points are drawn, and drawn
larger, so they still look solid. Cubes behind you are skipped. While you move,
the number of points drawn is lowered automatically if the frame rate drops,
and the full detail comes back as soon as you stop. **Detail** sets how many
points it aims for; lower it on a slower graphics card.

**Editing and measuring while walking.** Every Edit tool works in the
walk-through: choose Brush, Rectangle, Lasso, Object or Cut line in the Edit
panel (or press B, R, L, O) and click or drag in the walk-through; **Delete**,
**Restore**, *Show deleted points*, Undo / Redo, splitting and parts all show
there at once. **Measure** (M) works the same way: click two points. With a
tool chosen, the left button is the tool's and the **right button looks
around**; if the mouse is captured (click with no tool chosen), the tool acts
at the cross in the middle of the view. A lasso or rectangle drawn in the
walk-through stops just past the nearest surface inside it, so a lasso round a
person does not also take the rooms behind the wall; *Lasso depth* sets a
fixed depth instead. Esc first drops a half-made measurement or shape, then
leaves the walk-through.

### Model mode (CAD from the scans)

The **Model** tab makes editable CAD geometry from the scans instead of a
mesh: it finds the building's levels, traces **floor plans and sections** into
lines (DXF, for any CAD program), and fits **walls, floor slabs, roofs, columns
and pipes** as simple solids with their real dimensions (IFC for Revit,
ArchiCAD, FreeCAD and other BIM programs; STEP for Fusion, SolidWorks and other
mechanical CAD). Measurements snap to the fitted corners, edges and faces. It
is a self-contained add-on that can be removed; see
[docs/model-mode.md](docs/model-mode.md).

### Input formats

| Format | What is read |
| --- | --- |
| `.pcd` | every field (ascii, binary, binary_compressed) |
| `.ply` | every vertex property (ascii or binary); red/green/blue become `rgb` |
| `.las` / `.laz` | coordinates, intensity, colour, GPS time, returns, classification, point source ID and every extra field |
| `.e57` | each scan in the file separately, with its pose applied: coordinates (cartesian or spherical), intensity, colour, time stamp, row/column |

Colour is always kept as 8 bits per channel. A LAS file this program wrote
gives back its exact intensities (from `intensity_raw`).

### Output formats

| Format | Opens in | What is kept |
| --- | --- | --- |
| `.pcd` | PCL, CloudCompare, ROS | every field |
| `.ply` | CloudCompare, MeshLab, Blender | every field (`rgb` becomes red/green/blue, normals become nx/ny/nz) |
| `.las` / `.laz` | survey and GIS software, CloudCompare | intensity, colour, GPS time (from `timestamp`/`time`/`t`), `source` as point source ID; every other field as a LAS extra field |
| `.e57` | Recap, Cyclone, Scene, CloudCompare | coordinates (double precision), intensity, colour, time stamp; other fields are left out |
| `.copc.laz` | CloudCompare, QGIS, PDAL, Potree, [viewer.copc.io](https://viewer.copc.io); any LAS software as plain LAZ | as `.laz`, with the points arranged in an octree so viewers load only what is on screen |

LAS stores intensity as whole numbers 0-65535. Integer intensities are stored
as they are; fractional ones (e.g. 0-1 floats) are rescaled to the full 16-bit
range and the exact values are also kept in an `intensity_raw` extra field.
LAS coordinates are stored at 0.1 mm resolution. The app tells you after saving
if anything could not be stored in the chosen format.

Merging without clean-up, thinning or PCD compression is streamed: each scan is
moved and written a few million points at a time, so the merged cloud never has
to fit in memory. Those three options need every point at once.

Voxel size is *Automatic* by default (about 40,000 points per thinned cloud).
Set it by hand, in the scans' units, if alignment fails: larger is more robust
and faster, smaller is more precise. It never affects which points are merged.

### Alignment quality report

After aligning, the app measures how well the clouds agree, using the final
transforms (what will actually be merged):

* **Per pair:** overlap (share of points with a match in the other cloud), the
  **offset** (how far the clouds still are from their best fit, i.e. the
  estimated misalignment), the **fit** (RMS distance between the points that
  match) and how much of the overlap **changed** between the scans.
* **Grade:** *Good* when the offset is within 0.2 x the voxel size, at least
  30% overlaps and at least 60% of the overlap matches; *Fair* up to 0.5 x
  voxel, 15% overlap and 40% matching; otherwise *Poor*. The overall grade is
  the worst overlapping pair. Merging a *Poor* result asks for confirmation
  first.
* **Heatmap:** set *Colour* to *Alignment error* to colour every point by its
  distance to the other clouds. Blue fits, red is at the end of the scale,
  magenta *changed* means the scans disagree there by more than that, and grey
  was seen by only one scan (not an error). The app switches to the heatmap by
  itself when the grade is not *Good*.

Distances are point-to-plane (to the other cloud's local surface, not its
nearest point), so two scans of the same wall read close to the sensor noise
even though their points never coincide. Points further away than 1.5 x the
voxel size count as not overlapping.

**Moved objects don't count against the grade.** A car that left, a person
walking through or a door that opened makes the scans disagree in that spot,
but says nothing about the alignment. The offset comes from a robust ICP
(Tukey loss) started at the result: it ignores points that don't fit, so it
measures how far the *static* parts are off. Changed points (magenta) are also
left out of the fit. A real misalignment moves everything, so it still shows
up in the offset, and in walls and poles turning magenta along their whole
length rather than in a few separate objects.

![Heatmap of two well aligned scans where cars, people and a van moved: those show as magenta, the grade stays Good](docs/quality-heatmap.png)

### Clean-up while merging (optional)

By default every original point goes into the merged file. Three optional
steps, under *Clean-up while merging*, can tidy it up. They only ever **remove**
points: nothing is resampled, averaged or moved, so the points that remain keep
their exact coordinates and every field. The finished dialog says how many
points each step removed from each file.

Once the clouds are aligned, ticking or unticking an option updates the 3D
preview straight away: the points that would be removed are highlighted in
white (or hidden, if you choose *Hide*), with an estimate of how many points
that is. The preview works on the sample of each cloud that the view shows, so
it is quick even for very large scans; the merge itself uses every point.

* **Remove duplicate points where scans overlap.** Where two scans cover the
  same spot you otherwise get two interleaved copies of every surface. Space is
  split into small cubes (*cell*, automatic = a few times the point spacing); in
  each cube that several scans reach, only the points of the scan that measured
  it most densely are kept. That is usually the closest and most accurate scan,
  so its full detail is kept. Areas only one scan saw are never touched.
* **Remove isolated noise points.** Statistical outlier removal on each scan:
  points far from all their neighbours (dust, rain, stray returns) are dropped.
  Set conservatively, so real surfaces stay.
* **Remove moving objects (people, cars) seen in only one scan.** A point from
  one scan is dropped when another scan looked straight *through* that spot and
  hit something further away, and has no point of its own there. Things that
  were merely hidden from the other scan are kept. This needs each scan in its
  scanner's own frame (normal for raw scans; the PCD `VIEWPOINT` is used as the
  scanner position). A cloud that does not look like a single-position scan is
  not used as evidence, and the report says so.

### Command line

```bash
pcd-merge scan1.pcd scan2.pcd scan3.pcd -o merged.pcd --transforms transforms.json
pcd-merge scan1.pcd scan2.pcd -o merged.laz --min-spacing 0.02
```

Scans can be any input format, mixed. Folders work too:
`pcd-merge site_scans/ -o merged.pcd`. To add new scans to an earlier result:

```bash
pcd-merge --add-to merged.pcd new_scans/ -o merged.pcd
```

The output extension picks the format (`.pcd`, `.ply`, `.las`, `.laz`, `.e57`).
Options: `--voxel 0.3`, `--min-spacing 0.02` (thin the output), `--seed 1`
(repeatable runs), `--source-field` (adds a `source` field telling which file
each point came from), `--compressed` (PCD only), `--skip-unaligned` (leave out
clouds that could not be aligned) and `--force` (merge them anyway, in their
original position). Clean-up: `--dedup` (or `--dedup 0.1` to set the cell
size), `--denoise`, `--remove-moving`. The quality report is printed after
alignment.

## Limits and next steps

* Needs real overlap between scans; below roughly 30% alignment gets unreliable.
* Very repetitive or symmetric places (long identical corridors, empty fields)
  can align to the wrong spot. The quality report and the error heatmap are there
  to catch this.
* Planned (see `docs/plan-1.0.md`): drawing the COPC viewing file in the app itself, streamed
  at full quality.

## Large scans

Loading, aligning and merging run in the background. The progress bar shows
percent done, elapsed time and an estimate of the time left, and **Cancel**
stops the current step without losing anything already loaded.

Measured on three synthetic scans of about 24 million points each (71 million
in total, 4-core machine), from loading to a saved merged file:

| | Time | Peak memory |
|---|---|---|
| 0.3.0 | 55 to 60 s | 4.9 GB |
| 0.6.0 | 30 to 33 s | 3.8 GB |

To time your own machine: `python benchmarks/bench_large.py --points 20000000`.

GPU acceleration is not used: the Windows build of Open3D is CPU-only.

## Releasing a new version

Merging a PR no longer publishes anything, and PRs don't need to change the
version. Merge as many PRs as you like, then when you're ready:

1. On GitHub open **Actions > release > Run workflow** (branch `main`).
2. Leave *minor* selected (0.7.1 → 0.8.0), or pick *patch* / *major*, or type an
   exact version. Press **Run workflow**.
3. About 10 minutes later release `v0.8.0` is published with the zip, and the
   version bump is committed to `main`. Release notes list the PRs merged since
   the previous release.
4. Installed copies offer the update the next time they start.

If the build or its tests fail, nothing is published.

Checks on PRs: the quick tests run on every PR. The full Windows build (slow)
only runs when packaging, dependencies, the updater or the self-test change; to
build any branch anyway, use **Actions > windows build > Run workflow**. Its zip
is under the run's artifacts.
To build locally on Windows: `pip install -e ".[build]"` then
`pyinstaller packaging/pcd_merge.spec` (output in `dist\PCD Merge`).

## Development

```bash
pip install -e ".[test]"
pytest
```

The tests build a synthetic outdoor scene, cut overlapping scans out of it with
known poses, and check that alignment recovers those poses and that merging
keeps every point and field.
