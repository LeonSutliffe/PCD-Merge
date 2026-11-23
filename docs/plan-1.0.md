# PCD Merge 1.0: feature set and how the app works

Status: proposal for Leon to decide on (2026-11-23), restructured around
Leon's three modes. No code yet.

## Where we are (0.8.0)

Today PCD Merge is a single-window tool for one job: get several scans into one
cloud. It does that job well:

* Drag in `.pcd` files or a folder, press **Align** (FPFH + RANSAC with the
  ground set aside, ICP, pose graph, site mode for more than 6 scans).
* Escalating **Retry**, a quality report (Good / Fair / Poor, heat map, ignores
  moved objects).
* Clean-up with live preview (overlap dedup, noise, moving objects).
* Full-resolution merge to PCD / PLY / LAS / LAZ / E57 with optional minimum
  point spacing.
* Incremental "add scans to an existing merged cloud" with a JSON manifest.
* First-person walk-through, CLI, auto-updater, on-demand releases.

What it does not have, and what stops it being "more than an aligner":

1. **No saved work.** Closing the window loses the alignment, the clean-up
   settings and the view. The only memory is the merged file and its
   `merged.pcd.json`.
2. **Only reads PCD**, although it writes four other formats.
3. **No dead-end escape.** If automatic alignment fails, there is nothing the
   user can do by hand.
4. **Nothing to do with the result inside the app.** No measuring, no cutting,
   no slicing, no notes. The user has to open another program for every
   deliverable.
5. **The UI is one long left panel of buttons**, built up feature by feature.
   It will not hold another five features.

## The idea for 1.0

> Turn the app from "a merge button" into a **site workspace with three
> modes**: **Align**, **Edit** and **Inspect** (Leon's structure, 2026-11-23).
> One project per building, the original scans are never changed, and the
> user moves between the modes with tabs at the top of the window.

| Mode | Job | Quality vs speed | Can change data? |
|---|---|---|---|
| **1. Align** | Everything for getting scans into one frame and converting them: import, auto-align, Retry, manual 3-point alignment, quality report, merge, export/convert formats, min spacing, batch and incremental site work. | Downsampled for alignment, full resolution for merge (as today). | Transforms only |
| **2. Edit** | Check alignment and scan quality in orbit or walk-through, and clean the scans up: pick, brush, lasso and box selection, delete, split along a line or plane, plus the automatic clean-up (dedup, noise, moving objects). | Speed first: level-of-detail drawing, a detail slider, may drop points while moving. Edits still apply to every full-resolution point. | Yes, all undoable |
| **3. Inspect** | Read-only viewing at full quality: orbit and walk-through, click-to-measure in 3D, saved measurements, screenshots. | Quality first: a prepared viewing file streams every point. | No (only measurements and notes, which don't touch points) |

## How the modes fit together

```
 scans (PCD, E57, LAS/LAZ, PLY)
        |
   [ ALIGN ] -- transforms --> [ EDIT ] -- delete masks, splits --> [ INSPECT ]
        |                         |                                     |
   merged / converted       edited cloud export               viewing file (.pcdview)
   cloud export                                                 measurements, screenshots
```

* **Edits are stored per original scan**, as a "deleted points" mask and
  split regions, not as a new point file. So re-aligning in Align mode keeps
  the deletions, everything is undoable, and the project stays small.
* **Selections are made in space, not on drawn points.** In Edit mode only
  some points may be drawn, but a brush, lasso or box deletes *every*
  full-resolution point inside it, so lowering the detail never leaves
  half-deleted objects behind. The view shows how much is drawn
  ("showing 18% of points").
* **Automatic clean-up moves from Align to Edit**, next to the manual tools, so
  all ways of removing points are in one place and show in the same preview.
  Align keeps merge/convert, which then applies the Edit-mode deletions.
* **Inspect uses a prepared viewing file.** Building it is one step ("Prepare
  for inspection") that writes a level-of-detail octree of the edited,
  full-resolution cloud to disk. Inspect then streams it, so it can show more
  points than fit on the graphics card at full quality. The file is rebuilt
  when the edits change, and Inspect says when it is out of date.
* **The viewing file is COPC** (Cloud Optimized Point Cloud, a standard LAZ
  file with the octree built in), decided by Leon 2026-11-23 so clients can
  open it without this app. It opens in free tools such as CloudCompare, QGIS
  and web viewers that read COPC, and it is still an ordinary LAZ file for any
  LAS software. Measurements go in a small JSON file next to it.

## The modes in detail

### 1. Align

Today's features, kept as they are, plus:

* Read E57, LAS/LAZ and PLY as input.
* Manual 3-point alignment when auto fails (pick 3 matching points on two
  scans, ICP refines), and "align this scan to that one only".
* Lock a scan's transform once it has been checked.
* The current left-panel options move into a settings panel on the right.

### 2. Edit (preview and editing)

Viewing:

* Orbit mode with top / front / side / orthographic views, and the
  walk-through as a mode of the same view (no separate window).
* Colour by scan, intensity, RGB, height, or alignment error (the quality
  heat map), to spot misalignment and bad areas.
* Detail slider, eye-dome lighting, point size.

Selecting (works in both orbit and walk-through unless marked):

* **Click** a point; **brush**: a sphere under the cursor that selects as you
  paint (the natural tool in walk-through).
* **Grow to object**: click once and the connected cluster of points is
  selected (a chair, a person, a cone). The fastest way to remove small
  objects.
* **Lasso / rectangle** (orbit): selects everything under the shape through
  the whole depth, or with "front only" just the visible surface.
* **Box** and **slab between two planes** (orbit), **height range**.
* Add / subtract / invert selection, hide selection, isolate selection.

Acting on a selection:

* **Delete** (stored as a mask, undoable), **Restore** deleted points.
* **Split**: draw a line in a top or side view, or place a plane, and cut the
  scans into two parts. Each part becomes its own item in the project that can
  be shown, hidden, exported (as its own file) or deleted, e.g. split a
  building into floors. The original scans stay whole; a part is just a
  region (decided by Leon 2026-11-23).
* **Crop to selection / box**: keep only the region.
* Automatic clean-up (dedup, noise, moving objects) with its live preview.

### 3. Inspect (full view)

* Full-quality orbit and walk-through from the prepared viewing file.
* **Click-to-measure in 3D**: point to point, chained distances, height
  difference, angle, and a point's coordinates. Snaps to the nearest real
  point. Measurements are saved in the project, listed in the side panel, and
  exportable as CSV.
* Screenshots at a chosen resolution.
* No point edits. A one-click "Edit here" switches to Edit mode at the same
  view, so a problem found while inspecting is quick to fix.

## UI structure

```
+---------------------------------------------------------------------------+
| File  Edit  View  Help              [ Align ]  [ Edit ]  [ Inspect ]      |
+---------------+-------------------------------------------+---------------+
| PROJECT       |                                           | SETTINGS /    |
|  Scans (12)   |                                           | TOOLS         |
|   [x] scan01  |            3D view                        | (for the      |
|   [x] scan02  |   (orbit / top / front / side / walk)     |  current mode |
|  Parts (2)    |                                           |  or selected  |
|  Measurements |  [mode tool bar: select, brush, lasso,    |  item)        |
|  Exports      |   box, split, measure ...]                |               |
+---------------+-------------------------------------------+---------------+
| status / progress bar / time left / Cancel                                |
+---------------------------------------------------------------------------+
```

* **Left, Project tree.** Scans, split parts, measurements and exports, each
  with a tick box to show or hide it. Same in every mode.
* **Centre, 3D view.** One view for everything, orbit or walk-through. The tool
  bar along its edge changes with the mode.
* **Right, settings and tools.** Alignment options in Align, clean-up and
  selection options in Edit, measurement list in Inspect.
* **Mode tabs at the top.** Edit and Inspect show a hint until something is
  loaded; Inspect offers "Prepare for inspection" when its viewing file is
  missing or out of date.
* **Undo / Redo** for everything that changes the project.

The CLI keeps working throughout, for batch align, merge and convert.

## Project and file model

A project is a folder, so it is easy to copy, zip or put on a network drive:

```
Office Building.pcdproj/
  project.json        scans, transforms, quality, clean-up settings,
                      sections, measurements, notes, export presets
  edits/              per-scan delete masks and split regions (Edit mode)
  cache/              downsampled clouds, normals/features, preview LOD
                      (safe to delete; rebuilt on demand)
  view/               prepared viewing file for Inspect (rebuildable)
  photos/             pictures attached to notes
  exports/            default place for anything exported
```

Rules:

* **Scans are referenced, never changed.** `project.json` stores each scan's
  path (relative where possible), size and a quick hash, so a moved or
  replaced file is detected and the user is asked to point to it. Option:
  "Copy scans into the project" for handing a self-contained project over.
* **Edits are stored as instructions, not new point files.** A transform, a
  "deleted points" mask per scan, split regions, clean-up settings. The full
  cloud is only produced when exporting. This keeps projects small and every
  step reversible.
* **The cache makes reopening fast.** Opening a 30-scan project should show the
  3D view in seconds, not re-read 40 GB of scans.
* **`merged.pcd.json` keeps working.** An old merged cloud can be opened as a
  new project ("import merged cloud with manifest"), and incremental adding
  becomes simply "add scans to this project".
* **Versioned format.** `project.json` has a `format_version` so later
  releases can read 1.0 projects.

## Features by priority

### Must-have for 1.0

| Feature | Mode | Why |
|---|---|---|
| Project file, autosave, recent projects | all | Without it nothing else is worth building. |
| Mode tabs and the new layout (project tree, one 3D view, settings panel) | all | The frame for the three modes. |
| Read E57, LAS/LAZ, PLY as input | Align | Users get scans in whatever their scanner software exports. Writers already exist. |
| Manual 3-point alignment fallback, lock a scan | Align | A 1.0 can't leave the user stuck when auto fails. |
| Walk-through inside the main view, top/front/side/orthographic views | Edit, Inspect | Checking alignment and quality. |
| Colour by alignment error | Edit | Shows where scans disagree without leaving the view. |
| Click, brush and grow-to-object selection, delete, restore | Edit | Leon's core ask: remove noise and small objects in orbit or walk-through. |
| Lasso, rectangle and box selection | Edit | Mass selecting and deleting. |
| Split along a line or plane into parts | Edit | Leon's ask; also floors, wings, crop. |
| Undo / redo | Edit | Expected once users can delete. |
| Prepared viewing file (COPC) with streaming full-quality view | Inspect | Full quality on building-sized scans, and clients can open it without the app. |
| Click-to-measure in 3D (distance, height, coordinates), saved | Inspect | Leon's ask. |
| Export the edited cloud (with spacing, format) | Align | Today's merge, now with the edits applied. |

### Nice-to-have (1.0 if time allows, else 1.1)

| Feature | Why |
|---|---|
| Measuring in Edit mode too (same tool, quick checks while editing) | Saves switching modes for a quick check. |
| Notes: a pin on a point with text and optional photo, listed in the tree | Snag lists, site observations, "check this". |
| Plan outline tracing on a section: click to draw lines/polylines, export DXF | Turns a slice into a floor plan a CAD user can use. |
| Area and volume (polygon area on a section, box volume) | Quantities. |
| Change detection between two visits of the same site | The moved-objects logic already exists; showing "what's new / gone" in colour is a short step and a strong selling point. |
| Report export (HTML/PDF): scans, quality grade, heat map, measurements, sections, notes | Something to hand over or file. |
| Sections (slices) shown flat, export PNG/DXF | The basis of floor plans and elevations. |
| Free "PCD Merge Viewer": Inspect mode only, a small separate download for clients, opens COPC files and shows saved measurements | Clients get our walk-through and measuring, not just a generic viewer. |
| Walk-on-floor mode (gravity + collision) in the walk-through | Feels like walking the building. |
| Sample project in the installer and a first-run tour | Lets a new user see the workflow in 2 minutes. |

### Later (after 1.0)

* Meshing (Poisson or similar) and mesh export (OBJ/PLY).
* Automatic floor plan extraction (walls and openings from a slice).
* Plane / pipe / cylinder fitting, wall flatness and floor levelness maps.
* Georeferencing with surveyed control points, coordinate systems.
* GPU acceleration (shelved; CPU is already fast enough on Leon's PC).
* Plug-ins or scripting, a headless server mode.
* macOS / Linux builds.

### Far future

* **Colour from a camera.** Point colour (RGB) is already kept through
  align, merge, export and shown in the walk-through when a scan has it. If a
  camera is added to the device: colour points from its photos (camera
  calibration, timing sync, picking the best photo per point), exposure
  matching between scans, and 360° photo bubbles at each scan position in the
  walk-through.
* Scan-to-BIM: walls, floors, doors and windows as objects for Revit/IFC.
* Progress tracking across many visits (timeline slider in Inspect).
* Live alignment while scanning (scans streamed from the device as they are
  taken).
* Collaboration: share a project, comments on points, cloud hosting of the
  viewing file for clients.

## Suggested release path

1. **0.9: foundation.** Project file, input formats, mode tabs and the new
   layout with today's features moved into Align, walk-through inside the
   main view. Everything that works today must still work.
2. **0.10: Edit mode.** Selection tools, delete/restore, undo, split, auto
   clean-up moved here, colour by alignment error.
3. **0.11: Inspect mode.** Prepared viewing file and streaming, measuring.
   Manual 3-point alignment in Align.
4. **1.0.** Bug fixes from real scans, an Uninstall option, README rewritten as a
   user guide, sample project. Nice-to-haves that are ready go in.

Each step is a release Leon can try on real scans before the next starts.
Parallel PRs have been clashing on `gui.py`; the 0.9 layout change should split
it into one module per mode and panel so later features can be built side by
side.

## Decided (2026-11-23)

* Three modes: Align, Edit, Inspect (Leon's structure), with all the
  additions above (clean-up in Edit, full-resolution deletes, extra selection
  tools, colour by alignment error, prepared viewing file, measuring in Edit).
* Clients must be able to open the Inspect file without the app: COPC.
* A split creates parts of the project, not new scans.

## Site size (Leon, 2026-11-23)

Design for sites up to **200 m x 200 m x 200 m**. Depending on scan density
that is roughly 0.5 to several billion points in total, far more than today's
in-memory path (35.5M points in about 10 s) or the walk-through's 16M points on
the graphics card. So for 1.0:

* **Edit and Inspect must stream from disk** (the COPC octree), never load the
  whole site into memory or onto the GPU.
* **Merge and export must work in chunks** (out of core): apply each scan's
  transform and deletions and write it out piece by piece, instead of building
  one huge array.
* **Alignment is fine as it is**: it already works on downsampled copies.
* A synthetic 1-billion-point benchmark joins the CI benchmarks.

## Users (Leon, 2026-11-23)

For now: Leon and other hobbyists. So 1.0 stays focused on the three modes;
survey deliverables (DXF plans, reports, georeferencing) stay nice-to-have or
later. A short user guide and a sample project matter, because hobbyists
install it without help.

## Code signing plan

**Decided (Leon, 2026-11-23): no signing for 1.0.** The README keeps the
SmartScreen "More info > Run anyway" note. If signing is wanted later, the
free route is SignPath Foundation (no rule against AI-written code; the same
review and approval rules apply to every change). Apply around 1.0, once the
project has some public history. Needed before applying:

* Repository public, with an OSI licence (e.g. MIT) in `LICENSE`; all
  bundled libraries already have compatible licences.
* A "Code signing policy" section in the README and on the releases page, with
  the SignPath credit line, the team roles (Leon as author, reviewer and
  approver) and a privacy note (the app only contacts GitHub to check for
  updates).
* MFA on GitHub (and on SignPath).
* An **Uninstall** option (Start-menu entry or Settings > Apps), because the
  app installs itself and makes shortcuts.
* Release workflow sends the built exe to SignPath and waits for Leon's
  approval of each signing request.

Risk: SignPath checks "reputation" at its own discretion, so a young hobby
project may be asked to wait.

## Technical risk to check first

Writing COPC: laspy reads COPC but does not write it. The options are our own
octree writer on top of laspy/lazrs (the format is documented and the
walk-through already builds a similar grid), or bundling PDAL/untwine (large
on Windows). A short spike at the start of 0.11 should decide.
