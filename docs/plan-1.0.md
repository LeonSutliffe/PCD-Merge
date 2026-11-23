# PCD Merge 1.0: feature set and how the app works

Status: proposal for Leon to decide on (2026-11-23). No code yet.

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

> Turn the app from "a merge button" into a small **site workspace**: one
> project per building, the scans stay untouched, and the user can go from raw
> scans to the things they hand over (measurements, sections, plan
> outlines, a cleaned and thinned cloud, a short report) without leaving it.

Alignment stays the core and the reason people start the app. Everything new
works on the aligned scans, so it is useful before or without a merge.

## Workflow

The app follows the order the work happens in, shown as steps along the top of
the window. Each step can be revisited; later steps are never locked, they
just say what they are missing.

| Step | What the user does | Exists today? |
|---|---|---|
| 1. **Scans** | Create or open a project, drop scans (PCD, E57, LAS/LAZ, PLY). See them in their original positions. | Partly (PCD only, no project) |
| 2. **Align** | Auto-align, Retry, check the quality report. If a scan won't go, align it by picking 3 matching points. Lock scans that are right. | Yes, except manual pick and lock |
| 3. **Clean** | Dedup, noise, moving objects with live preview, plus a crop box and a manual "delete selection" brush. | Mostly; crop and manual delete are new |
| 4. **Use** | Walk through, measure, cut sections, trace plan outlines, add notes and photos to points. | Walk-through only |
| 5. **Export** | Merged cloud (format, spacing, region), sections as DXF/PNG, measurements as CSV, a PDF/HTML report. | Merged cloud only |

Steps 3 to 5 work on the *aligned scans plus transforms*, not on a merged file.
"Merge" becomes one kind of export, not the end of the program.

## UI structure

```
+---------------------------------------------------------------------------+
| File  Edit  View  Help           [1 Scans][2 Align][3 Clean][4 Use][5 Export] |
+---------------+-------------------------------------------+---------------+
| PROJECT       |                                           | INSPECTOR     |
|  Scans (12)   |                                           | (settings for |
|   [x] scan01  |            3D view                        |  the current  |
|   [x] scan02  |   (orbit / top / front / side / walk)     |  step or the  |
|  Sections (3) |                                           |  selected     |
|  Measurements |                                           |  item)        |
|  Notes (5)    |                                           |               |
|  Exports      |  [tool bar: select, measure, section,     |               |
|               |   crop, note, pick points]                |               |
+---------------+-------------------------------------------+---------------+
| status / progress bar / time left / Cancel                                |
+---------------------------------------------------------------------------+
```

* **Left, Project tree.** Everything the project contains, each with a tick box
  to show or hide it. Replaces today's file list.
* **Centre, 3D view.** One view for everything. Standard views (top, front,
  side, orbit) plus the walk-through as a mode of the same view instead of a
  separate window. A small tool bar for the tools of the current step.
* **Right, Inspector.** The settings of the current step (alignment options,
  clean-up options, export options) or of the selected item (a section's
  height, a measurement's value). Today's left-panel options move here.
* **Top, step bar.** Doubles as guidance for new users: the next step is
  highlighted, and a step with a problem (a scan that didn't align) shows a
  warning mark.
* **Undo / Redo** for everything that changes the project (lock, delete
  points, move a note, change a transform).

The current layout stays as a fallback during the change, and the CLI keeps
working throughout.

## Project and file model

A project is a folder, so it is easy to copy, zip or put on a network drive:

```
Office Building.pcdproj/
  project.json        scans, transforms, quality, clean-up settings,
                      sections, measurements, notes, export presets
  cache/              downsampled clouds, normals/features, preview LOD
                      (safe to delete; rebuilt on demand)
  photos/             pictures attached to notes
  exports/            default place for anything exported
```

Rules:

* **Scans are referenced, never changed.** `project.json` stores each scan's
  path (relative where possible), size and a quick hash, so a moved or
  replaced file is detected and the user is asked to point to it. Option:
  "Copy scans into the project" for handing a self-contained project over.
* **Edits are stored as instructions, not new point files.** A transform, a
  crop box, a "deleted points" mask per scan, clean-up settings. The full
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

These are what make it a release and not a 0.9.

| Feature | Why |
|---|---|
| Project file (above), autosave, recent projects | Without it nothing else is worth building. |
| Read E57, LAS/LAZ, PLY as input (same fields kept as for PCD) | Users get scans in whatever their scanner software exports. Writers already exist. |
| Manual 3-point alignment fallback | A 1.0 can't leave the user stuck when auto fails. Pick 3 matching points, then ICP refines. Also "align this scan to that one only". |
| Lock / unlock a scan's transform | So Retry and new scans don't move scans the user already checked. |
| New layout: project tree, single 3D view, inspector, step bar | Room for the new tools. |
| Standard views: top, front, side, orthographic toggle | Needed for sections and plans, and for checking alignment. |
| Measure: point-to-point distance, height difference, and a picked point's coordinates | The most common question asked of a scan. Snaps to the nearest real point. |
| Crop box / clipping box | Hide everything outside a box to look inside rooms, and export only that region. |
| Sections (slices): horizontal or vertical slab of set thickness, shown flat | The basis of floor plans and elevations. Export PNG at a chosen scale and DXF as points. |
| Export the merged cloud from the project (with crop, clean-up, spacing) | Today's merge, moved into the Export step. |
| Undo / redo | Expected once users can edit. |
| Code signing (or a decision not to) | SmartScreen's "Run anyway" looks broken to new users. See open questions. |

### Nice-to-have (1.0 if time allows, else 1.1)

| Feature | Why |
|---|---|
| Notes: a pin on a point with text and optional photo, listed in the tree | Snag lists, site observations, "check this". |
| Plan outline tracing on a section: click to draw lines/polylines, export DXF | Turns a slice into a floor plan a CAD user can use. |
| Area and volume (polygon area on a section, box volume) | Quantities. |
| Change detection between two visits of the same site | The moved-objects logic already exists; showing "what's new / gone" in colour is a short step and a strong selling point. |
| Manual point delete (brush or lasso) | For things the automatic clean-up misses. |
| Report export (HTML/PDF): scans, quality grade, heat map, measurements, sections, notes | Something to hand over or file. |
| Eye-dome lighting and point-size controls | Makes scans much easier to read without colour. |
| Walk-on-floor mode (gravity + collision) in the walk-through | Feels like walking the building. |
| Sample project in the installer and a first-run tour | Lets a new user see the workflow in 2 minutes. |

### Later (after 1.0)

* Meshing (Poisson or similar) and mesh export (OBJ/PLY).
* Automatic floor plan extraction (walls and openings from a slice).
* Plane / pipe / cylinder fitting, wall flatness and floor levelness maps.
* Georeferencing with surveyed control points, coordinate systems.
* Panoramic photo colourising (if the sensor has a camera).
* GPU acceleration (shelved; CPU is already fast enough on Leon's PC).
* Plug-ins or scripting, a headless server mode.
* macOS / Linux builds.

## Suggested release path

1. **0.9: foundation.** Project file, input formats, new layout with the
   current features moved into it, lock, undo. No new tools yet; everything
   that works today must still work.
2. **0.10: rescue and look.** Manual 3-point alignment, standard views, crop
   box.
3. **0.11: use.** Measure, sections with export, merged export from the
   project.
4. **1.0.** Bug fixes from real scans, signed installer, README rewritten as a
   user guide, sample project. Nice-to-haves that are ready go in.

Each step is a release Leon can try on real scans before the next starts.
Parallel PRs have been clashing on `gui.py`; the layout change in 0.9 should
split it into one module per panel/step so later features can be built side by
side.

## Open questions for Leon

1. **Who uses it, and what do they hand over?** This decides the "Use" and
   "Export" steps more than anything else. Typical answers and what they pull
   in:
   * *Just me, to get a clean merged cloud for other software* → keep 1.0
     small: projects, formats, manual alignment, crop, export.
   * *Surveyors / measured-building work* → sections, plan tracing, DXF,
     measurements, coordinates, maybe georeferencing.
   * *Construction / facilities* → notes with photos, change detection between
     visits, report export.
   * *Visualisation / clients* → walk-through polish, colour, mesh.
2. **Project folder as proposed, or stay file-based** (open scans, export,
   nothing saved)? Recommended: project folder.
3. **Code signing.** A certificate costs roughly a few hundred dollars a year
   (Azure Trusted Signing is cheaper). Worth it if people outside your own PC
   will install it.
4. **Is "1.0" for you only, or for other people?** If others, a user guide,
   sample data and a first-run tour become must-haves.
5. **Which is the most-wanted tool after alignment**: measuring, sections /
   floor plans, notes, or change detection? That one goes first in 0.11.
6. **Scan sizes to design for.** How many scans and points in your biggest
   site? It sets how hard the cache and level-of-detail work needs to be.
