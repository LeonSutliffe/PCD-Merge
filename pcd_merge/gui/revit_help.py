"""How to get the app's files into Autodesk Revit.

Revit only links point clouds in Autodesk's own indexed formats (.rcp
projects and .rcs scans), which only Autodesk software can write. ReCap,
which comes with Revit, turns an E57 file into those, so "Export for Revit"
saves an E57 (always in metres, double-precision coordinates, intensity and
colour) for ReCap to index.
"""

from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

STEPS = """\
<p><b>1. Index it in Autodesk ReCap</b> (it comes with Revit):</p>
<ol>
<li>Open ReCap and choose <i>New Project</i>, then <i>Import point cloud</i>.</li>
<li>Give the project a name and folder, pick the .e57 file and press
<i>Index scans</i>, then <i>Launch project</i>.</li>
<li>Save the project. ReCap writes an <b>.rcp</b> file (and its .rcs scan
files beside it). Keep them together.</li>
</ol>
<p><b>2. Link it in Revit:</b> <i>Insert</i> &gt; <i>Point Cloud</i>, pick the
.rcp file, set <i>Positioning</i> to <b>Auto - Origin to Origin</b> and press
<i>Open</i>.</p>
<p><b>Model mode IFC files</b> line up with the cloud if they are placed the
same way: <i>Insert</i> &gt; <i>Link IFC</i> (stays a reference and updates when
you save the IFC again), or <i>File</i> &gt; <i>Open</i> &gt; <i>IFC</i> to start
a Revit project with its levels. Walls, floors, columns and the rest come in
under matching Revit categories, as fixed shapes rather than native editable
walls.</p>
<p>The E57 is in metres. ReCap and Revit read it so, whatever units your
Revit project shows.</p>
"""


def show_revit_steps(parent, saved_path: str | None = None, notes=()) -> None:
    """The steps above, after an export (``saved_path``) or on their own."""
    box = QMessageBox(parent)
    box.setWindowTitle("PCD Merge: open in Revit")
    box.setIcon(QMessageBox.Information)
    head = (f"<p>Saved <b>{html.escape(saved_path)}</b>.</p>" if saved_path else "")
    extra = "".join(f"<p>{html.escape(n[0].upper() + n[1:])}.</p>" for n in notes)
    box.setTextFormat(Qt.RichText)
    box.setText(head + STEPS + extra)
    box.exec()
