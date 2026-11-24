"""IFC files (IFC4) for BIM programs: Revit, ArchiCAD, FreeCAD (BIM
workbench), BlenderBIM / Bonsai, SketchUp Pro, Tekla, Solibri, any IFC
viewer.

Each fitted solid becomes a real building element, not a mesh: an
``IfcWall``, ``IfcSlab`` (floors, ceilings and roofs), ``IfcColumn`` or
``IfcPipeSegment``, made of a rectangle or circle pushed out to a depth
(``IfcExtrudedAreaSolid``), so the receiving program can change its
length, thickness or height. Elements are grouped into the building's
storeys (``IfcBuildingStorey``) from the levels found.

Every element carries a property set ``PCD_Merge_Fit`` with how far the
scan points were from it (``FitRMS``) and how many points support it.

Coordinates stay those of the project (metres). The site is placed at a
round number near the building, and everything inside it relative to
that, so programs that work far from their origin badly still get small
numbers.
"""

from __future__ import annotations

import datetime as _dt
import math
import time
import uuid

import numpy as np

_B64 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"

CLASSES = {
    # kind -> (IFC class, predefined type)
    "wall": ("IFCWALL", ".SOLIDWALL."),
    "slab": ("IFCSLAB", ".FLOOR."),
    "roof": ("IFCSLAB", ".ROOF."),
    "column": ("IFCCOLUMN", ".COLUMN."),
    "pipe": ("IFCPIPESEGMENT", ".RIGIDSEGMENT."),
    "plane": ("IFCBUILDINGELEMENTPROXY", ".NOTDEFINED."),
}


def ifc_guid(u: uuid.UUID | None = None) -> str:
    """The 22 character compressed GUID IFC uses."""
    n = (u or uuid.uuid4()).int
    out = []
    for _ in range(22):
        out.append(_B64[n & 63])
        n >>= 6
    return "".join(reversed(out))


def step_str(s: str) -> str:
    """A STEP string: quotes doubled, anything outside plain ASCII encoded."""
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "'":
            out.append("''")
        elif ch == "\\":
            out.append("\\\\")
        elif 32 <= o < 127:
            out.append(ch)
        elif o <= 0xFFFF:
            out.append(f"\\X2\\{o:04X}\\X0\\")
        else:
            out.append(f"\\X4\\{o:08X}\\X0\\")
    return "'" + "".join(out) + "'"


def step_real(x: float) -> str:
    x = float(x)
    if not math.isfinite(x):
        x = 0.0
    if x == 0:
        return "0."
    s = f"{x:.10g}"
    if "e" in s or "E" in s:
        mant, exp = s.lower().split("e")
        if "." not in mant:
            mant += "."
        return f"{mant}E{int(exp)}"
    return s if "." in s else s + "."


class StepWriter:
    """Numbers the entities of a STEP physical file (#1 = ...)."""

    def __init__(self):
        self.lines: list[str] = []

    def add(self, text: str) -> str:
        self.lines.append(text)
        return f"#{len(self.lines)}"

    def point(self, p) -> str:
        return self.add("IFCCARTESIANPOINT((" + ",".join(step_real(v) for v in p) + "))")

    def direction(self, d) -> str:
        return self.add("IFCDIRECTION((" + ",".join(step_real(v) for v in d) + "))")

    def placement3d(self, origin, axis=None, ref=None) -> str:
        p = self.point(origin)
        a = self.direction(axis) if axis is not None else "$"
        r = self.direction(ref) if ref is not None else "$"
        return self.add(f"IFCAXIS2PLACEMENT3D({p},{a},{r})")

    def text(self, header: list[str]) -> str:
        body = [f"#{k + 1}={line};" for k, line in enumerate(self.lines)]
        return "\n".join(["ISO-10303-21;", "HEADER;"] + header + ["ENDSEC;", "DATA;"] + body
                         + ["ENDSEC;", "END-ISO-10303-21;", ""])


def write_ifc(path: str, elements, levels=None, project: str = "Scanned building",
              app_version: str = "") -> dict:
    """Write the elements (and the storeys) as IFC4. Returns counts per class."""
    from .model import Model

    levels = sorted(levels or [], key=lambda lv: lv.floor)
    model = Model(levels, list(elements))
    w = StepWriter()
    person = w.add("IFCPERSON($,$,'',$,$,$,$,$)")
    org = w.add("IFCORGANIZATION($,'PCD Merge',$,$,$)")
    pando = w.add(f"IFCPERSONANDORGANIZATION({person},{org},$)")
    app = w.add(f"IFCAPPLICATION({org},{step_str(app_version or '1')},'PCD Merge','PCDMerge')")
    now = int(time.time())
    owner = w.add(f"IFCOWNERHISTORY({pando},{app},$,.ADDED.,{now},$,$,{now})")
    units = [w.add("IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.)"),
             w.add("IFCSIUNIT(*,.AREAUNIT.,$,.SQUARE_METRE.)"),
             w.add("IFCSIUNIT(*,.VOLUMEUNIT.,$,.CUBIC_METRE.)"),
             w.add("IFCSIUNIT(*,.PLANEANGLEUNIT.,$,.RADIAN.)")]
    unit_set = w.add("IFCUNITASSIGNMENT((" + ",".join(units) + "))")
    world = w.placement3d((0.0, 0.0, 0.0))
    north = w.add("IFCDIRECTION((0.,1.))")
    ctx = w.add(f"IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-05,{world},{north})")
    body_ctx = w.add(f"IFCGEOMETRICREPRESENTATIONSUBCONTEXT('Body','Model',*,*,*,*,{ctx},$,"
                     ".MODEL_VIEW.,$)")
    proj = w.add(f"IFCPROJECT('{ifc_guid()}',{owner},{step_str(project)},$,$,$,$,({ctx}),"
                 f"{unit_set})")

    # the site sits at a round number near the building
    pts = np.concatenate([e.corners() for e in elements]) if len(elements) else np.zeros((1, 3))
    site_at = np.floor(pts.min(axis=0) / 10.0) * 10.0
    site_at[2] = 0.0
    site_pl = w.add(f"IFCLOCALPLACEMENT($,{w.placement3d(site_at)})")
    site = w.add(f"IFCSITE('{ifc_guid()}',{owner},'Site',$,$,{site_pl},$,$,.ELEMENT.,"
                 "$,$,$,$,$)")
    bldg_pl = w.add(f"IFCLOCALPLACEMENT({site_pl},{w.placement3d((0.0, 0.0, 0.0))})")
    bldg = w.add(f"IFCBUILDING('{ifc_guid()}',{owner},{step_str(project)},$,$,{bldg_pl},"
                 "$,$,.ELEMENT.,$,$,$)")
    w.add(f"IFCRELAGGREGATES('{ifc_guid()}',{owner},$,$,{proj},({site}))")
    w.add(f"IFCRELAGGREGATES('{ifc_guid()}',{owner},$,$,{site},({bldg}))")

    if not levels:
        from .levels import Level

        z0 = float(pts[:, 2].min()) if len(elements) else 0.0
        levels = [Level("Level 0", z0)]
        model.levels = levels
    storeys = []
    for lv in levels:
        pl = w.add(f"IFCLOCALPLACEMENT({bldg_pl},{w.placement3d((0.0, 0.0, lv.floor))})")
        st = w.add(f"IFCBUILDINGSTOREY('{ifc_guid()}',{owner},{step_str(lv.name)},$,$,{pl},"
                   f"$,$,.ELEMENT.,{step_real(lv.floor)})")
        storeys.append((st, pl, lv.floor))
    w.add(f"IFCRELAGGREGATES('{ifc_guid()}',{owner},$,$,{bldg},"
          f"({','.join(s for s, _, _ in storeys)}))")

    contained: dict[int, list[str]] = {}
    counts: dict[str, int] = {}
    numbers: dict[str, int] = {}
    for e in elements:
        k = model.level_of(e.level_height()) or 0
        st, st_pl, floor_z = storeys[k]
        cls, ptype = CLASSES.get(e.kind, CLASSES["plane"])
        u, v, wdir = e.axes()
        origin = np.asarray(e.origin, float) - site_at - np.array([0.0, 0.0, floor_z])
        local = w.add(f"IFCLOCALPLACEMENT({st_pl},{w.placement3d(origin, wdir, u)})")
        if e.shape == "cylinder":
            pos2 = w.add(f"IFCAXIS2PLACEMENT2D({w.add('IFCCARTESIANPOINT((0.,0.))')},$)")
            prof = w.add(f"IFCCIRCLEPROFILEDEF(.AREA.,$,{pos2},{step_real(e.radius)})")
        else:
            su, sv, _sw = map(float, e.size)
            centre = w.add(f"IFCCARTESIANPOINT(({step_real(su / 2)},{step_real(sv / 2)}))")
            pos2 = w.add(f"IFCAXIS2PLACEMENT2D({centre},$)")
            prof = w.add(f"IFCRECTANGLEPROFILEDEF(.AREA.,$,{pos2},{step_real(su)},"
                         f"{step_real(sv)})")
        solid = w.add(f"IFCEXTRUDEDAREASOLID({prof},{w.placement3d((0.0, 0.0, 0.0))},"
                      f"{w.direction((0.0, 0.0, 1.0))},{step_real(float(e.size[2]))})")
        rep = w.add(f"IFCSHAPEREPRESENTATION({body_ctx},'Body','SweptSolid',({solid}))")
        shape = w.add(f"IFCPRODUCTDEFINITIONSHAPE($,$,({rep}))")
        numbers[e.kind] = numbers.get(e.kind, 0) + 1
        from .model import KINDS

        name = e.name or f"{KINDS.get(e.kind, 'Element')} {numbers[e.kind]}"
        el = w.add(f"{cls}('{ifc_guid()}',{owner},{step_str(name)},"
                   f"{step_str(e.summary())},$,{local},{shape},{step_str(e.id)},{ptype})")
        props = [
            w.add(f"IFCPROPERTYSINGLEVALUE('FitRMS',$,IFCLENGTHMEASURE({step_real(e.rms)}),$)"),
            w.add(f"IFCPROPERTYSINGLEVALUE('ScanPoints',$,IFCINTEGER({int(e.points)}),$)"),
            w.add("IFCPROPERTYSINGLEVALUE('Source',$,IFCLABEL('Fitted to laser scan by "
                  "PCD Merge'),$)"),
        ]
        pset = w.add(f"IFCPROPERTYSET('{ifc_guid()}',{owner},'PCD_Merge_Fit',$,"
                     f"({','.join(props)}))")
        w.add(f"IFCRELDEFINESBYPROPERTIES('{ifc_guid()}',{owner},$,$,({el}),{pset})")
        contained.setdefault(k, []).append(el)
        counts[cls] = counts.get(cls, 0) + 1
    for k, els in contained.items():
        w.add(f"IFCRELCONTAINEDINSPATIALSTRUCTURE('{ifc_guid()}',{owner},$,$,"
              f"({','.join(els)}),{storeys[k][0]})")

    stamp = _dt.datetime.now().replace(microsecond=0).isoformat()
    import os

    header = [
        "FILE_DESCRIPTION(('ViewDefinition [ReferenceView_V1.2]'),'2;1');",
        f"FILE_NAME({step_str(os.path.basename(path))},'{stamp}',(''),(''),'PCD Merge',"
        f"'PCD Merge {step_str(app_version)[1:-1]}','');",
        "FILE_SCHEMA(('IFC4'));",
    ]
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(w.text(header))
    return counts
