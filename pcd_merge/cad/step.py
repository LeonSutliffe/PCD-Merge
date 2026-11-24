"""STEP files (AP214) for mechanical CAD: Fusion, SolidWorks, Inventor,
Onshape, FreeCAD, Rhino, SketchUp (with an importer), Creo, CATIA.

Each fitted solid is written as an exact boundary model: a box with six
flat faces, a cylinder with a curved side and two flat ends. They come in
as real solids (not meshes) that can be cut, joined, filleted and
measured. All solids are in one part; most programs list them as separate
bodies (in FreeCAD: *Part > Compound > Explode compound*). Units: metres.
"""

from __future__ import annotations

import datetime as _dt
import os

import numpy as np

from .ifc import step_real, step_str

# box corners k = a + 2b + 4c; faces as corner loops, counter-clockwise from outside
BOX_FACES = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]


class _W:
    def __init__(self):
        self.lines: list[str] = []

    def add(self, text: str) -> str:
        self.lines.append(text)
        return f"#{len(self.lines)}"

    def pt(self, p) -> str:
        return self.add("CARTESIAN_POINT('',(" + ",".join(step_real(v) for v in p) + "))")

    def dir(self, d) -> str:
        d = np.asarray(d, float)
        d = d / np.linalg.norm(d)
        return self.add("DIRECTION('',(" + ",".join(step_real(v) for v in d) + "))")

    def ax(self, origin, axis, ref) -> str:
        return self.add(f"AXIS2_PLACEMENT_3D('',{self.pt(origin)},{self.dir(axis)},"
                        f"{self.dir(ref)})")

    def vertex(self, p) -> str:
        return self.add(f"VERTEX_POINT('',{self.pt(p)})")

    def line_edge(self, va, vb, a, b) -> str:
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        length = float(np.linalg.norm(d))
        vec = self.add(f"VECTOR('',{self.dir(d)},{step_real(length)})")
        ln = self.add(f"LINE('',{self.pt(a)},{vec})")
        return self.add(f"EDGE_CURVE('',{va},{vb},{ln},.T.)")

    def face(self, oriented_edges, surface, outer=True) -> str:
        loop = self.add("EDGE_LOOP('',(" + ",".join(oriented_edges) + "))")
        bound = self.add(f"FACE_{'OUTER_' if outer else ''}BOUND('',{loop},.T.)")
        return self.add(f"ADVANCED_FACE('',({bound}),{surface},.T.)")

    def oe(self, edge, forward: bool) -> str:
        return self.add(f"ORIENTED_EDGE('',*,*,{edge},{'.T.' if forward else '.F.'})")


def _box(w: _W, e, name: str) -> str:
    c = e.corners()
    verts = [w.vertex(p) for p in c]
    edges = {}

    def edge(i, j):
        key = (min(i, j), max(i, j))
        if key not in edges:
            edges[key] = w.line_edge(verts[key[0]], verts[key[1]], c[key[0]], c[key[1]])
        return edges[key], i < j

    faces = []
    for loop in BOX_FACES:
        p = [c[k] for k in loop]
        n = np.cross(p[1] - p[0], p[3] - p[0])
        ref = p[1] - p[0]
        plane = w.add(f"PLANE('',{w.ax(p[0], n, ref)})")
        oes = []
        for k in range(4):
            ed, fwd = edge(loop[k], loop[(k + 1) % 4])
            oes.append(w.oe(ed, fwd))
        faces.append(w.face(oes, plane))
    shell = w.add("CLOSED_SHELL('',(" + ",".join(faces) + "))")
    return w.add(f"MANIFOLD_SOLID_BREP({step_str(name)},{shell})")


def _cylinder(w: _W, e, name: str) -> str:
    o = np.asarray(e.origin, float)
    u, _v, ax = e.axes()
    r, h = e.radius, float(e.size[2])
    top = o + h * ax
    v0, v1 = w.vertex(o + r * u), w.vertex(top + r * u)
    c_bot = w.add(f"CIRCLE('',{w.ax(o, ax, u)},{step_real(r)})")
    c_top = w.add(f"CIRCLE('',{w.ax(top, ax, u)},{step_real(r)})")
    e_bot = w.add(f"EDGE_CURVE('',{v0},{v0},{c_bot},.T.)")
    e_top = w.add(f"EDGE_CURVE('',{v1},{v1},{c_top},.T.)")
    e_seam = w.line_edge(v0, v1, o + r * u, top + r * u)
    side_srf = w.add(f"CYLINDRICAL_SURFACE('',{w.ax(o, ax, u)},{step_real(r)})")
    side = w.face([w.oe(e_bot, True), w.oe(e_seam, True), w.oe(e_top, False),
                   w.oe(e_seam, False)], side_srf, outer=False)
    bot = w.face([w.oe(e_bot, False)], w.add(f"PLANE('',{w.ax(o, -ax, u)})"))
    tp = w.face([w.oe(e_top, True)], w.add(f"PLANE('',{w.ax(top, ax, u)})"))
    shell = w.add(f"CLOSED_SHELL('',({side},{bot},{tp}))")
    return w.add(f"MANIFOLD_SOLID_BREP({step_str(name)},{shell})")


def write_step(path: str, elements, name: str = "Scanned building") -> int:
    """Write every element as a solid. Returns how many were written."""
    from .model import KINDS

    w = _W()
    appc = w.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    w.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,"
          f"{appc})")
    pc = w.add(f"PRODUCT_CONTEXT('',{appc},'mechanical')")
    pdc = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',{appc},'design')")
    prod = w.add(f"PRODUCT({step_str(name)},{step_str(name)},'',({pc}))")
    w.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,({prod}))")
    pdf = w.add(f"PRODUCT_DEFINITION_FORMATION('','',{prod})")
    pd = w.add(f"PRODUCT_DEFINITION('design','',{pdf},{pdc})")
    pds = w.add(f"PRODUCT_DEFINITION_SHAPE('','',{pd})")
    length = w.add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT($,.METRE.))")
    angle = w.add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    solid_angle = w.add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),{length},"
                "'distance_accuracy_value','confusion accuracy')")
    ctx = w.add(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT(({unc}))"
                f"GLOBAL_UNIT_ASSIGNED_CONTEXT(({length},{angle},{solid_angle}))"
                "REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY'))")
    origin = w.ax((0, 0, 0), (0, 0, 1), (1, 0, 0))
    solids = []
    numbers: dict[str, int] = {}
    for e in elements:
        numbers[e.kind] = numbers.get(e.kind, 0) + 1
        label = e.name or f"{KINDS.get(e.kind, 'Element')} {numbers[e.kind]}"
        solids.append(_cylinder(w, e, label) if e.shape == "cylinder" else _box(w, e, label))
    rep = w.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION({step_str(name)},"
                f"({','.join([origin] + solids)}),{ctx})")
    w.add(f"SHAPE_DEFINITION_REPRESENTATION({pds},{rep})")

    stamp = _dt.datetime.now().replace(microsecond=0).isoformat()
    header = [
        "FILE_DESCRIPTION(('Solids fitted to a laser scan by PCD Merge'),'2;1');",
        f"FILE_NAME({step_str(os.path.basename(path))},'{stamp}',(''),(''),'PCD Merge',"
        "'PCD Merge','');",
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));",
    ]
    body = [f"#{k + 1}={line};" for k, line in enumerate(w.lines)]
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(["ISO-10303-21;", "HEADER;"] + header + ["ENDSEC;", "DATA;"] + body
                          + ["ENDSEC;", "END-ISO-10303-21;", ""]))
    return len(solids)
