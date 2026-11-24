"""Model mode: editable CAD geometry from the scans, not meshes.

Everything for it lives in this one package so it can be tried and, if it
is not wanted, removed cleanly:

* ``levels``  finds the floors of a building (each storey's floor and ceiling).
* ``plan``    slices the site (a floor plan at a height, or a vertical section
              along a line) and traces the walls in the slice as 2D lines.
* ``fit``     fits planes (walls, floors, ceilings, roofs) and cylinders
              (columns, pipes) and turns them into simple solids.
* ``dxf``, ``ifc``, ``step``  write those out for CAD programs.
* ``snap``    lets measurements snap to the fitted geometry.
* ``gui``     the Model mode's panel, added to the window by :func:`gui.install`.

The rest of the app does not import anything from here except one guarded
line in ``gui/main_window.py``. To remove Model mode, delete this folder,
``tests/test_cad.py``, and that ``install_model_mode`` block in
``gui/main_window.py`` (the app also runs without the folder if the block
is left in); see ``docs/model-mode.md``.
"""
