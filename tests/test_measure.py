"""Measurements: values, saving in the project, CSV export."""

import csv
import math

import pytest

from pcd_merge.measure import Measurement, write_csv
from pcd_merge.project import Project


def test_distance_values():
    m = Measurement("distance", [[0, 0, 0], [3, 4, 12]])
    assert m.length == pytest.approx(13)
    assert m.horizontal == pytest.approx(5) and m.vertical == pytest.approx(12)
    assert m.slope_deg == pytest.approx(math.degrees(math.atan2(12, 5)))
    assert m.label() == "Distance 13.000 m"
    assert "Height difference +12.000 m" in m.details()
    p = Measurement("point", [[1.5, -2, 3.25]])
    assert p.details() == ["X 1.500", "Y -2.000", "Z 3.250"]
    with pytest.raises(ValueError):
        Measurement("distance", [[0, 0, 0]])
    with pytest.raises(ValueError):
        Measurement("angle", [[0, 0, 0]])


def test_saved_in_the_project_and_csv(tmp_path):
    proj = Project.create(str(tmp_path / "Site"))
    proj.measurements += [Measurement("distance", [[0, 0, 0], [1, 0, 0]], name="Door"),
                          Measurement("point", [[5, 5, 5]], visible=False)]
    proj.save()
    again = Project.open(proj.folder)
    assert [m.label() for m in again.measurements] == ["Door", "Point (5.000, 5.000, 5.000)"]
    assert again.measurements[1].visible is False
    out = tmp_path / "m.csv"
    write_csv(str(out), again.measurements)
    rows = list(csv.DictReader(open(out)))
    assert rows[0]["name"] == "Door" and float(rows[0]["distance"]) == 1.0
    assert rows[1]["kind"] == "point" and rows[1]["distance"] == ""
