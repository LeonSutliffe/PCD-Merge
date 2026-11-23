"""The self-test the Windows build runs (``PCD Merge.exe --selftest``) must pass
from source too, so a broken import is caught before the Windows build."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


def test_selftest_passes(tmp_path):
    from pcd_merge.selftest import run

    log = tmp_path / "selftest.log"
    assert run(str(log)) == 0, log.read_text()
    text = log.read_text()
    assert "read back ok" in text and "project ok" in text and "GUI ok" in text
