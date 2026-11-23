"""Escalating Retry: each press tries something genuinely different.

A plain re-run only changes RANSAC's random start; the voxel size and the
features are the same every time, so a systematic failure (repetitive
walls, little overlap, a scale that doesn't suit the scene) comes back
unchanged. Instead each Retry uses a coarser voxel size, so features describe
larger neighbourhoods, and more RANSAC runs. The final ICP pass refines at
half the voxel size, so precision is lost only mildly; the quality report
still measures the result against the full clouds.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .quality import GRADES
from .registration import AlignSettings


@dataclass(frozen=True)
class RetryLevel:
    voxel_factor: float  # times the voxel size used by the first Align
    attempts_factor: int  # times the normal number of RANSAC runs

    def describe(self, voxel: float) -> str:
        return (f"voxel {voxel:.3g} ({self.voxel_factor:g}× the first try), "
                f"{self.attempts_factor}× more search attempts")


# Press 1, 2, 3, ... of Retry. After the last level, further presses repeat
# it with a new random start.
LEVELS = (
    RetryLevel(1.5, 2),
    RetryLevel(2.0, 3),
    RetryLevel(3.0, 4),
)


def level(press: int) -> RetryLevel:
    """The level for the ``press``-th Retry (1-based)."""
    return LEVELS[min(press, len(LEVELS)) - 1]


def settings_for(base: AlignSettings, base_voxel: float, press: int,
                 seed: int) -> AlignSettings:
    lv = level(press)
    return dataclasses.replace(
        base,
        voxel_size=base_voxel * lv.voxel_factor,
        ransac_attempts=base.ransac_attempts * lv.attempts_factor,
        seed=seed,
    )


def score(result, quality) -> tuple:
    """Bigger is better. Grade first, then how many clouds joined up, then
    :attr:`quality.QualityReport.score`, which ranks by the estimated
    misalignment (offset), then agreement, then overlap. Moved objects don't
    affect any of these, so they can't make a worse alignment win."""
    return (
        -GRADES.index(quality.grade),
        int(sum(bool(c) for c in result.connected)),
        round(float(quality.score), 6),
    )


def is_better(new: tuple, best: tuple | None) -> bool:
    """Whether a new (result, quality) beats the best so far."""
    return best is None or score(*new) > score(*best)
