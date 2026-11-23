from types import SimpleNamespace

from pcd_merge import retry
from pcd_merge.registration import AlignSettings


def test_levels_escalate_then_repeat():
    factors = [retry.level(k).voxel_factor for k in range(1, 6)]
    assert factors == sorted(factors)
    assert factors[0] > 1
    assert factors[-1] == factors[-2]  # past the last level it stays there


def test_settings_for_scales_voxel_and_attempts():
    base = AlignSettings(voxel_size=None, anchor=True)
    s = retry.settings_for(base, 0.2, 2, seed=7)
    assert s.voxel_size == 0.2 * retry.level(2).voxel_factor
    assert s.ransac_attempts == base.ransac_attempts * retry.level(2).attempts_factor
    assert s.seed == 7
    assert s.anchor is True  # everything else is kept


def _rq(grade, connected, overlaps, rmse=0.05):
    pairs = [SimpleNamespace(overlap=o, overlaps=o > 0.02, rmse=rmse) for o in overlaps]
    return SimpleNamespace(connected=connected), SimpleNamespace(grade=grade, pairs=pairs)


def test_best_result_ordering():
    poor = _rq("Poor", [True, True], [0.5])
    fair = _rq("Fair", [True, True], [0.2])
    fair_more = _rq("Fair", [True, True], [0.4])
    good_split = _rq("Good", [True, False, True], [0.6])
    good_all = _rq("Good", [True, True, True], [0.3])
    assert retry.is_better(poor, None)
    assert retry.is_better(fair, poor)  # grade beats overlap
    assert not retry.is_better(poor, fair)
    assert retry.is_better(fair_more, fair)
    assert retry.is_better(good_all, good_split)  # more clouds joined wins
    assert not retry.is_better(fair, fair)  # a tie keeps the earlier result
    # overlap noise doesn't matter; the more precise result does
    assert retry.is_better(_rq("Good", [True, True], [0.61], rmse=0.03),
                           _rq("Good", [True, True], [0.62], rmse=0.05))
    assert not retry.is_better(_rq("Good", [True, True], [0.62], rmse=0.05),
                               _rq("Good", [True, True], [0.61], rmse=0.03))
