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


def _rq(grade, connected, overlaps, offset=0.01, agreement=0.95, voxel=0.2):
    from pcd_merge.quality import PairQuality, QualityReport

    pairs = [PairQuality(0, 1, o, 0.02, 0.01, 0.03, grade, offset, agreement) for o in overlaps]
    q = QualityReport(voxel, voxel * 1.5, pairs, [0.5] * len(connected), [0.02] * len(connected))
    assert q.grade == grade
    return SimpleNamespace(connected=connected), q


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
    # overlap noise doesn't matter; the smaller offset (misalignment) does
    assert retry.is_better(_rq("Good", [True, True], [0.61], offset=0.005),
                           _rq("Good", [True, True], [0.62], offset=0.02))
    assert not retry.is_better(_rq("Good", [True, True], [0.62], offset=0.02),
                               _rq("Good", [True, True], [0.61], offset=0.005))
    # with the same offset, more agreement wins
    assert retry.is_better(_rq("Good", [True, True], [0.6], agreement=0.95),
                           _rq("Good", [True, True], [0.6], agreement=0.80))


def test_retry_ranking_matches_the_quality_score():
    results = [_rq("Good", [True, True], [0.6], offset=o) for o in (0.03, 0.001, 0.01)]
    best = None
    for r in results:
        if retry.is_better(r, best):
            best = r
    assert best is max(results, key=lambda r: r[1].score)
