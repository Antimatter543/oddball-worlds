from collections import Counter

from oddball.common.splits import _bucket, known_positive_tics, split_of


def test_stable_and_deterministic():
    assert [split_of(t) for t in range(1000, 1100)] == [split_of(t) for t in range(1000, 1100)]
    assert _bucket(123456789) == _bucket(123456789)


def test_proportions_roughly_80_10_10():
    known = known_positive_tics()
    c = Counter(split_of(t) for t in range(1, 50_001) if t not in known)
    n = sum(c.values())
    assert abs(c["train"] / n - 0.8) < 0.01
    assert abs(c["val"] / n - 0.1) < 0.01
    assert abs(c["test"] / n - 0.1) < 0.01


def test_known_positives_override():
    known = known_positive_tics()
    assert len(known) > 100  # docs/known_positives.csv is committed
    assert 185336364 in known  # Boyajian's star
    assert all(split_of(t) == "known" for t in known)
