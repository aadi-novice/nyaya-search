import pytest

from nyaya.eval.stats import paired_bootstrap


def test_identical_samples_give_zero_interval():
    ci = paired_bootstrap([1, 0, 1, 1], [1, 0, 1, 1], samples=500, seed=7)
    assert ci.mean == ci.low == ci.high == 0.0
    assert ci.n == 4 and ci.samples == 500


def test_clear_drop_excludes_zero():
    a = [1.0] * 80 + [0.0] * 20
    b = [1.0] * 50 + [0.0] * 50
    ci = paired_bootstrap(a, b, samples=2000, seed=7)
    assert ci.mean == pytest.approx(0.30)
    assert 0.0 < ci.low < 0.30 < ci.high
    assert ci.scaled(100).mean == pytest.approx(30.0)


def test_deterministic_and_chunked():
    a = [i % 3 == 0 for i in range(50)]
    b = [i % 2 == 0 for i in range(50)]
    one = paired_bootstrap(a, b, samples=3000, seed=1)
    two = paired_bootstrap(a, b, samples=3000, seed=1)
    assert one == two
    assert one.low <= one.mean <= one.high


def test_errors():
    with pytest.raises(ValueError, match="length"):
        paired_bootstrap([1], [1, 2], samples=10, seed=1)
    with pytest.raises(ValueError, match="no paired"):
        paired_bootstrap([], [], samples=10, seed=1)


def test_as_dict():
    d = paired_bootstrap([1, 0], [0, 0], samples=10, seed=1).as_dict()
    assert set(d) == {"mean", "low", "high", "samples", "n"}
