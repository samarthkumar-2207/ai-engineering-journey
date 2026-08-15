import pytest

from exceptions import calculate_accuracy


@pytest.mark.parametrize(
  "correct,total,expected",
  [
    (80, 100, 0.8),
    (100, 100, 1.0),
    (0, 100, 0.0),
    (50, 100, 0.5),
  ],
)
def test_calculate_accuracy(correct, total, expected):
    assert calculate_accuracy(correct, total) == expected

def test_zero_total():
    with pytest.raises(
        ValueError,
        match="Total must be greater than zero."):
        calculate_accuracy(80, 0)


def test_negative_correct():
    with pytest.raises(
        ValueError,
        match="Correct must be non-negative."):
        calculate_accuracy(-1, 100)


def test_correct_greater_than_total():
    with pytest.raises(
        ValueError,
        match="Correct cannot be greater than total."):
        calculate_accuracy(101, 100)
