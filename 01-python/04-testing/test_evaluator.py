import pytest

from ai_engineering.evaluator import Evaluator

@pytest.fixture
def evaluator():
    return Evaluator()


def test_accuracy(evaluator):
   predictions = [1, 0, 1, 1, 0]
   actual = [1, 0, 0, 1, 0]

   accuracy = evaluator.evaluate(predictions, actual)
   assert accuracy == 0.8

def test_mismatched_lengths(evaluator):
    predictions = [1, 0, 1]
    actual = [1, 0]

    with pytest.raises(ValueError):
        evaluator.evaluate(predictions, actual)

def test_perfect_accuracy(evaluator):
    predictions = [1, 0, 1, 1]
    actual = [1, 0, 1, 1]

    assert evaluator.evaluate(predictions, actual) == 1.0

def test_zero_accuracy(evaluator):
    predictions = [1, 1, 1, 1]
    actual = [0, 0, 0, 0]

    assert evaluator.evaluate(predictions, actual) == 0.0