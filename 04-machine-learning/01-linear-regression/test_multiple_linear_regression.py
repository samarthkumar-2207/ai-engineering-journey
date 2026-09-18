import numpy as np

from multiple_linear_regression import MultipleLinearRegression


def test_model_learns_multiple_features():

    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6]
    ], dtype=float)

    y = np.array([
        10,
        15,
        20,
        25,
        30
    ], dtype=float)

    model = MultipleLinearRegression(
        learning_rate=0.01,
        epochs=5000
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert np.allclose(predictions, y, atol=0.001)


def test_model_has_one_weight_per_feature():

    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4]
    ], dtype=float)

    y = np.array([
        10,
        15,
        20
    ], dtype=float)

    model = MultipleLinearRegression()

    model.fit(X, y)

    assert len(model.w) == X.shape[1]


def test_loss_decreases():

    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6]
    ], dtype=float)

    y = np.array([
        10,
        15,
        20,
        25,
        30
    ], dtype=float)

    model = MultipleLinearRegression(
        learning_rate=0.01,
        epochs=5000
    )

    model.fit(X, y)

    assert model.loss_history[-1] < model.loss_history[0]