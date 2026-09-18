import numpy as np

from linear_regression import LinearRegression


def test_model_learns_linear_relationship():
    X = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([35, 45, 55, 65, 75], dtype=float)

    model = LinearRegression(
        learning_rate=0.01,
        epochs=5000
    )

    model.fit(X, y)

    assert abs(model.w - 10) < 0.1
    assert abs(model.b - 25) < 0.1


def test_predictions_are_close():
    X = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([35, 45, 55, 65, 75], dtype=float)

    model = LinearRegression(
        learning_rate=0.01,
        epochs=5000
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert np.allclose(predictions, y, atol=0.2)


def test_loss_decreases():
    X = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([35, 45, 55, 65, 75], dtype=float)

    model = LinearRegression(
        learning_rate=0.01,
        epochs=5000
    )

    model.fit(X, y)

    assert model.loss_history[-1] < model.loss_history[0]