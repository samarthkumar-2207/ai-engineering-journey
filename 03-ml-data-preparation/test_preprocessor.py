import numpy as np

from preprocessor import (
    split_features_target,
    shuffle_data,
    train_test_split,
    StandardScaler,
)

def test_split_features_target():
    data = np.array([
        [10, 20, 0],
        [30, 40, 1],
    ])

    X, y = split_features_target(data)

    np.testing.assert_array_equal(
        X,
        [
            [10, 20],
            [30, 40],
        ],
    )

    np.testing.assert_array_equal(y, [0, 1])

def test_shuffle_data_keeps_alignment():
    X = np.array([
        [10],
        [20],
        [30],
        [40],
    ])

    y = np.array([100, 200, 300, 400])

    X_shuffled, y_shuffled = shuffle_data(X, y, seed=42)

    for x, target in zip(X_shuffled, y_shuffled):
        assert x[0] * 10 == target

def test_train_test_split():
    X = np.array([
        [1],
        [2],
        [3],
        [4],
        [5],
    ])

    y = np.array([10, 20, 30, 40, 50])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
    )

    assert len(X_train) == 3
    assert len(X_test) == 2

    assert len(y_train) == 3
    assert len(y_test) == 2

def test_standard_scaler():
    X = np.array([
        [10, 100],
        [20, 200],
        [30, 300],
    ])

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(
        np.mean(X_scaled, axis=0),
        [0, 0],
    )

    np.testing.assert_allclose(
        np.std(X_scaled, axis=0),
        [1, 1],
    )

def test_standard_scaler_constant_feature():
    X = np.array([
        [10, 100],
        [20, 100],
        [30, 100],
    ])

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    assert np.all(np.isfinite(X_scaled))

    np.testing.assert_array_equal(
        X_scaled[:, 1],
        [0, 0, 0],
    )