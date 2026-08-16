import numpy as np

from dataset_analyzer import analyze_dataset


def test_feature_means():
    X = np.array([
        [100, 2],
        [200, 4],
        [300, 6],
    ])

    means, _, _, _ = analyze_dataset(X)

    np.testing.assert_array_equal(means, [200, 4])


def test_centered_mean():
    X = np.array([
        [100, 2],
        [200, 4],
        [300, 6],
    ])

    _, _, centered, _ = analyze_dataset(X)

    means = np.mean(centered, axis=0)

    np.testing.assert_allclose(means, [0, 0])


def test_normalized_mean():
    X = np.array([
        [100, 2],
        [200, 4],
        [300, 6],
    ])

    _, _, _, normalized = analyze_dataset(X)

    means = np.mean(normalized, axis=0)

    np.testing.assert_allclose(means, [0, 0])