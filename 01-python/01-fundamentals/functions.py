def calculate_average(values: list[float]) -> float:
    return sum(values) / len(values)

def calculate_accuracy(correct: int, total: int) -> float:
    return correct / total

def evaluate_model(predictions: list[int], actual: list[int]) -> float:
    if len(predictions) != len(actual):
        raise ValueError("Predictions and actual values must have the same length.")
    correct = 0
    for p, a in zip(predictions, actual):
        if p == a:
            correct += 1
    return calculate_accuracy(correct, len(actual))

predictions = [1, 0, 1, 1, 0]
actual = [1, 0, 0, 1, 0]

accuracy = evaluate_model(predictions, actual)
print(f"Model Accuracy: {accuracy:.2%}")
