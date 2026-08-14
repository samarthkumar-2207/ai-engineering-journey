from ai_core.model import MLModel

model1 = MLModel("fraud-detector", 1, "scikit-learn", "Random Forest")

predictions = [1, 0, 1, 1, 0]
actual = [1, 0, 0, 1, 0]

accuracy = model1.evaluator.evaluate(predictions, actual)

print(f"Accuracy: {accuracy:.2%}")

model1.show_info()