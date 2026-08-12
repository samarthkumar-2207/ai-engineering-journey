class Model:
    def __init__(self, name, version, framework):
        self.name = name
        self.version = version
        self.framework = framework

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value):
        if value < 1:
            raise ValueError("Version must be at least 1")

        self._version = value

    def show_info(self):
        print(f"Model name: {self.name}")
        print(f"Version: {self.version}")
        print(f"Framework: {self.framework}")

    def update_version(self):
        self.version += 1
        return self.version

class MLModel(Model):
    def __init__(self, name, version, framework, algorithm):
        super().__init__(name, version, framework)
        self.algorithm = algorithm
        self.evaluator = Evaluator()

    def show_info(self):
        super().show_info()
        print(f"Algorithm: {self.algorithm}")

class Evaluator:
    @staticmethod
    def calculate_accuracy(correct: int, total: int) -> float:
     return correct / total

    def evaluate(self, predictions, actual):
            if(len(predictions) != len(actual)):
                raise ValueError("Predictions and actual values must have the same length.")
            correct = 0
            for p, a in zip(predictions, actual):
                if p == a:
                    correct += 1
            return self.calculate_accuracy(correct, len(actual))
    
model1 = MLModel("fraud-detector", 1, "scikit-learn", "Random Forest")

predictions = [1, 0, 1, 1, 0]
actual = [1, 0, 0, 1, 0]

accuracy = model1.evaluator.evaluate(predictions, actual)

print(f"Accuracy: {accuracy:.2%}")

model1.update_version()
model1.show_info()


