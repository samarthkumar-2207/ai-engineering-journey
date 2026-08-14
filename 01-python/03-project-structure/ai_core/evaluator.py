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