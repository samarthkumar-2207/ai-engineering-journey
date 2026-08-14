from ai_core.evaluator import Evaluator

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