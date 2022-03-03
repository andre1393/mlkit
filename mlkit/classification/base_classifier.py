from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    def __init__(self):
        self.classes_ = None

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def predict_proba(self, x):
        pass
