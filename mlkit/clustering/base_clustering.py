from abc import ABC, abstractmethod

from mlkit.base_model import BaseModel


class BaseClustering(BaseModel, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    def fit_predict(self, x):
        self.fit(x)
        return self.predict(x)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
