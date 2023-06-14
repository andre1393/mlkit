from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        self.model_fitted = False

    def fit(self, *args, **kwargs):
        r = self._fit(*args, **kwargs)
        self.model_fitted = True
        return r

    @abstractmethod
    def _fit(self, *args, **kwargs):
        pass
