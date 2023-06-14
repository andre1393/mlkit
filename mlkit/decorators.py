from mlkit.exceptions import NotFittedException


def validate_model_fitted(func):
    def wrapper(self, *args, **kwargs):
        if not self.model_fitted:
            raise NotFittedException()
        return func(self, *args, **kwargs)
    return wrapper
