def fitted(func):
    def wrapper(self, *args, **kwargs):
        if self._fitted:
            return func(self, *args, **kwargs)
        else:
            raise RuntimeError("Model not fitted yet")
    return wrapper
