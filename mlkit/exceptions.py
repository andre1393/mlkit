class NotFittedException(Exception):
    def __init__(self, message=("This instance is not fitted yet. Call 'fit' with appropriate arguments before using"
                                "this estimator.")):
        self.message = message
        super().__init__(self.message)
