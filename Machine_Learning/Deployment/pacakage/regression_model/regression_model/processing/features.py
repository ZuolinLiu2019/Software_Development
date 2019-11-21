import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from regression_model.processing.error import InvalidModelInputError

class LogTransformer(BaseEstimator, TransformerMixin):
    """Logarithm transform. """

    def __init__(self, varaibles=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accomodate the pipeline
        return self

    def transform(self, X):
        X = X.copy()

        # check that the values are positive for log transform
        if not (X[self.variables] > 0).all().all():
            vars_ = self.variables[(X[self.variables] <= 0).any()]
            raise InvalidModelInputError(
                f'Variables contain zero or negative values, '
                f'can't apply log for vars: {_vars_}')
        for feature in self.variables:
            X[feature] = np.log(X[feature])

        return X