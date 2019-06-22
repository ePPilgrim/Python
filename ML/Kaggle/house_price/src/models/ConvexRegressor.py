import numpy as np
from sklearn.base import RegressorMixin

class ConvexRegressor(RegressorMixin):
    __delim__ = '$$&&$$'
    __coef__ = 'c$o$e$f$'

    def __init__(self, ests, convparams=None):
        self.ests = ests
        self.convparams = convparams
        if convparams is None:
            val = 1.0 / len(self.estimators)
            self.convparams = {x.__class__.__name__: val for x in self.ests}
        self.__estrep__ = {x.__class__.__name__: x for x in ests}

    def get_params(self, deep=True):
        return {'ests': self.ests, 'convparams': self.convparams}

    def set_params(self, **params):
        for key, val in params.items():
            estname, delim, var = key.partition(self.__delim__)
            if delim is '':
                self.__setattr__(key, val)
            else:
                self.__estrep__[estname].set_params(**{var: val})
        return self

    def fit(self, X, y, sample_weight=None):
        for est in self.ests:
            est = est.fit(X, y)
        return self

    def predict(self, X):
        y = 0
        for est in self.ests:
            y += self.convparams[est.__class__.__name__] * est.predict(X)
        return y


def convregparam(estimator, params):
    return {estimator.__class__.__name__ + ConvexRegressor.__delim__ + x: y for x, y in params.items()}


def convregconv(ests, convparams):
    return {ests[i].__class__.__name__: convparams[i] for i in range(len(ests))}