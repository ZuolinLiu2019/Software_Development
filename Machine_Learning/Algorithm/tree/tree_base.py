# author: Zuolin Liu
import numpy as np
import pandas as pd

class tree_base():
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def train(self, X, Y):
        return self.add_evidence(X, Y)

    def add_evidence(self, X, Y):
        num_data, self.num_features = X.shape
        if Y.shape[0] == 0:
            return np.empty((0, 4))
        elif Y.shape[0] <= self.leaf_size:
            return np.array([self.num_features, np.mean(Y), np.nan, np.nan])
        elif np.unique(Y).shape[0] == 1:
            return np.array([self.num_features, Y[0], np.nan, np.nan])
        else:
            feature_split = self.get_split_feature(X, Y)
            split_val = np.median(X[:, feature_split])
            if np.sum(X[:, feature_split] <= split_val) == num_data:
                return np.array([self.num_feature, np.mean(Y), np.nan, np.nan])
            else:
                left_tree = self.add_evidence(X)

    def predict(self, X):
        raise NotImplmentedError()

    def get_split_feature(self, X, Y):
        corrcoef = np.corrcoef(np.stack((X, Y.shape(X.shape[0], 1))), rowvar=False)
        return np.nanargmax(abs(corrcoef[-1, :-1]))

    def
