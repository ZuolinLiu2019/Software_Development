# author: Zuolin Liu

from .utils.load_data import *
from decision_metrics import *
import random


class node:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, label=None):
        """Create a decision function to select between left and right nodes.

        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.

        Args:
            left (node): left child node.
            right (node): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.label = label

    def decide(self, x):
        """Get a child node based on the decision function.

        Args:
            x (list(int)): vector for feature.

        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.label:
            return self.label
        elif self.decision_function(x):
            return self.left.decide(x)
        else:
            return self.right.decide(x)

class decision_tree():
    """Class for tree-building and classification."""

    def __init__(self, depth_limit=float("inf")):
        """Create a decision tree with a set depth limit.

        Starts with an empty root.

        Args:
            depth_limit (float): The maximum depth to build the tree.
        """
        self.root = None
        self.depth_limit = depth_limit

    def fit(self, X, Y, split_by="random"):
        """Build the tree from root using __build_tree__().

        Args:
            X (list(list(int)): List of features.
            Y (list(int)): Available classes.
        """
        self.root = self.__build_tree__(X, Y)

    def predict(self, X):
        """Use the fitted tree to classify a list of example features.

        Args:
            X (list(list(int)): List of features.

        Return:
            A list of class labels.
        """
        if not isinstance(X[0], list):
            X = [X]
        Y_p = []
        for row in X:
            Y_p.append(self.root.decide(row))
        return Y_p

    def __build_tree__(self, X, Y, depth=0, split_by="random"):
        """Build tree that automatically finds the decision functions.

        Args:
            features (list(list(int)): List of features.
            classes (list(int)): Available classes.
            depth (int): max depth of tree.  Default is 0.

        Returns:
            Root node of decision tree.
        """

        if len(set(Y)) == 1:
            return node(None, None, None, label=Y[0])
        elif depth >= self.depth_limit or len(X) == 0:
            return node(None, None, None, label=vote(Y))

        feature_split = self.__determine_split_feature__(X, Y, split_by=split_by)
        feature = [row[feature_split] for row in X]
        medium_v = medium(feature)
        X_left, X_right = self.__split_features__(X, feature, medium_v)
        Y_left, Y_right = self.__split_class__(Y, feature, medium_v)

        decision_function = lambda feature:feature[feature_split] <= medium_v

        return node(self.__build_tree__(X_left, Y_left, depth+1),
                    self.__build_tree__(X_right, Y_right, depth+1),
                    decision function)

    def __split_class__(self, Y, feature, medium_v):
        Y_left = [Y[i] for i in range(len(Y)) if feature[i] <= medium_v]
        Y_right = [Y[i] for i in range(len(Y)) if feature[i] > medium_v]
        return Y_left, Y_right

    def __split_features__(self, X, feature, medium_v):
        X_left = [X[i] for i in range(len(X)) if feature[i] <= medium_v]
        X_right = [X[i] for i in range(len(X)) if feature[i] > medium_v]
        return X_left, X_right

    def __determine_split_feature__(self, X, Y, split_by="gini"):
        if split_by == "gini":
            return self.split_by_gini__(X, Y)
        elif split_by == "random":
            return self.__split_by_random__(X, Y)
        else:
            return self.__split_by_corr__(X, Y)

    def __split_by_gini__(self, X, Y):
        gini_gains = []
        for i in range(len(X[0])):
            feature = [row[i] for row in X]
            medium_v = medium(feature)
            Y_curr = self.split_class__(Y, feature, medium_v)
            gini.gains.append(gini_gain(Y, Y_cur))
        max_gini_gains = max(gini_gains)
        for i in range(len(gini_gains)):
            if gini_gains[i] == max_gini_gains:
                return i

    def __split_by_corr__(self, X, Y):
        raise NotImplemented()

    def __split_by_random__(self, X, Y):
        return random.choice(range(len(X[0])))
