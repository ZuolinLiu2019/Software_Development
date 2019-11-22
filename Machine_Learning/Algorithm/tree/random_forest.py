from .bagging.bagging import bagging
from decision_tree import decision_tree

class random_forest():
    """Random forest class"""
    def __init__(self, num_trees, depth_limit, sample_rate, attribute_rate):
        """Create a random forest.

         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             sample_rate (float): percentage of example samples.
             attr_rate (float): percentage of attribute samples.
        """
        kwargs = {"depth_limit":depth_limit}
        self.model = bagging(decision_tree, kwargs=kwargs, num_bags=num_trees,
                    sample_rate=sample_rate, attribute_rate=attribute_rate)

    def fit(self, X, Y):
        """Build a random forest of decision trees using Bootstrap Aggregation.

        Args:
            X (list(list(int)): List of features.
            Y (list(int)): Available classes.
        """
        return self.model.fit(X, Y)


    def predict(self, X):
        """Classify a list of features based on the trained random forest.

        Args:
            X (list(list(int)): List of features.
        """
        return self.model.predict(X)
