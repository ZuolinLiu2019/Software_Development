import random

 bagging():
    """A class of Bootstrap Aggregation """
    def __init__(self, learner, kwargs={}, num_bags=20,
                sample_rate=0.8, attribute_rate, verbose=False):
        self.learner = learner
        self.num_bags = num_bags
        self.kwargs = kwargs
        self.sample_rate = sample_rate
        self.attribute_rate = attribute_rate
        self.attribute_indices = []
        self.verbose = verbose

    def fit(self, X, Y):
        """Build models using Bootstrap Aggregation.

            X (list(list(int)): List of features.
            Y (list(int)): labels.
        """
        self.models = []
        num_examples = len(X)
        num_attr = len(X[0])
        for _ in range(self.num_bags):
            sample_idx = random.choices(range(num_examples),
                                        k=int(num_examples*self.sample_rate))
            attribute_idx = random.choices(range(num_attr),
                                        k=int(num_attr*self.attribute_rate))
            model = self.learner(**self.kwargs)
            X_train = [[X[i][j] for j in attribute_idx]for i in sample_idx]
            Y_train = [Y[i] for i in sample_idx]
            model.fit(X_train, Y_train)
            self.models.append(model)
            self.attr_idx.append(attribute_idx)
        return self.models

    def predict(self, X):
        """Predict outcomes using a list of features based on the trained models.

        Args:
            X (list(list(int)): List of features.
        """
        if not isinstance(X[0], list):
            X = [X]
        predictions = []
        for row in features:
            prediction = []
            for i in range(self.num_bags):
                feature = [row[j] for j in self.attribute_indices[i]]
                prediction.append(self.models[i].predict(feature))
            predictions.append(vote(prediction))
        return predictions
