import random

def generate_k_folds(dataset, k):
    """Split dataset into folds.

    Randomly split data into k equal subsets.

    Fold is a tuple (training_set, test_set).
    Set is a tuple (examples, classes).

    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.

    Returns:
        List of folds.
    """
    folds = []
    n = len(dataset[0])
    indicies = list(range(n))
    for _ in range(k):
        random.shuffle(indicies)
        test_indicies = indicies[:n//k]
        train_indicies = indicies[n//k:]
        X_test = [dataset[0][i] for i in test_indicies]
        Y_test = [dataset[1][i] for i in test_indicies]
        X_train = [dataset[0][i] for i in train_indicies]
        Y_train = [dataset[1][i] for i in train_indicies]
        folds.append(((X_train, Y_train), (X_test, Y_test)))
    return folds
