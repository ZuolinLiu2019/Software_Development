def gini_impurity(Y_c):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.

    Args:
        Y_c (list(int)): Vector of classes given as 0 or 1.

    Returns:
        Floating point number representing the gini impurity.
    """

    Y_set = set(Y_c)
    sum_p2 = 0.
    for y in Y_set:
        p = sum([1. for j in Y_c if j==y])/len(Y_c)
        sum_p2 += p**2
    return 1.0 - sum_p2


def gini_gain(Y_prev, Y_curr):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        Y_prev (list(int)): previous classes, Vector of classes given as 0 or 1.
        Y_curr (list(list(int): current classes, A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    prev_gini = gini_impurity(Y_prev)
    remainder = 0.0
    for curr in Y_curr:
        remainder += gini_impurity(curr) * len(Y_curr) / len(Y_prev)
    return prev_gini - remainder
