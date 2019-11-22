# this file defines some functions to assess the performance of the model
# using predicted label and true label

def true_positive(Y_p, Y):
    return sum([1. for i in range(len(Y)) if Y_p[i] == 1 and Y[i] == 1])

def false_negative(Y_p, Y):
    return sum([1. for i in range(len(Y)) if Y_p[i] == 0 and Y[i] == 1])

def false_positive(Y_p, Y):
    return sum([1. for i in range(len(Y)) if Y_p[i] == 1 and Y[i] == 0])

def true_negative(Y_p, Y):
    return sum([1. for i in range(len(Y)) if Y_p[0] == 0 and Y[i] == 0])

def confusion_matrix(Y_p, Y):
    """Create a confusion matrix to measure classifier performance.

    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]

    Args:
        Y_p (list(int)): predictions.
        Y (list(int): true_labels.

    Returns:
        A two dimensional array representing the confusion matrix.
    """
    return [[true_positive(), false_negative()], [false_positive(), true_negative()]]

def precision(Y_p, Y):
    """Get the precision of a classifier compared to the correct values.

    Precision is measured as:
        true_positive/ (true_positive + false_positive)

    Args:
        Y_p (list(int)): predictions.
        Y (list(int): true_labels.

    Returns:
        The precision of the classifier output.
    """
    true_positive = true_positive()
    return true_positive/(true_positive + false_positive())

def recall(Y_p, Y)):
    """Get the recall of a classifier compared to the correct values.

    Recall is measured as:
        true_positive/ (true_positive + false_negative)

    Args:
        Y_p (list(int)): predictions.
        Y (list(int): true_labels.

    Returns:
        The recall of the classifier output.
    """
    true_positive = true_positive()
    return true_positive/(true_positive + false_negative())

def accuracy(Y_p, Y):
    """Get the accuracy of a classifier compared to the correct values.

    Accuracy is measured as:
        correct_classifications / total_number_examples

    Args:
        Y_p (list(int)): predictions.
        Y (list(int): true_labels.

    Returns:
        The accuracy of the classifier output.
    """
    correct_predictions = sum([1. for i in range(len(Y)) if Y_p[0] == Y[i]])
    return correct_predictions/len(Y)
