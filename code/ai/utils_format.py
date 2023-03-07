import numpy as np
import pandas as pd

NUM_DIMS_VECTOR = 1
NUM_DIMS_MATRIX = 2


def check_data(data):
    # check type
    if not isinstance(data, (np.ndarray, pd.DataFrame)):
        raise TypeError

    # check dimensions
    if data.ndim != NUM_DIMS_MATRIX:
        raise ValueError

    # convert type to Pandas
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    return data


def check_num_values(num_values):
    num_values = int(num_values)
    if num_values <= 1:
        raise ValueError
    return num_values


def check_values(values, num_values=None):  # regression
    # check type
    if not isinstance(values, (np.ndarray, pd.Series)):
        raise TypeError

    # check dimensions
    if values.ndim != NUM_DIMS_VECTOR:
        raise ValueError
    if num_values is not None:
        num_values = check_num_values(num_values)
        if isinstance(values, np.ndarray):
            values_unique = np.unique(values[~np.isnan(values)])
        else:  # Pandas series
            values_unique = values.dropna().unique()

        if values_unique.size != num_values:
            # will raise if the values do not contain all possible ones
            raise ValueError

    # convert type
    if isinstance(values, np.ndarray):
        values = pd.Series(values)
    return values


def check_num_classes(num_classes):
    num_classes = int(num_classes)
    if num_classes <= 1:
        raise ValueError
    return num_classes


def check_classes(classes, num_classes=None):  # classification
    # check type
    if not isinstance(classes, (np.ndarray, pd.Series)):
        raise TypeError

    # check dimensions
    if classes.ndim != NUM_DIMS_VECTOR:
        raise ValueError
    if num_classes is not None:
        num_classes = check_num_classes(num_classes)
        if isinstance(classes, np.ndarray):
            classes_unique = np.unique(classes[~np.isnan(classes)])
        else:  # Pandas series
            classes_unique = classes.dropna().unique()

        if classes_unique.size != num_classes:
            # will raise if the classes do not contain all possible ones
            raise ValueError

    # convert type to Pandas
    if isinstance(classes, np.ndarray):
        classes = pd.Series(classes)
    return classes


def check_labels(labels, num_classes=None):
    # check type
    if not isinstance(labels, (np.ndarray, pd.Series)):
        raise TypeError

    # check dimensions
    if labels.ndim != NUM_DIMS_VECTOR:
        raise ValueError
    if num_classes is not None:
        num_classes = check_num_classes(num_classes)
        if labels.size != num_classes:
            raise ValueError

    # convert type to Pandas
    if isinstance(labels, np.ndarray):
        labels = pd.Series(labels)
    return labels


def check_regress_problem(data, values, num_values=None):
    # check each
    data = check_data(data)
    values = check_values(values, num_values)

    # check agreement of dimensions
    num_samples, num_features = data.shape
    if values.size != num_samples:
        raise ValueError

    return data, values


def check_classif_problem(data, classes, labels, num_classes=None):
    # check each
    data = check_data(data)
    classes = check_classes(classes, num_classes=num_classes)
    labels = check_labels(labels, num_classes=num_classes)

    # check agreement of dimensions
    num_samples, num_features = data.shape
    if classes.size != num_samples:
        raise ValueError

    # check agreement of classes and labels
    set_classes = set(classes.dropna().unique().tolist())
    set_labels = set(labels.tolist())
    if set_labels != set_classes:
        raise ValueError

    return data, classes, labels


