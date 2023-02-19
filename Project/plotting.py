import itertools

import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt

from typing import List, NoReturn


def plot_confusion_matrix(cm: np.ndarray,
                          classes: List[str],
                          normalize: bool = False,
                          title: str = 'Confusion matrix',
                          cmap: matplotlib.colors.LinearSegmentedColormap = plt.cm.Blues) -> NoReturn:
    """ Plots confusion matrix in a readable format.
    :param cm: confusion matrix
    :type cm: numpy array
    :param classes: list of classes to plot as tick labels  
    :type classes: list of str
    :param normalize: if the data is normalize
    :type normalize: boolean
    :param title: title of the figure.
    :type title: str
    :param cmap: colormap of the figure
    :type cmap: matplotlib.colors.LinearSegmentedColormap

    """
    if normalize:
        # Normalize the confusion matrix by row to have correct fractions
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Handle cases where all values in a row are zero to avoid NaN
        cm[np.isnan(cm)] = 0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Format to display the values in the cells of the plot
    fmt = '.2f' if normalize else 'd'
    # Determine threshold for color of text in cell based on value
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
