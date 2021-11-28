import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def confusion_matrix(Y_true, Y_pred):
    cm = metrics.confusion_matrix(Y_true, Y_pred)
    percent = np.array(cm, np.float32)
    for i in range(cm.shape[0]):
        percent[i, :] = percent[i, :] / percent.sum(axis=1)[i]

    categories = [i for i in range(1, cm.shape[0]+1)]
    group_percentages = ["{0:.2%}\n".format(value) for value in percent.flatten()]
    group_counts = ["({0:0.0f})".format(value) for value in cm.flatten()]
    box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_percentages, group_counts)]
    box_labels = np.asarray(box_labels).reshape(cm.shape[0],cm.shape[1])
    sns.heatmap(cm, annot=box_labels, fmt='', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def metrics_summary(Y_true, Y_pred):
    accuracy = accuracy_score(Y_true, Y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(Y_true, Y_pred)

    print(f"Mean Accuracy: %0.2f" % (accuracy))
    print(f"Mean Precision: %0.2f" % (precision.mean()))
    print(f"Mean Recall: %0.2f" % (recall.mean()))
    print(f"Mean F-measure: %0.2f" % (f1.mean()))
    print()
    for i, score in enumerate(recall):
        print(f"Recall for class {i+1}: %0.2f" % (score))
    print()
    for i, score in enumerate(f1):
        print(f"F-measure for class {i+1}: %0.2f" % (score))