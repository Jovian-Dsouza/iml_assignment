import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Function to generate the confusion matrix with percentage entries
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

# Function to compute the performance evaluation metrics
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


###################################################
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil

def plot_decision_regions(X, y, clf,
                    legend = 1,
                    markers='s^oxv<>',
                    colors=('#1f77b4,#ff7f0e,#3ca02c,#d62728,'
                            '#9467bd,#8c564b,#e377c2,'
                            '#7f7f7f,#bcbd22,#17becf'),
                    zoom_factor = 1,
                    hide_spines=True,
                ):
    dim = X.shape[1]
    ax = plt.gca()

    feature_index = (0, 1)
    x_index, y_index = feature_index

    marker_gen = cycle(list(markers))
    n_classes = np.unique(y).shape[0]
    colors = colors.split(',')
    colors_gen = cycle(colors)
    colors = [next(colors_gen) for c in range(n_classes)]
    colors.append(colors[0])

    # Get minimum and maximum
    x_min, x_max = (X[:, x_index].min() - 1./zoom_factor,
                    X[:, x_index].max() + 1./zoom_factor)

    y_min, y_max = (X[:, y_index].min() - 1./zoom_factor,
                    X[:, y_index].max() + 1./zoom_factor)

    xnum, ynum = plt.gcf().dpi * plt.gcf().get_size_inches()
    xnum, ynum = floor(xnum), ceil(ynum)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=xnum),
                         np.linspace(y_min, y_max, num=ynum))

    
    X_grid = np.array([xx.ravel(), yy.ravel()]).T
    X_predict = np.zeros((X_grid.shape[0], dim))
    X_predict[:, x_index] = X_grid[:, 0]
    X_predict[:, y_index] = X_grid[:, 1]

    Z = clf.predict(X_predict.astype(X.dtype))
    Z = Z.reshape(xx.shape)

    cset = ax.contourf(xx, yy, Z,
                       colors=colors,
                       levels=np.arange(Z.max() + 2) - 0.5,
                       alpha=0.45, 
                       antialiased=True,
                       )
    ax.contour(xx, yy, Z, cset.levels,
               colors='k',
               linewidths=0.5,
               antialiased=True)
    ax.axis([xx.min(), xx.max(), yy.min(), yy.max()])

    for idx, c in enumerate(np.unique(y)):
        y_data = X[y == c, y_index]
        x_data = X[y == c, x_index]


        ax.scatter(x=x_data,
                   y=y_data,
                   c=colors[idx+1],
                   marker=next(marker_gen),
                   label=c,
                   alpha=0.8, 
                   edgecolors='black',
                   )
    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    if legend: 
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels,
                    framealpha=0.3, scatterpoints=1, loc=legend)

    return ax

def plot_decision_region_pair_wise(X, y, clf, i, j,
                    legend = 1,
                    markers='s^oxv<>',
                    colors=('#1f77b4,#ff7f0e,#3ca02c,#d62728,'
                            '#9467bd,#8c564b,#e377c2,'
                            '#7f7f7f,#bcbd22,#17becf'),
                    zoom_factor = 1,
                    hide_spines=True,
                ):
    dim = X.shape[1]
    ax = plt.gca()

    feature_index = (0, 1)
    x_index, y_index = feature_index

    marker_gen = cycle(list(markers))
    n_classes = np.unique(y).shape[0]
    colors = colors.split(',')
    colors_gen = cycle(colors)
    colors = [next(colors_gen) for c in range(n_classes)]
    colors.append(colors[0])

    # Get minimum and maximum
    x_min, x_max = (X[:, x_index].min() - 1./zoom_factor,
                    X[:, x_index].max() + 1./zoom_factor)

    y_min, y_max = (X[:, y_index].min() - 1./zoom_factor,
                    X[:, y_index].max() + 1./zoom_factor)

    xnum, ynum = plt.gcf().dpi * plt.gcf().get_size_inches()
    xnum, ynum = floor(xnum), ceil(ynum)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=xnum),
                         np.linspace(y_min, y_max, num=ynum))

    
    X_grid = np.array([xx.ravel(), yy.ravel()]).T
    X_predict = np.zeros((X_grid.shape[0], dim))
    X_predict[:, x_index] = X_grid[:, 0]
    X_predict[:, y_index] = X_grid[:, 1]

    # print(X_predict.shape)
    Z = clf.predict_binary(X_predict.astype(X.dtype), i)
    Z[Z==1] = i
    Z[Z==0] = j

    # print(np.unique(Z))
    Z = Z.reshape(xx.shape)

    color_contour = [colors[c] for c in np.unique(Z)]
    # print(color_contour)
    # print(Z)
    cset = ax.contourf(xx, yy, Z,
                       colors=colors,
                       levels=np.arange(Z.max() + 2) - 0.5,
                       alpha=0.45, 
                       antialiased=True,
                       )
    ax.contour(xx, yy, Z, cset.levels,
               colors='k',
               linewidths=0.5,
               antialiased=True)
    ax.axis([xx.min(), xx.max(), yy.min(), yy.max()])

    for idx, c in enumerate(np.unique(Z)):
        y_data = X[y == c, y_index]
        x_data = X[y == c, x_index]

        # print(c, colors[c])
        ax.scatter(x=x_data,
                   y=y_data,
                   c=colors[c],
                   marker=next(marker_gen),
                   label=c,
                   alpha=0.8, 
                   edgecolors='black',
                   )
    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    if legend: 
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels,
                    framealpha=0.3, scatterpoints=1, loc=legend)

    return ax