import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

def get_mean(x):
    x_mean = []
    for i in range(x.shape[0]):
        mean = np.mean(x[i])
        x_mean.append(mean)
    return x_mean


def get_median(x):
    x_median = []
    for i in range(x.shape[0]):
        median = np.median(x[i])
        x_median.append(median)
    return x_median


def get_std(x):
    x_std = []
    for i in range(x.shape[0]):
        std = np.std(x[i])
        x_std.append(std)
    return x_std


def get_var(x):
    x_var = []
    for i in range(x.shape[0]):
        var = np.var(x[i])
        x_var.append(var)
    return x_var


def calculateMSE(y_pred, y, m):
    num = y_pred - y
    num = num * num * m
    mseloss = torch.sum(num) / (33 * 49)
    # print(mseloss.shape)

    return mseloss


def calculateMAE(y_pred, y, m):
    num = torch.abs(y_pred - y) * m
    maeloss = torch.sum(num)/torch.sum(m)

    return maeloss


def fan2(y_pred, y, m):
    num = (y_pred - y) * m
    fan = np.linalg.norm(num.detach().numpy(), ord=2)

    return torch.tensor(fan, requires_grad=True)


def MSE(y_pred, y):
    H, W = y_pred.shape[0], y_pred.shape[1]
    num = y_pred - y
    num = num * num
    mseloss = torch.sum(num) / (H * W)

    return mseloss


def plot_roc_and_auc_score(outputs, labels, title):
    false_positive_rate, true_positive_rate, threshold = roc_curve(labels, outputs)
    auc_score = roc_auc_score(labels, outputs)
    plt.plot(false_positive_rate, true_positive_rate, label = 'ROC curve, AREA = {:.4f}'.format(auc_score))
    plt.plot([0,1], [0,1], 'red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0, 1, 0, 1])
    plt.title(title)
    plt.legend(loc = 'lower right')
    plt.savefig('roc.png')
