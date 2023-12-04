import torch
import torch.nn as nn


def mean_absolute_percentage_error(y_true, y_pred):
    #mape = MeanAbsolutePercentageError()
    return torch.mean(torch.abs(y_true - y_pred) / torch.abs(y_true))
    #return mape(y_pred, y_true)


def RMSELoss(y_true, y_pred):
    eps = 1e-6
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(y_true, y_pred) + eps)