"""
The utility functions of ADT

Authors:
    JasonHans
"""

from sklearn.metrics import precision_recall_fscore_support

def metrics(y_true, y_pred):
    '''
    estimate metrics for precision, recall and f1.

    :param y_true: ndarray, the ground truth label list
    :param y_pred: ndarray, the predicted result list
    :return: precision(float), recall(float), f1(float)
    '''

    # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    return precision, recall, f1