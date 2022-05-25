"""
The utility functions of ADT

Authors:
    JasonHans
"""

from sklearn.neural_network import MLPClassifier
from model.utils import metrics

class MLP(object):

    def __init__(self, solver='lbfgs', alpha=1e-5, random_state=1):
        '''

        :param solver:
        :param alpha:
        :param hidden_layer_sizes:
        :param random_state:
        '''

        self.classifier = MLPClassifier(solver=solver, alpha=alpha,
                                        random_state=random_state)

    def fit(self, X, y):
        '''

        :param X:
        :param y:
        :return:
        '''

        print('******** Model Fit *********')
        self.classifier.fit(X, y)

    def predict(self, X):
        '''

        :param X:
        :return:
        '''

        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        print(y_pred)
        print(y_true)
        precision, recall, f1 = metrics(y_true, y_pred)
        print('Precision:{:.3f}, recall:{:.3f}, F1:{:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1
