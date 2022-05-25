import sys
sys.path.append("../")
import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier



if __name__ == "__main__":
    x_train = pd.read_csv("../data/x_train.csv", dtype=np.float64, engine='c', na_filter=False, memory_map=True)
    y_train = pd.read_csv("../data/y_train.csv", dtype=np.int, engine='c', na_filter=False, memory_map=True)
    y_train = label_binarize(y_train, classes=[0, 1, 2, 3])

    x_test = pd.read_csv("../data/x_test.csv", dtype=np.float64, engine='c', na_filter=False, memory_map=True)
    y_test = pd.read_csv("../data/y_test.csv", dtype=np.float64, engine='c', na_filter=False, memory_map=True)
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3])



    model = OneVsRestClassifier(svm.LinearSVC(random_state = 0, verbose = 1))

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("report:")
    print(classification_report(y_test, y_pred, digits=3))
    '''             precision    recall  f1-score   support
    
              0       0.78      0.85      0.81     42276
              1       0.83      0.66      0.74     18960
              2       0.59      0.34      0.44     13591
              3       0.59      0.35      0.44     13170
              4       0.00      0.00      0.00      8151

    avg / total       0.67      0.60      0.62     96148
    '''