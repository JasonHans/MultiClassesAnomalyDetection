import sys
from sklearn.model_selection import train_test_split
sys.path.append('../')
from model.MLP import MLP
import pandas as pd
from classifiers.utils import metrics
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import *


if __name__ == '__main__':


    x_train = pd.read_csv("../data/x_train.csv", dtype=np.float64, engine='c', na_filter=False, memory_map=True)
    y_train = pd.read_csv("../data/y_train.csv", dtype=np.int, engine='c', na_filter=False, memory_map=True)
    x_test = pd.read_csv("../data/x_test.csv", dtype=np.float64, engine='c', na_filter=False, memory_map=True)
    y_test = pd.read_csv("../data/y_test.csv", dtype=np.float64, engine='c', na_filter=False, memory_map=True)

    model = MLP()
    model.fit(x_train, y_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)

    y_pred = model.predict(x_test)
    print(y_pred)
    print(y_test)

    # metrics(y_test, y_pred, 4)

    print(classification_report(y_test, y_pred, digits=3))