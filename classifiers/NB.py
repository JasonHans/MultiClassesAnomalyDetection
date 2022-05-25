from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
import pandas as pd
import numpy as np

x_train = pd.read_csv("../data/x_train.csv", dtype=np.float64, engine='c',na_filter=False,memory_map=True)
y_train = pd.read_csv("../data/y_train.csv", dtype=np.int, engine='c',na_filter=False, memory_map=True)
x_test = pd.read_csv("../data/x_test.csv", dtype=np.float64, engine='c',na_filter=False,memory_map=True)
y_test = pd.read_csv("../data/y_test.csv", dtype=np.float64, engine='c',na_filter=False, memory_map=True)

# 2.定义分类器
clf = GaussianNB()

# 3.模型训练
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("report:")
print(classification_report(y_test, y_pred, digits=3))