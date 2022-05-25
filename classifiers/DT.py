from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

x_train = pd.read_csv("../data/x_train.csv", dtype=np.float64, engine='c',na_filter=False,memory_map=True)
y_train = pd.read_csv("../data/y_train.csv", dtype=np.int, engine='c',na_filter=False, memory_map=True)
x_test = pd.read_csv("../data/x_test.csv", dtype=np.float64, engine='c',na_filter=False,memory_map=True)
y_test = pd.read_csv("../data/y_test.csv", dtype=np.float64, engine='c',na_filter=False, memory_map=True)


x_train = np.array(x_train)
# y_train = np.array(x_train)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)


y_pred = clf.predict(x_test)

print("report:")
print(classification_report(y_test, y_pred, digits=3))



