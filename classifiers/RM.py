from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize


x_train = pd.read_csv("../data/x_train.csv", dtype=np.float64, engine='c',na_filter=False,memory_map=True)
y_train = pd.read_csv("../data/y_train.csv", dtype=np.int, engine='c',na_filter=False, memory_map=True)
y_train = label_binarize(y_train, classes=[0, 1, 2, 3])

x_test = pd.read_csv("../data/x_test.csv", dtype=np.float64, engine='c',na_filter=False,memory_map=True)
y_test = pd.read_csv("../data/y_test.csv", dtype=np.float64, engine='c',na_filter=False, memory_map=True)
y_test = label_binarize(y_test, classes=[0, 1, 2, 3])


forest = RandomForestClassifier(random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=2)
multi_target_forest.fit(x_train, y_train)
y_pred = multi_target_forest.predict(x_test)
print("report:")
print(classification_report(y_test, y_pred, digits=3))