#########################################################################################
# LogisticRegression(random_state=0, solver='sag',multi_class='ovr', verbose = 1)
#########################################################################################

from sklearn.linear_model import LogisticRegression
import pandas as pd
from datetime import datetime
from sklearn.metrics import *
import numpy as np


x_train = pd.read_csv("../data/x_train.csv", dtype=np.float64, engine='c',na_filter=False,memory_map=True)
y_train = pd.read_csv("../data/y_train.csv", dtype=np.int, engine='c',na_filter=False, memory_map=True)
x_test = pd.read_csv("../data/x_test.csv", dtype=np.float64, engine='c',na_filter=False,memory_map=True)
y_test = pd.read_csv("../data/y_test.csv", dtype=np.float64, engine='c',na_filter=False, memory_map=True)


# train
btime = datetime.now()
lr_clf = LogisticRegression(random_state=0, solver='sag',multi_class='ovr', verbose = 1) # l2
lr_clf.fit(x_train, y_train)
print('all tasks done. total time used:%s s.\n\n'%((datetime.now() - btime).total_seconds()))

y_pred = lr_clf.predict(x_test)

print(classification_report(y_test, y_pred , digits=3))