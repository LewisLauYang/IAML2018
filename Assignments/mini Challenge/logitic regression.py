
import sys
sys.path.append('..')
import os
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC, SVC
from utils import plotter

path = os.path.abspath(os.path.join(os.getcwd(), ".."))

data_path = os.path.join(path, 'datasets', 'Images_C_Train.csv')
train_data = pd.read_csv(data_path,delimiter=',')

X_train = train_data.iloc[:,1:501]

y_train = train_data['is_person']


data_path = os.path.join(path, 'datasets', 'Images_C_Validate.csv')
train_data = pd.read_csv(data_path,delimiter=',')

X_valid = train_data.iloc[:,1:501]

y_valid = train_data['is_person']


data_path = os.path.join(path, 'datasets', 'Images_C_Test.csv')
test_data = pd.read_csv(data_path,delimiter=',')

X_test = test_data.iloc[:,0:500]

y_test = test_data['is_person']

number = 20

loss = {}

Csteps = np.logspace(start=-5, stop=5, num=number)

for C in Csteps:
    lr = LogisticRegression(solver="lbfgs", C=C)
    lr.fit(X=X_train, y=y_train)
    probability = lr.predict_proba(X=X_valid)
    logloss = log_loss(y_true=y_valid, y_pred=probability)
    loss[C] = logloss


#sort the loss,after sorted, loss become a list
#so, need to change to dictionary again
loss = sorted(loss.items(), key=lambda x: x[1])

loss = dict((key, value) for key, value in loss)

bestC = list(loss.keys())[0]

print(bestC)

lr = LogisticRegression(solver="lbfgs", C=bestC)
lr.fit(X=X_train, y=y_train)
print(lr.score(X=X_valid, y=y_valid))





