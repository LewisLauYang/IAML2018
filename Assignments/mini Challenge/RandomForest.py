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


loss = {}

estimatorStep = np.linspace(start=100,stop=1000,num=20,dtype=np.int64)

for estimators in estimatorStep:
    print('estimators: ',estimators)
    rf = RandomForestClassifier(n_estimators=estimators, random_state=42, oob_score=True)
    rf.fit(X=X_train, y=y_train)
    probability = rf.predict_proba(X=X_valid)
    logloss = log_loss(y_true=y_valid, y_pred=probability)
    loss[estimators] = logloss

loss = sorted(loss.items(), key=lambda x: x[1])

loss = dict((key, value) for key, value in loss)

bestEstimator = list(loss.keys())[0]
bestLoss = list(loss.values())[0]

print('best n_estimators:{} best loss:{}'.format(bestEstimator,bestLoss))



{'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}

maxDepthStep = np.linspace(start=3,stop=14,num=4)
minSamplesSplit = np.linspace(start=2,stop=50,num=4)

for max_depth in maxDepthStep:
    for min_Samples_Split in minSamplesSplit:
        rf = RandomForestClassifier(n_estimators=526, random_state=42, oob_score=True)
        rf.fit(X=X_train, y=y_train)
        probability = rf.predict_proba(X=X_valid)
        logloss = log_loss(y_true=y_valid, y_pred=probability)
        print('max_depth:{}  min_Samples_Split:{} logloss:{}'.format(max_depth,min_Samples_Split,logloss))


# rf = RandomForestClassifier(n_estimators=bestEstimator, random_state=42, criterion="entropy")
# rf.fit(X=X_train, y=y_train)
# print(rf.predict_proba(X=X_valid))


