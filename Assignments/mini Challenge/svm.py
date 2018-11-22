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
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


path = os.path.abspath(os.path.join(os.getcwd(), ".."))

data_path = os.path.join(path, 'datasets', 'Images_C_Train.csv')
train_data = pd.read_csv(data_path, delimiter=',')

X_train = train_data.iloc[:, 1:501]

y_train = train_data['is_person']

data_path = os.path.join(path, 'datasets', 'Images_C_Validate.csv')
train_data = pd.read_csv(data_path, delimiter=',')

X_valid = train_data.iloc[:, 1:501]

y_valid = train_data['is_person']

data_path = os.path.join(path, 'datasets', 'Images_C_Test.csv')
test_data = pd.read_csv(data_path, delimiter=',')

X_test = test_data.iloc[:, 0:500]

y_test = test_data['is_person']

step_number = 10

# stepArray = np.logspace(start=-4, stop=4, num=step_number)

stepArray = np.linspace(start=20,stop=180,num=7)

gammas = [0.1,0.2, 0.4, 0.6, 0.8, 1.6, 3.2, 6.4, 12.8]

loss = 1.0
bestC = 0.0
bestGamma = 0.0



for C in stepArray:
    for gamma in gammas:
        svc_rbf = SVC(kernel="rbf", gamma=gamma, C=C, probability=True)
        svc_rbf.fit(X=X_train,y = y_train)
        probability = svc_rbf.predict_proba(X=X_valid)
        logloss = log_loss(y_true=y_valid, y_pred=probability)
        print('C:{} gamma:{} logloss:{}'.format(C,gamma,logloss))
        if logloss < loss:
            loss = logloss
            bestC = C
            bestGamma = gamma

print('best C:{} best gamma:{} best loss:{}'.format(bestC,bestGamma,loss))



