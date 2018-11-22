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
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB




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

loss = 1.0
bestAlpha = 0

alphas = np.linspace(start=0.1, stop=2.0, num=15)

for alpha in alphas:
    nb = MultinomialNB(alpha = alpha)
    nb.fit(X=X_train,y=y_train)
    probability = nb.predict_proba(X=X_valid)
    logloss = log_loss(y_true=y_valid, y_pred=probability)
    print('alpha:{} logloss:{}'.format(alpha, logloss))
    if logloss < loss:
        loss = logloss
        bestAlpha = alpha

print('best alpha:{} best loss:{}'.format(bestAlpha, loss))







