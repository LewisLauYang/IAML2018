import os
import sys
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

sys.path.append('..')
from utils.plotter import scatter_jitter, plot_confusion_matrix


pd.set_option('display.width',None)



# ========== Question 2.1 --- [6 marks] ==========
# Load the cleaned datasets train_20news.csv and test_20news.csv into pandas dataframes news_train and news_test
# respectively. Using pandas summary methods, confirm that the data is similar in both sets.

data_path = os.path.join(os.getcwd(), 'datasets', 'train_20news.csv')
news_train = pd.read_csv(data_path,delimiter=',')

data_path = os.path.join(os.getcwd(), 'datasets', 'test_20news.csv')
news_test = pd.read_csv(data_path,delimiter=',')

# print(news_train.describe())
# print(news_train.shape)
# print('#######################')
# print(news_test.describe())
# print(news_test.shape)


# ========== Question 2.2 --- [4 marks] ==========
# [Text] Answer (in brief) the following two questions:
#
# What is the assumption behing the Naive Bayes Model?
# What would be the main issue we would have to face if we didn't make this assumption?

#1,各个特征相互独立


# ========== Question 2.3 --- [8 marks] ==========
# [Code] By using the scatter_jitter function, display a scatter plot of the features w281_ico and w273_tek for the cleaned dataset A. Set the jitter value to an appropriate value for visualisation. Label axes appropriately.
# [Text] What do you observe about these two features? Does this impact the validity of the Naive Bayes assumption? Why or why not?



plt.figure()
plt.subplot(111)

scatter_jitter(news_train['w281_ico'],news_train['w273_tek'])

plt.xlabel('w281_ico')
plt.ylabel('w273_tek')

plt.show()

#有线性相关，非独立


# ========== Question 2.4 --- [7 marks] ==========
# [Text] What is a reasonable baseline against which to compare the classiffication performance?
# Hint: What is the simplest classiffier you can think of?.
# [Code] Estimate the baseline performance on the training data in terms of classification accuracy.

countFrame = news_train.groupby("class").count()

print(countFrame)

print(countFrame.iloc[:,0].max())
print(sum(countFrame.iloc[:,0]))

basline = countFrame.iloc[:,0].max() / sum(countFrame.iloc[:,0])



print(basline)





# ========== Question 2.5 --- [12 marks] ==========
# [Code] Fit a Gaussian Naive Bayes model to the cleaned dataset.
#
# [Code] Report the classification accuracy on the training dataset and plot a Confusion Matrix for the result (labelling the axes appropriately).
#
# [Text] Comment on the performance of the model. Is the accuracy a reasonable metric to use for this dataset?
#
# Hint: You may make use of utility functions we provided, as well as an sklearn method for computing confusion matrices

#
#
X = news_train.drop('class',axis=1)

y = news_train['class']



gnb = GaussianNB()

gnb.fit(X=X,y=y)

tr_pred = gnb.predict(X=X)

print(tr_pred)
#
ca = accuracy_score(y, tr_pred)

cm = confusion_matrix(y, tr_pred)

cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]

plt.figure()

plt.subplot(1,1,1)

labels = ['alt.atheism','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.sport.baseball','rec.sport.hockey']

sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels, vmin=0., vmax=1., annot=True)

plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()

print('accuracy',ca)

print(cm)


# ========== Question 2.6 --- [3 marks] ==========
# [Text] Comment on the confusion matrix from the previous question. Does it look like what you would have expected? Explain.











# ========== Question 2.7 --- [12 marks] ==========
# Now we want to evaluate the generalisation of the classifier on new (i.e. unseen data).
#
# [Code] Use the classifier you trained in Question 2.5 (i.e. on the cleaned dataset) and test its performance on the test dataset.

# Display classification accuracy and plot a confusion matrix of the performance on the test data.
#
# [Code] Also, reevaluate the performance of the baseline on the test data.
#
# [Text] In a short paragraph (3-4 sentences) compare and comment on the results with (a) the training data and (b) the baseline (on the test data).




# labels = ['alt.atheism','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.sport.baseball','rec.sport.hockey']
#
# X = news_train.drop('class',axis=1)
#
# y = news_train['class']
#
testX = news_test.drop('class',axis=1)

testy = news_test['class']

gnb = GaussianNB()

gnb.fit(X=X,y=y)

tr_pred = gnb.predict(X=testX)

print(tr_pred)

ca = accuracy_score(testy, tr_pred)

cm = confusion_matrix(testy, tr_pred)

cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]

plt.figure()

plt.subplot(1,1,1)

sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels, vmin=0., vmax=1., annot=True)

plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()

print('accuracy',ca)

print(cm)



###################

trainFrame = news_train.groupby("class").count()

column = trainFrame.iloc[:,0]

maxIndex = column[column == column.max()].index.values[0]

testFrame = news_test.groupby("class").count()

testMax = testFrame.iloc[4,0]

testBaseLine = testMax / sum(testFrame.iloc[:,0])

print(testBaseLine)

#
# print(countFrame.iloc[:,0].max())
# print(sum(countFrame.iloc[:,0]))
#
# basline = countFrame.iloc[:,0].max() / sum(countFrame.iloc[:,0])




# ========== Question 2.8 --- (LEVEL 11) --- [7 marks] ==========
# [Code] Fit a Gaussian Naive Bayes model to the original raw dataset (including the outliers)
# and test its performance on the test set.
#
# [Text] Comment on the output and explain why or why not cleaning affects the classifier.



data_path = os.path.join(os.getcwd(), 'datasets', 'raw_20news.csv')
news_raws = pd.read_csv(data_path,delimiter=',')

X = news_raws.drop('class',axis=1)

y = news_raws['class']

testX = news_test.drop('class',axis=1)

testy = news_test['class']

gnb = GaussianNB()

gnb.fit(X=X,y=y)

tr_pred = gnb.predict(X=testX)

cm = confusion_matrix(testy, tr_pred)

cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]

plt.figure()

plt.subplot(1,1,1)

labels = ['alt.atheism','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.sport.baseball','rec.sport.hockey']

sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels, vmin=0., vmax=1., annot=True)

plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()





