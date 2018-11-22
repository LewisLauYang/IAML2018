# ========== Question 1.1 --- [12 marks] ==========
# We will first get a feel for the data. IMPORTANT: Show all your code!
#
#   (a) [Code] Load the training dataset Images_A_Train.csv into a pandas dataframe, keeping only the Visual Features and the is_person column.
#     Hint: You may wish to first have a look at the column names
#   (b) [Code] Using suitable pandas methods, summarise the key properties of the data, and
#   (c) [Text] comment on your observations from (b) (dimensionality, data ranges, anything out of the ordinary).

import os
import sys
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC, SVC
from utils import plotter

data_path = os.path.join(os.getcwd(), 'datasets', 'Images_A_Train.csv')
image_train = pd.read_csv(data_path,delimiter=',')

image_train_features = image_train.iloc[:,1:501]

image_train_isPerson = image_train['is_person']

print(image_train_features.shape)

print(image_train_features.describe())

print(image_train_isPerson.shape)

print(image_train_isPerson.describe())
#
# print(sum(image_train_features.loc[0]))
# print(sum(image_train_features.loc[1]))
# print(sum(image_train_features.loc[2]))

#person的值只有0，1， 0代表没有人物，1代表有人物
#feature都是小数,


# ========== Question 1.2 --- [8 marks] ==========
# Now we will prepare the testing set in a similar manner.
#
#   (a) [Code] Load the testing dataset Images_A_Test.csv into a pandas dataframe: again extract the Visual Features and the is_person column.
#   (b) [Code] Using similar methods to Q1.1 verify that the testing set is similar to the training set.
#   (c) [Text] Indicate the dimensionality, and comment on any discrepancies if any (if they are similar, just say so).

data_path = os.path.join(os.getcwd(), 'datasets', 'Images_A_Test.csv')
image_test = pd.read_csv(data_path,delimiter=',')

image_test_features = image_test.iloc[:,1:501]

image_test_isPerson = image_test['is_person']


# print(image_test_features.shape)
#
# print(image_test_features.describe())
#
# print(image_test_isPerson.shape)
#
# print(image_test_isPerson.describe())
#
# print(sum(image_test_features.loc[0]))
# print(sum(image_test_features.loc[1]))
# print(sum(image_test_features.loc[2]))


# ========== Question 1.3 --- [5 marks] ==========
# We will now prepare the data for training.
#
#   (a) [Code] Split both the training and testing sets into a matrix of features (independent) variables [X_tr/X_tst] and a vector of prediction (dependent) variables [y_tr/y_tst]. [Optional] As a sanity check, you may wish to verify the dimensionality of the X/y variables.
#   (b) [Code] Using seaborn's countplot function, visualise the distribution of the person-class (True/False) in the training and testing sets (use two figures or sub-plots). Annotate your figures.
#   (c) [Text] Do you envision any problems with the distribution under both sets? Would classification accuracy be a good metric for evaluating the performance of the classifiers? Why or why not?
#

X_tr = image_train_features.values;
X_tst = image_test_features.values;
y_tr = image_train_isPerson.values;
y_tst = image_test_isPerson.values;

print(X_tr.shape)
print(X_tst.shape)
print(y_tr.shape)
print(y_tst.shape)

plt.figure()

plt.subplot(121)

ax1 = sns.countplot(x=y_tr)
ax1.set_title('Train Data')
ax1.set_xticklabels(['is not person','is person'])

plt.subplot(122)

ax2 = sns.countplot(x=y_tst)

ax2.set_title('Test Data')
ax2.set_xticklabels(['is not person','is person'])

plt.show()

#两个的dataset的比例非常接近，不一定，因为准确率，https://www.cnblogs.com/zhizhan/p/4870429.html


# 2. Exploring Different Models for Classification
# ========== Question 2.1 --- [3 marks] ==========
# As always, we wish to start with a very simple baseline classifier, which will provide a sanity check when training more advanced models.
#
#   (a) [Text] Define a baseline classifier (indicate why you chose it/why it is relevant).
#   (b) [Code] Report the accuracy such a classifier would achieve on the testing set.
#
# (a) Your answer goes here:

# because the probability of "is not person" is higher than "is person", so the baseline is to always
# classify the object is "is not person"

isNotPerson_data = image_test_isPerson[image_test_isPerson == 0]


print(isNotPerson_data.count() / image_test_isPerson.count())


# ========== Question 2.2 --- [9 marks] ==========
# Let us now train a more advanced Model.
#
#   (a) [Code] Train a LogisticRegression classifier using default settings, except for the solver parameter which you should set to lbfgs. Report the classification accuracy score on the testing set.
#   (b) [Text] Comment on the performance of the Logistic Regressor in comparison with the baseline model.
#   (c) [Code] Visualise the errors using an appropriate method to justify your answer to (c).
#   (d) [Text] Referring back to the observations in Q1.1, and assuming that we know that the features should be informative, why do you think this may be happening?
lr = LogisticRegression(solver='lbfgs')
lr.fit(image_train_features, image_train_isPerson)
# test_predit = lr.predict(X=image_test_features)
# print('Classification accuracy on test set: {:.3f}'.format(lr.score(image_test_features, image_test_isPerson)))
#
# 和baseline一样
#
# cm = confusion_matrix(image_test_isPerson, test_predit)
#
# print(cm)
#
# plt.figure()
#
# plt.subplot(1,1,1)
#
# labels = ['not person','is person']
#
# sns.heatmap(cm, xticklabels=labels, yticklabels=labels, vmin=0., vmax=1., annot=True)
#
# plt.title('Confusion matrix')
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
#
# plt.show()




# ========== Question 2.3 --- [13 marks] ==========
# You should have noticed that the performance of the above logistic regressor is less than satisfactory. Let us attempt to fix this by preprocessing the inputs X.
#
#   (a) [Text] Before applying the processing, comment on whether you should base any parameters of the preprocessing on the training or testing set or both and what repurcussions this may have.
#   (b) [Code] Following from your observations in Q2.2.(e), process the features in both the training as well as the testing sets accordingly. Hint: There is an sklearn package which may be very useful.
#   (c) [Code] Now Train a Logistic Regressor on the transformed training set, keeping the same settings as in the previous question. Report the classification accuracy on the testing set and visualise the errors in a similar way to Q2.2(d).
#   (d) [Text] Finally comment on the comparative performance with Q2.2.

standardScaler = StandardScaler().fit(image_train_features)
train_x = standardScaler.transform(image_train_features)
test_x = standardScaler.transform(image_test_features)

lr1 = LogisticRegression(solver='lbfgs')
lr1.fit(train_x, image_train_isPerson)
test_predit1 = lr1.predict(X=test_x)
print('Classification accuracy on test set: {:.3f}'.format(lr.score(test_x, image_test_isPerson)))

ca = accuracy_score(image_test_isPerson, test_predit1)

cm = confusion_matrix(image_test_isPerson, test_predit1)

cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]


print(cm)

plt.figure()

plt.subplot(1,1,1)

labels = ['not person','is person']

sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels, vmin=0.0, vmax=1.0, annot=True)

plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()

#



# ========== Question 2.4 --- [18 marks] ==========
# So far we have used default settings for training the logistic regression classifier. Now we want to optimise the hyperparameters of the classifier, namely the regularisation parameter C. We will do this through K-fold cross-validation. You should familiarise yourself with the interpretation of the C parameter.
#
#   (a) [Text] Why do we use cross-validation to optimise the hyper-parameters, rather than using the test-set?
#   (b) [Code] Load the datasets Images_B_Train.csv and Images_B_Test.csv (this ensures everyone is using the same pre-processed data). Again, extract the relevant columns (dim1 through dim500 and the is_person class) from each dataset, and store into X_train/X_test and y_train/y_test variables.
#   (c) [Code] Using Cross-Validation on the Training set (a 5-fold split should be sufficient: set shuffle=True and random_state=0), perform a search for the best value of C in the range 1e-5 to 1e5 (Hint: the KFold split method will come in handy). Keep track of the validation-set accuracy per-fold for each value of C in an array. Think carefully about the best way to cover the search space: i.e. the step-lengths and number of steps.
#   (d) [Code] Plot the mean and standard-deviation (across folds) of the accuracy as a function of C. Hint: you may find the matplotlib's errorbar function useful. Be careful to use the correct scale on the x-axis. Using the mean values, report the regularisation parameter with the best accuracy (alongside its accuracy): N.B. Do not pick the optimal value "by hand", instead use an appropriate numpy function.
#   (e) [Text] Comment on the output, especially as regards the effect of the regularisation parameter (you should write between 3 and 4 sentences).
#   (f) [Code] By using the optimal value (i.e. the one that yields the highest average K-Fold classification accuracy) train a new LogisticRegression classifier and report the classification accuracy on the validation set.
#
# N.B.: Keep track of the KFold object you created as we will keep using it

#交叉验证能够进行多次训练和验证，而使用测试数据只进行一次验证，交叉验证的训练效果更好。



data_path = os.path.join(os.getcwd(), 'datasets', 'Images_B_Train.csv')
train_data = pd.read_csv(data_path,delimiter=',')

X_train = train_data.iloc[:,0:500]

y_train = train_data['is_person']


data_path = os.path.join(os.getcwd(), 'datasets', 'Images_B_Test.csv')
test_data = pd.read_csv(data_path,delimiter=',')

X_test = test_data.iloc[:,0:500]

y_test = test_data['is_person']

number = 20
splits = 5


kf = KFold(n_splits=splits, shuffle=True, random_state=0)

scores = []
mean_array = []
std_array = []

Csteps = np.logspace(start=-5, stop=5, num=number)

print(Csteps)

for kf_train_indexes, kf_test_indexes in kf.split(X_train):

    print('kf processing')
    eachKFScoresDic = []
    for C in Csteps:
        lr = LogisticRegression(solver="lbfgs", C=C)
        lr.fit(X=X_train.loc[kf_train_indexes],y = y_train.loc[kf_train_indexes])
        lr_score = lr.score(X=X_train.loc[kf_test_indexes], y=y_train.loc[kf_test_indexes])
        eachKFScoresDic.append(lr_score)
    scores.append(eachKFScoresDic)



npScores = np.array(scores).reshape(splits,number)

mean_array = np.mean(npScores,axis=0)

std_array = np.std(npScores,axis=0)

# for column in range(npScores.shape[1]):
#     values = npScores[:,column]
#     mean_array.append(np.mean(values))
#     std_array.append(np.std(values))

print(mean_array)
print(std_array)
#
#
plt.figure()

# plt.errorbar(x=np.linspace(start=1e-5, stop=1e5, num=20),y=mean_array,yerr=std_array)

plt.errorbar(x=Csteps, y=mean_array,yerr=std_array, fmt='o', color='black',ecolor='lightgray',elinewidth=3, capsize=0)

plt.xticks(Csteps,rotation=90)

plt.semilogx()

plt.show()

best_accuracy = np.max(mean_array)

best_index = np.argwhere(mean_array == best_accuracy)

bestC = Csteps[best_index]

print(best_accuracy)

print(bestC)

lr = LogisticRegression(solver="lbfgs", C=bestC[0][0])
lr.fit(X=X_train, y=y_train)
lr_score = lr.score(X=X_test, y=y_test)

print('test scorce:',lr_score)



# ========== Question 2.5 --- (LEVEL 11) --- [12 marks] ==========
# Let us attempt to validate the importance of the various features for classification. We could do this like we did for linear regression by looking at the magnitude of the weights. However, in this case, we will use the RandomForestClassifier to give us a ranking over features.
#
#   (a) [Text] How can we use the Random-Forest to get this kind of analysis? Hint: look at the feature_importances property in the SKLearn implementation.
#   (b) [Code] Initialise a random forest classifier and fit the model by using training data only and 500 trees (i.e. n_estimators=500). Set random_state=42 to ensure reproducible results and criterion=entropy but leave all other parameters at their default value. Report the accuracy score on both the training and testing sets.
#   (c) [Text] Comment on the discrepancy between training and testing accuracies.
#   (d) [Code] By using the random forest model display the names of the 10 most important features (in descending order of importance).
#
# RandomForestClassifier
# sorted by the contribution of the features, and sklearn have a property called feature_importances and
# it can output the feature importances array.

rf = RandomForestClassifier(n_estimators=500, random_state=42,criterion="entropy")
rf.fit(X=X_train,y=y_train)
print ("accuracy of training set: ",rf.score(X=X_train, y=y_train))
print ("accuracy of test set: ",rf.score(X=X_test, y=y_test))

# 对于随机森林，重新检验训练数据，并且对于得分是1并不惊讶。对比与Logisitic Regression,随机森林的准确率要稍微高一点，但是并不多

print(rf.feature_importances_)

dataFrame = pd.DataFrame(X_train.columns,columns=['featureName'])
dataFrame['importants'] = rf.feature_importances_

sort_dataFrame = dataFrame.sort_values('importants',ascending=False)

for i in range(10):
    print(sort_dataFrame.iloc[i,0])




# ========== Question 2.6 --- [12 marks] ==========
# We would like now to explore another form of classifier: the Support Vector Machine. A key decision in training SVM's is what kind of kernel to use. We will explore with three kernel types: linear, radial-basis-functions and polynomials. To get a feel for each we will first visualise typical decision boundaries for each of these variants. To do so, we have to simplify our problem to two-dimensional input (to allow us to visualise it).
#
#   (a) [Code] Using the training set only, create a training X matrix with only the dim21 and dim51 columns. N.B. Python (and numpy) use zero-based indexing. Then train three distinct classifiers on this 2D data. Use a linear kernel for one, an rbf kernel for another (set gamma='auto') and a second order (degree) polynomial kernel for the other. Set C=1 in all cases. Using the function plot_SVM_DecisionBoundary from our own library (it exists under the plotters module), plot the decision boundary for all three classifiers.
#   (b) [Text] Explain (intuitively) the shape of the decision boundary for each classifier (i.e. comment on what aspect of the kernel gives rise to it). Use this to comment on how it relates to classification accuracy.
#

X_dim21 = X_train['dim21']
X_dim51 = X_train['dim51']
y_svm = y_train

X_21_and_51 = np.array([X_dim21,X_dim51])
X_21_and_51 = X_21_and_51.T

print(X_21_and_51)


svc_linear = SVC(kernel="linear",C=1).fit(X=X_21_and_51, y=y_svm)
svc_rbf = SVC(kernel="rbf",gamma='auto',C=1).fit(X=X_21_and_51, y=y_svm)
svc_poly = SVC(kernel="poly",degree=2,C=1).fit(X=X_21_and_51, y=y_svm)

print(svc_linear.score(X=X_21_and_51, y=y_svm))
print(svc_rbf.score(X=X_21_and_51, y=y_svm))
print(svc_poly.score(X=X_21_and_51, y=y_svm))

plotter.plot_SVM_DecisionBoundary(clfs=[svc_linear,svc_rbf,svc_poly],X=X_21_and_51,y=y_svm,
                                  title=['linear','rbf','polynomial'],
                                  labels=['dim21','dim51'])
plt.show()


#核函数


# ========== Question 2.7 --- [14 marks] ==========
# Let us now explore the polynomial SVM further. We will go back to using the FULL dataset (i.e. the one we loaded in Question 2.4). There are two parameters we need to tune: the order of the polynomial and the regression coefficient. We will do this by way of a grid-search over parameters. To save computational time, we will use a constrained search space:
#
#   (a) [Code] Define an appropriate search space for C in the range 1e-2 to 1e3 using 6-steps (think about the step-size), and for the degree in the range 1 through 5 inclusive (5 steps). Using the K-fold iterator from Q2.5, optimise the values for C and the degree in the above specified range. Keep track of the mean cross-validation accuracy for each parameter combination.
#   (b) [Code] Using a seaborn heatmap, plot the fold-averaged classification accuracy for each parameter combination (label axes appropriately). Finally also report the combination of the parameters which yielded the best accuracy.
#   (c) [Code] Retrain the (polynomial-kernel) SVC using the optimal parameters found in (b) and report its accuracy on the Testing set.
#   (d) [Text] Explain the results relative to the Logistic Classifier.

step_number = 6
degreeStepNumber = 5

accuracy_array = np.ndarray((degreeStepNumber,step_number))
print(accuracy_array.shape)

stepArray = np.logspace(start=-2, stop=3, num=step_number)

degreeStepArray = np.linspace(start=1, stop=5, num=degreeStepNumber)



C_index = 0
for C in stepArray:
    degree_index = 0
    for degree in degreeStepArray:
        scores = []
        for kf_train_indexes, kf_test_indexes in kf.split(X_train):
            svc_poly = SVC(kernel="poly", degree=degree, C=C)
            svc_poly.fit(X=X_train.loc[kf_train_indexes],y = y_train.loc[kf_train_indexes])
            svc_score = svc_poly.score(X=X_train.loc[kf_test_indexes], y=y_train.loc[kf_test_indexes])
            scores.append(svc_score)
        mean_score = np.mean(scores,axis=0)
        accuracy_array[degree_index,C_index] = mean_score
        degree_index+=1
        print()
    C_index +=1

plt.figure()

sns.heatmap(accuracy_array,xticklabels=stepArray,yticklabels=degreeStepArray,vmin=0.,vmax=1.,annot=True)

plt.xlabel('C')
plt.ylabel('degree')
# plt.semilogx()
plt.show()

svc_poly = SVC(kernel="poly", degree=1, C=1.0)
svc_poly.fit(X=X_train, y=y_train)
svc_score = svc_poly.score(X=X_test, y=y_test)

print("Accuracy of svc_poly:",svc_score)

#########

# Mini challenge


#Logistic Regression


data_path = os.path.join(os.getcwd(), 'datasets', 'Images_C_Train.csv')
train_data = pd.read_csv(data_path,delimiter=',')

X_train = train_data.iloc[:,1:501]

y_train = train_data['is_person']


data_path = os.path.join(os.getcwd(), 'datasets', 'Images_C_Validate.csv')
train_data = pd.read_csv(data_path,delimiter=',')

X_valid = train_data.iloc[:,1:501]

y_valid = train_data['is_person']


data_path = os.path.join(os.getcwd(), 'datasets', 'Images_C_Test.csv')
test_data = pd.read_csv(data_path,delimiter=',')

X_test = test_data.iloc[:,0:500]

y_test = test_data['is_person']



Csteps = np.logspace(start=-5, stop=5, num=20)

scores = []

for C in Csteps:
    print(C)
    lr = LogisticRegression(solver="lbfgs", C=C)
    lr.fit(X=X_train,y = y_train)
    lr_score = lr.score(X=X_valid, y=y_valid)
    scores.append(lr_score)

index = np.argwhere(scores == max(scores))

plt.figure()

plt.semilogx()
plt.plot(Csteps,scores)

plt.show()

print('最大值',scores[index[0][0]])

print('最大值C',Csteps[index[0][0]])











