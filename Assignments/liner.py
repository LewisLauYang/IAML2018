# Nice Formatting within Jupyter Notebook

from IPython.display import display

# Allows multiple displays from a single code-cell

# System functionality
import sys
sys.path.append('..')

# Import Here any Additional modules you use. To import utilities we provide, use something like:
#   from utils.plotter import plot_hinton

# Your Code goes here:

from utils.plotter import plot_hinton

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
pd.set_option('display.width',None)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from math import sqrt, log

# ========== Question 1.1 --- [8 marks] ==========
# Load the dataset train_auto_numeric.csv into a pandas DataFrame called auto_numeric. Using any suitable pandas functionality,
#
# [Code] summarise and
# [Text] comment upon
# the key features of the data. Show all your code!

data_path = os.path.join(os.getcwd(),'datasets','train_auto_numeric.csv')
auto_numeric = pd.read_csv(data_path,delimiter=',')
print(auto_numeric.describe())
print(auto_numeric.columns.values)
print(auto_numeric.head())


# ========== Question 1.2 --- [18 marks] ==========
# We will now examine the attributes in some detail. Familiarise yourself with the concept of Correlation Coefficients (start from the Lecture on Generalisation and Evaluation).
#
# [Code] Analyse first the relationship between each attribute and price:
# Compute the correlation coefficient between each attribute and price, and
# Visualise the (pairwise) distribution of each attribute with price
# [Text] Given the above, which attributes do you feel may be most useful in predicting the price? (mention at least 5). How did you reach this conclusion? Hint: which is the more useful of the above tools?
# [Code] Now we will analyse the relationship between the attributes themselves. Use an appropriate pairwise visualisation tool to display graphically the relationship between each pair of attributes you selected in (2).
# [Text] Do any attributes exhibit significant correlations between one-another? (restrict your analysis to useful attributes identified above)
# [Text] Which attributes (give examples) would you consider removing if we wish to reduce the dimensionality of the problem and why?

#ignore pirce column

for column in auto_numeric.columns:
    if column == 'price':
        break
    correlation = auto_numeric.loc[:,column].corr(auto_numeric.loc[:,'price'])

    print('the correlation between {} and price is {}'.format(column,correlation))


# pltColumn = 3
# pltRow = (len(auto_numeric.columns) - 1) / pltColumn
#
# plt.figure(figsize=(pltColumn * 6.4, pltRow * 9.8))
#
#
# sns.pairplot(auto_numeric, x_vars=auto_numeric.columns[0:-1], y_vars='price')

# for i in range(len(auto_numeric.columns) - 1):
#     ax = plt.subplot(pltRow, pltColumn, i + 1)
#     columnName = auto_numeric.columns[i]
#     sns.pairplot(auto_numeric, x_vars=['normalized-losses','wheel-base'], y_vars='price')


# plt.show()

#wheel-base, length, width,height,city-mpg,

# attributes = ['wheel-base','length','width','engine-size','highway-mpg']
#
# plt.figure()
#
# sns.pairplot(auto_numeric, x_vars=attributes, y_vars=attributes)
#
# plt.show()



#5,normalized-losses, height,stroke ,compression-ratio,peak-rpm,mean-effective-pressure,torque

#correlation is low 0.15

#
# 2. Simple Linear Regression
# When applying machine learning in practice it can be prudent to start out simple in order to get a feeling for the dataset and for any potential difficulties that might warrant a more sophisticated model.
# We will thus begin by studying a simple Linear Regression model. Such a model will consider the relationship between a dependent (response) variable and only one independent (explanatory) variable, which we take to be the engine-power.
#
# ========== Question 2.1 --- [5 marks] ==========
# [Code] Produce a scatter plot of price against engine-power (label the axis).
# [Text] What are your thoughts about the ability of the variable to predict the price?

# plt.figure()
# plt.subplot(1,1,1)
# sns.scatterplot(x='engine-power',y='price',data=auto_numeric)
# plt.xlabel('engine-power')
# plt.ylabel('price')
# plt.show()


# ========== Question 2.2 --- [8 marks] ==========
# [Code] Now visualise the distribution of the car price (again label the axes). Choose a sensible value for the number of bins in the histogram.
# [Text] Comment on why the price variable may not be easy to model using linear regression, and suggest possible preprocessing to improve its applicability.
# At the same time, explain why it is not conclusive that it is the case at this stage.
# N.B. There is no need to carry out the preprocessing at this stage, just comments


# plt.hist(auto_numeric.iloc[:,-1], bins=20)
# plt.xlabel("Price")
# plt.ylabel("Number of Occurences")
# plt.show()




# ========== Question 2.3 --- [3 marks] ==========
# We want to prepare our dataset for training/testing. Extract the dependent variable into a vector and the independent attribute into
# another. Split the dataset with 80% for training and the remaining 20% for testing, naming the resulting arrays X_train, X_test, y_train and y_test.
#
# Hint: you may use Scikit's train_test_split: set the random state to 0 for reproducibility.
#
# N.B. For technical reasons, X_train/X_test must be 2D arrays: extend the dimensions of the independent attribute before splitting the dataset,
# such that the shape of the resulting array is (n,1) where n is the number of instances in the dataset.


X = auto_numeric['engine-power']
y = auto_numeric['price']

X = np.reshape(X.values, (X.shape[0],1))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, test_size=.2,random_state=0)



# ========== Question 2.4 --- [4 marks] ==========
# Decide on a simple baseline to predict the price variable. Implement it and display its parameter.
#
# Hint: This should be just 1 line of code + a print/display

# print('the baseline is:',y_train.mean())



# ========== Question 2.5 --- [7 marks] ==========
# Now we want to build a simple linear regression model. We will use Scikit-learn's LinearRegression class.
#
# [Code] Train a LinearRegression model and report its parameters: N.B. Here we mean the weights of the Regression Function.
# [Text] Interpret the result, and comment on what impact this has if any on the relevance of the engine-power attribute to predict the price.

# lm = LinearRegression()
# a = lm.fit(X_train,y_train)
# print(lm.get_params())


# ========== Question 2.6 --- [9 marks] ==========
# Now we will evaluate and compare the performance of our models on the testing data.
#
# [Code] Produce a scatter plot of the test-data price data-points. Add the regression line to the plot and
# show the predictions on the testing set by using a different marker. Finally plot also the baseline predictor (same figure). Label your axes and provide a legend.
# [Text] Just by looking at this plot, how do the two models compare?

# print(X_test)
# print(y_test)
# plt.figure()
# plt.subplot(1,1,1)
# plt.scatter(X_test, y_test, label="Test Data")
# testY_pred = lm.predict(X=X_test)
# plt.scatter(X_test,testY_pred, label="Test predition Data")
# plt.plot(X_test,testY_pred,label="Test Regression Line",color='yellow')
# plt.xlabel("engine-power")
# plt.ylabel("price")
# plt.legend()
# plt.show()



# ========== Question 2.7 --- [20 marks] ==========
# You might have noticed that the above plot is not easy to interpret.
#
# [Code] Generate another plot, this time showing a histogram of the residuals under both models (label everything).
# [Code] Report also the Coefficient of Determination ($R^2$) and Root Mean Squared Error (RMSE) on the same hold-out testing set for both predictors.
#     Hint: Scikit Learn has functions to help in evaluating both measures.
# [Text] Comment on the result. Hint: In your answer, you should discuss what the graph is showing and
# what the two values are measuring, and finally compare the two models under all measures/plots.

# plt.hist(y_test - testY_pred,bins=20)
# plt.xlabel('residuals')
# plt.ylabel('Number of Occurences of residuals')
# plt.title('residuals of engine-power')
# plt.show()
#
# r2_score = r2_score(y_test,testY_pred)
# print('r2 score is',r2_score)
#
# mse = mean_squared_error(y_test,testY_pred)
# print('RMSE value is',mse)



# ========== Question 2.8 --- [9 marks] ==========
# So far we have used a hold-out test set for validation.
#
# [Text] What are the repurcussions of this for interpreting the above results?
#
# [Code] To solve this problem, we will use k-fold cross-validation to evaluate the performance of the regression model.
# By using Scikit-learn's KFold class construct a 5-fold cross-validation object. ' \
# 'Set shuffle=True and random_state=0. [Optional] You may wish to visualise the training/validation indices per fold. The split method comes in handy in this case.
#
# N.B. You will use this KFold instance you are about to create throughout most of the remainder of this Assignment - keep track of it!
#
# [Code] Then train a new Linear Regression Model using the cross_val_predict function. Report the Coefficient of Determination ($R^2$) and Root Mean Squared Error (RMSE).
#
# [Text] Relate these to the previous results.


kf = KFold(159, shuffle=True, random_state=0)
#
# lm2 = LinearRegression()
# y_pred2 = cross_val_predict(lm2, X_train, y=y_train, cv=kf)
#
#
#
# r2_score = r2_score(y_test,testY_pred)
# print('r2 score is',r2_score)
#
# mse = mean_squared_error(y_test,testY_pred)
# print('RMSE value is',mse)


# ========== Question 2.9 --- (LEVEL 11) --- [18 marks] ==========
# [Code] Load the new dataset train_auto_base.csv into a pandas DataFrame auto_base.
# Again by using the engine-power attribute as predictor and price as target variable build a LinearRegression model on this dataset. Report the $R^2$ and RMSE metrics for this model (on testing set).
#
# [Code/Text] You should notice a significant change in performance. Where is this coming from?
# Use visualisation/analysis methods you have learnt to answer this question. Document your code and describe your analysis (via inline comments) as you progress.
# Your written answer should be just a short paragraph (1-3 sentences) describing your conclusion.
#
# Hint: you may find it easier to understand what is happening if you use a hold-out test-set rather than cross-validation in this case. Also, make use of pandas methods to help you.


# data_path = os.path.join(os.getcwd(),'datasets','train_auto_base.csv')
# auto_base = pd.read_csv(data_path,delimiter=',')
#
#
# X = auto_base['engine-power']
# y = auto_base['price']
#
# X = np.reshape(X.values, (X.shape[0],1))
#
# lr = LinearRegression()
#
# lr.fit(X=X,y=y)
#
# print(X_test)
#
# preditionY = lr.predict(X=X_test)
#
# print('preditionY',preditionY)
#
# r2Score = r2_score(y_test,preditionY)
# print('r2 score is',r2Score)
#
# mse = mean_squared_error(y_test,preditionY)
# print('RMSE value is',mse)






# 3. Multivariate Linear Regression
# In this Section we will fit a Multivariate Linear Regression model (still using LinearRegression) to the dataset: i.e.
#
# we will now train a model with multiple explanatory variables and ascertain how they affect our ability to predict the retail price of a car.
#
# N.B. We will use the KFold instance you created in Question 2.8 to train & validate our models.
#
# ========== Question 3.1 --- [6 marks] ==========
# [Code] Train a Multi-Variate LinearRegression model on the original auto_numeric dataframe you loaded in Question 1.1,
# and evaluate it using the KFold instance you created in Question 2.8 (report RMSE and $R^2$).
# [Text] Comment on the result, and compare with the univariate linear regression model we trained previously (Question 2.5).


# lm = LinearRegression(normalize=True)
# X = auto_numeric.drop(['price'], axis=1)
# y = auto_numeric['price']
# y_pred = cross_val_predict(lm, X, y=y, cv=kf)
#
# print(y_pred)
#
#
# r2Score = r2_score(y,y_pred)
# print('r2 score is',r2Score)
#
# mse = mean_squared_error(y,y_pred)
# print('RMSE value is',mse)



# ========== Question 3.2 --- [4 marks] ==========
# [Code] Examine the scatter plot of engine-size vs price (plot below)
# [Text] Why might this cause a problem for linear regression?

# X = auto_numeric['engine-size']
#
# plt.figure()
#
# plt.subplot(1,1,1)
#
# plt.scatter(X,y)
#
# plt.xlabel('engine-size')
#
# plt.ylabel('price')
#
# plt.show()

# ========== Question 3.3 --- [10 marks] ==========
# In class we discussed ways of preprocessing features to improve performance in such cases.
#
# [Code] Transform the engine-size attribute using an appropriate technique from the lectures (document it in your code) and show the transformed data (scatter plot).
# [Code] Then retrain a (Multi-variate) LinearRegression Model (on all the attributes including the transformed engine-size) and report $R^2$ and RMSE.
# [Text] How has the performance of the model changed when compared to the previous result? and why so significantly?

# for i in range(len(auto_numeric['engine-size'])):
#     value = auto_numeric['engine-size'].iloc[i]
#     value = log(value, 2)
#     auto_numeric['engine-size'].iloc[i] = value
#
# plt.figure()
# plt.subplot(1,1,1)
# plt.scatter(auto_numeric['engine-size'],auto_numeric['price'])
# plt.xlabel('engine-size')
#
# plt.ylabel('price')
#
# plt.show()
#
#
# lm = LinearRegression(normalize=True)
# X = auto_numeric.drop(['price'], axis=1)
# y = auto_numeric['price']
# y_pred = cross_val_predict(lm, X, y=y, cv=kf)
#
# print(y_pred)
#
#
# r2Score = r2_score(y,y_pred)
# print('r2 score is',r2Score)
#
# mse = mean_squared_error(y,y_pred)
# print('RMSE value is',mse)



# ========== Question 3.4 --- (LEVEL 11) --- [12 marks] ==========
# The simplicity of Linear Regression allows us to interpret the importance of certain features in predicting target variables.
# However this is not as straightforward as just reading off the coefficients of each of the attributes and ranking them in order of magnitude.
#
# [Text] Why is this? How can we linearly preprocess the attributes to allow for a comparison? Justify your answer.
# [Code] Perform the preprocessing you just mentioned on the transformed data-set from Question 3.3, retrain the Linear-Regressor and report the coefficients in a readable manner.
# Tip: To simplify matters, you may abuse standard practice and train the model once on the entire data-set with no validation/test set.
# [Text] Which are the three (3) most important features for predicting price under this model?

#Stepwise regression

X = auto_numeric.drop('price',axis=1)
y = auto_numeric['price']

XColumns = X.columns.values

print(XColumns)

r2ScoreDic = {}



for i in range(len(X)):

    tempScoreDic = {}
    for column in XColumns:
        #train each attribute and compare r2Score
        #selectMax

        if column not in r2ScoreDic.keys():


            trainColums = []
            if len(r2ScoreDic.keys()) == 0:
                trainX = auto_numeric[column]
                trainX = np.reshape(trainX.values, (trainX.shape[0], 1))
            else:
                trainColums = list(r2ScoreDic.keys())
                trainColums.append(column)
                trainX = auto_numeric[trainColums]

            lr = LinearRegression(normalize=True)
            lr.fit(trainX, y)
            pred_y = lr.predict(X=trainX)
            score = r2_score(y, pred_y)
            tempScoreDic[column] = score

    sortedTempList = sorted(tempScoreDic.items(),key = lambda d:d[1],reverse=True)
    key,value = sortedTempList[0]
    r2ScoreDic[key] = value
    print(r2ScoreDic)
    print(len(r2ScoreDic))







