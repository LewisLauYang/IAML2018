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

print(news_train.describe())
print(news_train.shape)
print('#######################')
print(news_test.describe())
print(news_test.shape)


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


X = news_train.drop('class',axis=1)

y = news_train['class']

mnb =

































