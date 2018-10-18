

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

#========== Question 1.1 --- [10 marks] ==========

data_path = os.path.join(os.getcwd(), 'datasets', 'raw_20news.csv')
news_raw = pd.read_csv(data_path,delimiter = ',')
print('row {} column {}'.format(news_raw.shape[0],news_raw.shape[1]))
print(news_raw.describe())

#========== Question 1.2 --- [6 marks] ==========

# print(news_raw.columns.values)


#========== Question 1.3 --- [4 marks] ==========
# A = news_raw.loc[:,['w520_sit','w2_pins']]
#
# print(A)
#
# plt.figure()
# plt.subplot(111)
# sns.stripplot(news_raw['class'], A, jitter=True, alpha = 0.3)
#
# plt.show()

#对于class的五个分类，w2_pins对于每个分类，大部分数值都是接近0的数字

#========== Question 1.4 --- [8 marks] ==========


# list = []
#
# for i in range(4):
#     singleFeatureData = np.array(news_raw[news_raw.columns[i]])
#     list.append(singleFeatureData)
#
# plt.figure(figsize=(6.4,9.6))
# plt.subplot(211)
#
# scatter_jitter(list[0],list[1])
#
# plt.xlabel(news_raw.columns[0])
# plt.ylabel(news_raw.columns[1])
#
#
# plt.subplot(212)
#
# scatter_jitter(list[2],list[3])
# plt.xlabel(news_raw.columns[2])
# plt.ylabel(news_raw.columns[3])
#
#
# plt.show()




#========== Question 1.5 --- [15 marks] ==========

#大于100

#正态分布，中位数+-3方差

news_preClean = news_raw.copy()

outliers = {}

for column in news_preClean.columns:
    if column == 'class':
        break

    columnMean = news_preClean[column].mean()
    columnStd = news_preClean[column].std()
    colomnFrame = news_preClean[column]
    for i in range(len(news_preClean[column])):

        value = news_preClean[column].iloc[i]

        if (value < columnMean - 3 * columnStd) or (value > columnMean + 3 * columnStd):
            news_preClean[column].iloc[i] = -1
            outliers[i] = 1


news_preClean_withNan = news_preClean[news_preClean > 0]

news_clean = news_preClean_withNan.dropna(axis=0, how='any')


firstHalfDataList = []
secondHalfDataList = []

for i in range(len(news_clean.columns) - 1):
    singleFeatureData = []

    for value in news_clean[news_clean.columns[i]]:
        singleFeatureData.append(value)

    if i < (len(news_clean.columns) - 1) / 2:
        firstHalfDataList.extend(singleFeatureData)
    else:
        secondHalfDataList.extend(singleFeatureData)

plt.figure()
plt.subplot(1,1,1)
scatter_jitter(firstHalfDataList,secondHalfDataList)
plt.show()

print('the shape of new_clean is {}'.format(news_clean.shape))

print('the outliers number is {}'.format(news_raw.shape[0] - news_clean.shape[0]))


# ========== Question 1.6 --- (LEVEL 11) --- [10 marks] ==========
# [Code] Visualise some of the outlier documents and some of the inlier ones.
# [Text] Comment on the observations. Also comment on whether it is appropriate to do such cleaning on just the training data or on the entire data-set (including testing).



plt.figure(figsize=(6.4,9.6))
plt.subplot(211)

plt.title('inlier')
scatter_jitter(news_clean['w519_fashion'],news_clean['w520_sit'])

plt.xlabel('w519_fashion')
plt.ylabel('w520_sit')


plt.subplot(212)


indexList = list(outliers.keys())

outDataFrame = news_raw.iloc[indexList,:]

print(outDataFrame)

plt.title('outliers')
scatter_jitter(outDataFrame['w519_fashion'],outDataFrame['w520_sit'])
plt.xlabel('w519_fashion')
plt.ylabel('w520_sit')


plt.show()



