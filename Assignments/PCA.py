
from IPython.display import display # Allows multiple displays from a single code-cell

# For Getting the Data
from sklearn.datasets import fetch_20newsgroups, load_digits
from sklearn.feature_extraction.text import TfidfVectorizer

# System functionality
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler



# ========== Question 1.1 --- [9 marks] ==========
digits = load_digits()

print(digits.keys())

# print(digits.images)
print(digits.images.shape)

columnNames = []
for i in range(64):
    columnNames.append('pixel{}'.format(i + 1))

data_set = pd.DataFrame(data=digits.data,columns=columnNames)
target_data = digits.target


print(data_set.describe())
print(data_set.shape)
print(data_set.head())


# the column 1 - 64 represent the pixel, one column represent one pixel.
# each row represent one image
# the target_data represent the true number of the image.



# ========== Question 1.2 --- [12 marks] ==========

# globleStd = np.std(data_set,axis=0)
# print(globleStd.values)
# globleStd = np.reshape(a=globleStd.values, newshape=(8,8))
#
# plt.figure(figsize=(6.4 * 2,4.8 * 6))
# plt.subplot(6,2,1)
#
#
# sns.heatmap(globleStd, annot=True)
# plt.title("Standard Deviation over the entire mnist dataset")
#
#
# groupbyData = data_set.groupby(target_data)
# groupbyData = groupbyData.std()
#
# for i in range(10):
#     plt.subplot(6,2,i+3)
#     stdData = groupbyData.iloc[i,:].values
#     stdData = np.reshape(a=stdData, newshape=(8,8))
#     plt.title('digit:{}'.format(i))
#     sns.heatmap(stdData, annot=True)
#
# plt.show()

#Data is easier to process and use in low dimensions, and the overhead of the algorithm is greatly reduced.

# As can be seen from the figure, the left and right sides are mostly black (ie, the variance is close to 0).
# Corresponding to the figure, for the recognition of the digit, the effect of the pixels on the left and
# right sides of the picture is not as great as the effect of the middle pixel.
# Therefore,it also shows that the dimensions corresponding to the pixels on the left and right sides are not very important.



# ========== Question 2.1 --- [16 marks] ==========


# pca = PCA(n_components=data_set.shape[1],svd_solver='full')
# pca = pca.fit(data_set)
#
# plt.figure()
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
#
# X = np.arange(0,64)
#
# explained_variance_ratio = pca.explained_variance_ratio_
#
# ax1.bar(X,explained_variance_ratio)
# ax1.set_xlim(-1,65)
# ax1.set_ylabel('explained variance ratio')
# ax1.set_xlabel('number of component')
#
# ax2.plot(pca.explained_variance_,color="red")
# ax2.set_ylabel('explained variance')
#
# eightyPercentIndex = 0
# cumululative_explained_variance = 0.0
#
# for i in range (len(explained_variance_ratio)):
#     cumululative_explained_variance += explained_variance_ratio[i]
#     if (eightyPercentIndex == 0 and cumululative_explained_variance > 0.8):
#         eightyPercentIndex = i+1
#
# ax1.vlines(x=eightyPercentIndex, ymin=0, ymax =.15, linestyles="dotted",color='black')
# label = "{} Component".format(eightyPercentIndex)
# plt.annotate(label,
#              xy=(13,0),
#              xytext=(+50,+50),
#              textcoords='offset points',
#              arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2',color='black'),
#              color='black',
#              fontsize=12,
#              zorder=10)
#
#
# plt.show()

# From the figure, the variance value decreases rapidly from
# close to 0.15 at the beginning, and reaches about 0.02
# when it reaches the thirteenth component.
# Then the downward trend tends to be gentle.
# Because our goal is to reduce the dimension,
# then if the dimension can be reduced from 64 to 13, it will greatly improve efficiency.
# On the other hand, because the pixel occupying a small variance
# has little effect on the recognition of the entire picture,
# as shown in the graph of question 1.2.



# First, if you can reduce from 64 to 13 dimensions,
# the classification algorithm will increase efficiency when processing these data.
# 13 components can explain 80% of the variance, which proves
# that the remaining components contribute little to the recognition of the image.



# ========== Question 2.2 --- [10 marks] ==========


# plt.figure(figsize=(6.4 * 2, 4.8 * 2))
# plt.subplot(2,2,1)
#
# plot_mean = pca.mean_
# plot_mean = np.reshape(a = plot_mean, newshape=(8,8))
# ax1 = sns.heatmap(plot_mean,annot=True,cmap='binary')
# ax1.set_title('mean')
#
# plot_components = pca.components_
#
# for i in range(3):
#     plt.subplot(2,2,i+2)
#     plot_component = np.reshape(a = plot_components[i], newshape=(8,8))
#     ax = sns.heatmap(plot_component,annot=True,cmap='binary')
#     ax.set_title("the {} component".format(i))
#
#
# plt.show()



# ========== Question 2.3 --- [14 marks] ==========





# plt.figure(figsize=(6.4,4.8 * 2))
#
# plt.subplot(2,1,1)
#
# first_image_data = digits.data[0,:]
# plot_first = np.reshape(a = first_image_data, newshape=(8,8))
# ax = sns.heatmap(plot_first,annot=True,cmap='binary')
# ax.set_title("the original data of first image")
#
# projectValue = pca.transform(first_image_data.reshape(1, -1)).flatten()
#
# print(first_image_data.shape)
# print(projectValue.shape)
#
# newImage = pca.mean_ + np.dot(first_image_data,projectValue)
#
# plt.subplot(2,1,2)
# plot_first_new = np.reshape(a = newImage, newshape=(8,8))
# ax1 = sns.heatmap(plot_first_new,cmap='binary')
# ax1.set_title("the new data of first image")
#
# plt.show()


# ========== Question 2.4 --- (LEVEL 11) --- [18 marks] ==========

kf = KFold(n_splits=5,random_state=0,shuffle=True)
svc_linear = SVC(kernel="linear")

pca = PCA(n_components=data_set.shape[1])
pca_data = pca.fit_transform(data_set)

scores = []
pca_scores = []
for kf_train_indexes, kf_test_indexes in kf.split(data_set):
    svc_linear.fit(X=data_set.loc[kf_train_indexes],y=target_data[kf_train_indexes])
    svc_score = svc_linear.score(X=data_set.loc[kf_test_indexes],y=target_data[kf_test_indexes])

    scores.append(svc_score)

    svc_linear.fit(X=pca_data[kf_train_indexes],y=target_data[kf_train_indexes])
    svc_score = svc_linear.score(X=pca_data[kf_test_indexes],y=target_data[kf_test_indexes])

    pca_scores.append(svc_score)



print('mean acc for raw data',np.mean(scores))
print('mean acc for PCA-transformed data',np.mean(pca_scores))
#

# nbScore = []
# pca_nbScore = []
# gaussianNB = GaussianNB()
# for kf_train_indexes, kf_test_indexes in kf.split(data_set):
#     gaussianNB.fit(X=data_set.loc[kf_train_indexes],y=target_data[kf_train_indexes])
#     multScore = gaussianNB.score(X=data_set.loc[kf_test_indexes],y=target_data[kf_test_indexes])
#     nbScore.append(multScore)
#
#     gaussianNB.fit(X=pca_data[kf_train_indexes],y=target_data[kf_train_indexes])
#     multScore = gaussianNB.score(X=pca_data[kf_test_indexes],y=target_data[kf_test_indexes])
#     pca_nbScore.append(multScore)
#
# print('mean acc for raw data',np.mean(nbScore))
# print('mean acc for PCA-transformed data',np.mean(pca_nbScore))


nbScore = []
pca_nbScore = []
gaussianNB = GaussianNB()
for kf_train_indexes, kf_test_indexes in kf.split(data_set):
    gaussianNB.fit(X=data_set.loc[kf_train_indexes],y=target_data[kf_train_indexes])
    multScore = gaussianNB.score(X=data_set.loc[kf_test_indexes],y=target_data[kf_test_indexes])
    nbScore.append(multScore)

    gaussianNB.fit(X=pca_data[kf_train_indexes],y=target_data[kf_train_indexes])
    multScore = gaussianNB.score(X=pca_data[kf_test_indexes],y=target_data[kf_test_indexes])
    pca_nbScore.append(multScore)

print('mean acc for raw data',np.mean(nbScore))
print('mean acc for PCA-transformed data',np.mean(pca_nbScore))


