
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
from utils.plotter import plot_voronoi



# Load the Data with only the required categories
cats = ['alt.atheism', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
        'rec.sport.baseball', 'rec.sport.hockey']
newsgroups_train = fetch_20newsgroups(subset='train', categories=cats,
                                      remove=('headers', 'footers', 'quotes'), data_home='./datasets')
# Transform Data to TF-IDF and store in X/y
vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000, min_df=2, stop_words='english',
                             token_pattern='(?u)\\b[A-z]{2,}\\b')
X = pd.DataFrame(vectorizer.fit_transform(newsgroups_train.data).todense(),
                 columns=vectorizer.get_feature_names())
y = pd.Series(np.array(cats)[newsgroups_train.target])


# print(X.describe())
# print(X.shape)
# print('sum',sum(X.iloc[1,:]))
# print(y.shape)
# print(y[0:10])

# X is the features, and have 1000 columns(1000 words), each word represent one feature,and the value of X
# is the tf-idf weights of each word. each row represent the features in one article
# y is the target label of X, each row means the type of the article

# If you simply use the frequency of times, then the weight of some high-frequency words will be even greater.
# TF-IDF tends to filter out common words and retain important words


# ========== Question 2.1 --- [6 marks] ==========
# We will now use K-Means clustering as a means of unsupervised learning of the document classes. Familiarise yourself with the implementation and then answer the following questions.
#
#   (a) [Text] The KMeans algorithm is non-deterministic. Explain what is meant by this, why this is the case, and how the final model is selected (3 sentences).
#   (b) [Text] One of the parameters we need to specify when using k-means is the number of clusters. What is a reasonable number for this problem and why? Hint: Look at the y values, which are in a Pandas Series.
#   (b) [Code] Create and fit a K-Means model to the training data X with your specified number of clusters.
# For reproducability, set random_state=1000 -- keep other arguments at default values. Keep track of the k-means object created.

#a)
# Use this algorithm for the same batch of data, the final result is not necessarily the same
# Because each initial average is not necessarily the same
# The final result may be a local optimal solution

#b)because there are five labels of y, so the reasonable number of clusters is 5

kmeans = KMeans(n_clusters=5,random_state=1000)
kmeans.fit(X)

# ========== Question 2.2 --- [6 marks] ==========
# We will now evaluate the quality of this fit, primarily through the Adjusted Rand Index (ARI) of the model.
#
#   (a) [Text] By referring to the sklearn documentation, describe what the ARI (adjusted_rand_score) measures about the quality of the fit. What is a disadvantage of this measure for evaluating clustering performance? (2-3 sentences)
#   (b) [Code] Compute (and display) the Adjusted Rand Index of the fitted model.
#   (c) [Text] Comment (1 or 2 sentences) on the quality of the fit as expressed by this measure.

# measures the similarity of the two assignments,The value is between [-1,1],
# the negative number means the result is not good, the closer to 1 the better
# ARI need the ture labels to measures the similarity

y_pred = kmeans.predict(X)
ARI = adjusted_rand_score(y_pred,y)
print('Adjusted Rand Index is', ARI)


# as I mentioned that the score is the closer to 1 the better.
# the score is about 0.2. This score is not particularly satisfactory and it can be improved



# ========== Question 2.3 --- [12 marks] ==========
# Let us explore the quality of the fit further. We will visualise bar-charts of the fit.
#
#   (a) [Code] Create a bar-chart of the number of data-points with a particular class assigned to each cluster centre. You should be able to do this in one plot using seaborn functionality. Make sure to distinguish the cluster-centres (using for example grid-lines), and label the plot appropriately. As part of the cluster labels, include the total number of data-points assigned to that cluster. Hint: it might be useful to specify the image width using pylab.rcParams['figure.figsize'] = (width, height) to make things clearer.
#   (b) [Text] Comment (3 to 4 sentences) on the distribution of datapoints to cluster centres, and how this relates to the ARI measure above.

# plt.figure(figsize=(12.8,9.6))
# sns.countplot(x=kmeans.labels_, hue=y)
# plt.xlabel("cluster ID")
# plt.title("counts of true labels")
# plt.legend(fontsize=10)
# plt.show()

# As can be seen from the figure, only the display of cluster 4 is satisfactory because this cluster represents a relatively independent alt.atheism. Clustering 0, 2 is a clustering of PC and mac, respectively, because the relationship between the two categories is relatively close, so they are gathered together. Cluster 1 contains almost all categories.
# So overall, the clustering algorithm does not perform well, and it also matches the ARI score.


# ========== Question 2.4 --- (LEVEL 11) --- [11 marks] ==========
# Another way to quantify the quality of the fit is to use the Aggregate Intra-Cluster Distance (this is known as the inertia within SKLearn).
#
#   (a) [Text] Again by referring to the sklearn documentation, describe what this measure reports. Indicate why it may be less straightforward at judging the quality of the clustering than the ARI. (2 to 3 sentences).
#   (b) [Code] Report the Inertia of the fitted model as well as the mean distance between each data-point and the global mean. Compute also a distance matrix such that the entry with index (i,j) shows the distance between centre i and j.
#   (c) [Text] Using the above values, comment on what the Inertia score tells us about the quality of the fit, as well as anything else you can say about the clusters. (2 to 3 sentences)



# inertia is used to evaluate whether the number of clusters is appropriate. The smaller the distance, the better the clustering.
# how,it has two drawbacks one is that clusters are assumpted be convex and isotropic, another is that In high dimensional space, the Euclidean distance will become inflated.
# just like this case, there are 1000 features


# inertia = kmeans.inertia_
#
# global_mean_distance = np.mean(np.linalg.norm(X-X.mean(), axis=0))
#
#
#
#
# distances = np.empty((5,5))
# for i in range(5):
#     for j in range(5):
#         distances[i,j] = np.linalg.norm(kmeans.cluster_centers_[i] - kmeans.cluster_centers_[j])
#
#
# print ("inertia:", inertia)
# print ("the global mean distance:",global_mean_distance)
# print ("distance matrix: \n",distances)

# The global average distance is 1.55, a total of 2845 rows of data, then the total distance exceeds 4400 without clustering.
# Inertia is 2500, which means the algorithm has a certain effect, but the distance of 2500 is not ideal.



# ========== Question 2.5 --- [16 marks] ==========
# We will now investigate using PCA dimensionality reduction to try and improve the quality of the fit.
#
#   (a) [Text] Give one reason why PCA might be preferrable in certain cases in reducing dimensionality over just picking a subset of the features.
#   (b) [Code] Pick 10 values in the range [1, ... 1000] inclusive, representing feature-space dimensionality n. Use a log-scale to span this range efficiently. For each of these values, reduce the dimensionality of X to the respective size (i.e. PCA with n components), and then fit a 5-centre KMeans classifier, storing the adjusted_rand_score for each dimensionality. N.B: Set the random_state=1000 for both PCA and K-Means objects to ensure reproducability.
#   (c) [Code] Plot the the adjusted_rand_score against the number of principal components. Scale the axes appropriately to visualise the results, and label the plot.
#   (d) [Text] Comment on the effect dimensionality reduction had on the K-Means clustering and what could give rise to this (2 - 3 sentences).
#   (e) [Code] Fit a 5-cluster K-Means object to the data with the dimensionality that produced the best ARI. In a similar manner to Question 2.3, plot a bar-chart of the number of data-points from each class assigned to each cluster. N.B: Remember to set random_state=1000 for both PCA and K-Means objects, and label all your diagrams.
#   (f) [Text] Compare the clustering distribution in (e) and in Question 2.3 (a). Hint: comment briefly (1 to 2 sentences) on the distribution of classes.

# Because all features are of certain importance and cannot be abandoned

# components = np.logspace(start=0,stop=3,num=10)
#
# scores = []
#
# for component in components:
#     component = int(component)
#     pca = PCA(n_components=component,random_state=1000)
#     pca_data = pca.fit_transform(X=X)
#     pca_kmeans = KMeans(n_clusters=5,random_state=1000)
#     pca_kmeans.fit(X=pca_data)
#     y_pca_pred = pca_kmeans.predict(pca_data)
#     pca_ARI = adjusted_rand_score(y,y_pca_pred)
#     scores.append(pca_ARI)
#     print('component is: {} adjusted_rand_score is:{}'.format(component,pca_ARI))

#
# As can be seen from the figure, the effect is best when n_component is 2, and the effect is second when ARI is 0.268. When n_component is 4, the ARI is 0.234.
# The reduction in the dimensions of other values does not work better than when there is no reduction.

# plt.figure()
#
# ax = sns.scatterplot(x = components, y=scores)
#
# plt.semilogx()
#
# plt.xlabel('number of component')
#
# plt.ylabel('adjusted rand index')
#
# plt.title("adjusted rand index of different number of component")
#
#
# plt.show()

#



# plt.figure(figsize=(12.8,9.6))
#
# pca = PCA(n_components=2, random_state=1000)
# pca_data = pca.fit_transform(X=X)
# pca_kmeans = KMeans(n_clusters=5, random_state=1000)
# pca_kmeans.fit(X=pca_data)
# y_pca_pred = pca_kmeans.predict(pca_data)
# pca_ARI = adjusted_rand_score(y,y_pca_pred)
# print(pca_ARI)
# sns.countplot(x=pca_kmeans.labels_, hue=y)
# plt.xlabel("cluster ID")
# plt.title("counts of true labels")
# plt.legend(fontsize=10)
# plt.show()


#


# ========== Question 2.6 --- [16 marks] ==========
# Another benefit of Dimensionality Reduction is that it allows us to visualise the data. That is, while we cannot visualise a 1000-feature space, we can pick the top two components and visualise those. We will do this by means of a Voronoi Diagram, which we will use to analyse the cluster centres.
#
#   (a) [Text] By explaining what a Voronoi Diagram is, indicate why it is useful in visualising the extent (in space) of K-Means clusters?
#   (b) [Code] Using the function plot_voronoi which we provided in the utils package, visualise the clusters in the two-dimensional PCA space (top two components). Mark each cluster centre, and visualise some (Hint: sub-sample) of the data points to give you an idea of where the true classes lie. Make sure that the key elements of the plot are clearly visible/discernible (you may need to tweak some parameters) and label clearly all necessary elements of the plot (color-coding, data points). Tip: you may need to change y to a numeric value if using matplotlib rather than seaborn.
#   (c) [Text] What can you conclude from the plot as to the classification-performance of the K-Means classifier?


# Because there is a P point in each block in the graph, all the points in the block have less or equal distances than any other P in other block.
# This is similar to K-Means, all points are finally classified to the shortest distance cluster

pca = PCA(n_components=2, random_state=1000)
pca_data = pca.fit_transform(X=X)
pca_kmeans = KMeans(n_clusters=5, random_state=1000)
pca_kmeans.fit(X=pca_data)

x_min, x_max = pca_data[:, 0].min(), pca_data[:, 0].max()
y_min, y_max = pca_data[:, 1].min(), pca_data[:, 1].max()

data_range = x_min,x_max,y_min,y_max

plt.figure()

plot = plot_voronoi(pca_kmeans,data_range)


# plt.annotate(s="rec.sport.hockey", xy=[-.4,.4], xytext=[-.4,.3])
# plt.annotate(s="comp.sys.ibm.pc.hardware", xy=[.4,.4], xytext=[.1,.35])
# plt.annotate(s="alt.atheism", xy=[-.2, -.4], xytext=[-.3,-.4])
# plt.annotate(s="comp.sys.mac.hardware", xy=[.4,-.2], xytext=[.15,-.2])
# plt.annotate(s="rec.sport.baseball", xy=[0,.2], xytext = [-.5, -.1])

colors = ['black','blue','green','red','purple']
#
#
labels = ['rec.sport.hockey','comp.sys.ibm.pc.hardware','alt.atheism','comp.sys.mac.hardware','rec.sport.baseball']

# colors = ['black']
# labels = ['comp.sys.ibm.pc.hardware']
for color, label in zip(colors, labels):
    plt.scatter(pca_data[y == label, 0], pca_data[y == label, 1], color=color, label=label, s=2)

centroids = pca_kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],s=70,color='w', zorder=10)

plt.legend()

# xy = (centroids[0,0],centroids[0,1])

# plt.annotate('rec.sport.hockey',
#              xy=(centroids[0,0],centroids[0,1]),
#              xytext=(+30,+30),
#              textcoords='offset points',
#              arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2',color='w'),
#              color='w',
#              fontsize=12,
#              zorder=10)
#
# plt.annotate('rec.sport.hockey',
#              xy=(centroids[0,0],centroids[0,1]),
#              xytext=(+30,+30),
#              textcoords='offset points',
#              arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2',color='w'),
#              color='w',
#              fontsize=12,
#              zorder=10)
#
# plt.annotate('rec.sport.hockey',
#              xy=(centroids[0,0],centroids[0,1]),
#              xytext=(+30,+30),
#              textcoords='offset points',
#              arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2',color='w'),
#              color='w',
#              fontsize=12,
#              zorder=10)
#
# plt.annotate('rec.sport.hockey',
#              xy=(centroids[0,0],centroids[0,1]),
#              xytext=(+30,+30),
#              textcoords='offset points',
#              arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2',color='w'),
#              color='w',
#              fontsize=12,
#              zorder=10)
#
# plt.annotate('rec.sport.hockey',
#              xy=(centroids[0,0],centroids[0,1]),
#              xytext=(+30,+30),
#              textcoords='offset points',
#              arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2',color='w'),
#              color='w',
#              fontsize=12,
#              zorder=10)

plt.title('Voronoi Diagram of K-means clustering')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

# As can be seen from the figure, most of the green points (atheism) are concentrated together,
# because atheism is independent and has no relevance to other classifications.
# Instead, black points (hockey) and purple points (baseball) are all gathered together
# because they are related. The same is true for the blue points (PC) and the red points (mac).
# Therefore, the K-Means classifier has a good effect on independent tags,
# but for related labels, the classification effect is not ideal. In general,
# the classification-performance of the K-Means classifier needs more improvement.
