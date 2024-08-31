#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Learning

# * Unsupervised learning is a machine learning technique where the model learns from unlabeled data.
# * The algorithm's job is to discover patterns, structures, or relationships within the data on its own.

# ## How it works
# * Unlabeled data: The model is fed raw data without any predefined labels or categories.
# * Pattern discovery: The algorithm explores the data, identifying similarities, differences, and underlying structures.
# * Clustering: One common approach is to group similar data points together into clusters, revealing hidden patterns.
# * Dimensionality reduction: Another technique involves reducing the number of features while preserving essential information.
# 

# ## Common Techniques
# * Clustering:
#     * K-Means
#     * Hierarchical Clustering
#     * DBSCAN
# * Dimensionality Reduction:
#     * Principal Component Analysis (PCA)
#     * t-Distributed Stochastic Neighbor Embedding (t-SNE)
# 

# ## Clustering
# * Groups similar data points together into clusters.
# * Clustering is a popular unsupervised machine learning technique that groups data points into clusters based on their similarity. 
# * There are different types of clustering algorithms, and they differ in their approach and underlying assumptions.
# 

# ## How Clustering Works
# 1.Data Preparation: The data is preprocessed to handle missing values, outliers, and normalization.
# 
# 2.Distance Metric: A suitable distance metric is chosen to measure the similarity between data points (e.g., Euclidean distance, Manhattan distance).
# 
# 3.Cluster Formation: The clustering algorithm groups data points based on their similarity, creating clusters.
# 

# ### How will you define the similarity between different observations? How can we say that two points are similar to each other? 

# ## Types of Distance Metrics in Machine Learning
#     1.Euclidean Distance
#     2.Manhattan Distance
#     3.Minkowski Distance
#     4.Hamming Distance
# 

# # Euclidean Distance

# • Euclidean Distance represents the shortest distance between two vectors.
# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # K-Means Clustering

# * K-Means Clustering algorithm aims to partition n observations into k clusters, where each observation belongs to the cluster with the nearest mean.
# * Discover underlying patterns or structures within the data without any prior knowledge of the group memberships.
# * It assumes that the clusters are spherical, equally sized, and have the same variance.
# * The goal is to minimize the within-cluster sum of squares (WCSS), which is the sum of the squared distances between data points and their respective cluster centroids.
# 
# ![image.png](attachment:image.png)
# 

# ## How is WCSS calculated?
# 1. Calculate the distance: For each data point in a cluster, compute the Euclidean distance between the data point and the centroid of that cluster.
# 2. Square the distance: Square the calculated distance for each data point.
# 3. Sum within the cluster: Add up the squared distances for all data points within a single cluster.
# 4. Sum across clusters: Calculate the total WCSS by summing the WCSS of all clusters.
# 
# ![image.png](attachment:image.png)

# * The primary objective of K-Means clustering is to minimize the WCSS.
# * This means finding cluster assignments that result in data points being as close as possible to their respective cluster centroids. 
# * A lower WCSS indicates that the data points are tightly clustered around their centroids, suggesting a good clustering result.
# * WCSS is better when the sum of squares between points within a cluster is less and the sum of squares between clusters is high.
# * Low within-cluster sum of squares: This indicates that data points within a cluster are tightly packed together, meaning the cluster is compact and cohesive.
# * High between-cluster sum of squares: This suggests that the clusters are well-separated from each other, indicating distinct groupings within the data.
# 

# In[ ]:





# #  Iris flower data

# * The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician, eugenicist, and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis.[[More details](https://en.wikipedia.org/wiki/Iris_flower_data_set#:~:text=The%20Iris%20flower%20data%20set,example%20of%20linear%20discriminant%20analysis.)]
# 
# * The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor).
# * Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. 
# * Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other.

# ![https://miro.medium.com/max/2550/0*GVjzZeYrir0R_6-X.png](https://miro.medium.com/max/2550/0*GVjzZeYrir0R_6-X.png)
# 
# 

# This study we try to clustering Iris Dataset used Kmeans
# 
# [Attribute Information:
# ](https://archive.ics.uci.edu/ml/datasets/iris)
# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. class:
# -- Iris Setosa
# -- Iris Versicolour
# -- Iris Virginica

# # import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# # Reading dataset

# In[2]:


from sklearn.datasets import load_iris
data= load_iris()
data


# In[3]:


iris=pd.DataFrame(data=data.data,columns=data.feature_names)


# In[4]:


# Add the 'target' column to the DataFrame
# iris['target'] = data['target']
# Add the 'target_names' as categorical data
iris['species'] = pd.Categorical.from_codes(data.target, categories=data['target_names'])

iris.head()


# In[5]:


iris.info()
iris[0:10]


# In[6]:


iris.columns


# In[7]:


iris.rename({'sepal length (cm)':'sepal_length', 'sepal width (cm)':'sepal_width', 'petal length (cm)':'petal_length',
       'petal width (cm)':'petal_width'},inplace=True,axis=1)


# In[8]:


iris


# In[9]:


#Frequency distribution of species"
iris_outcome = pd.crosstab(index=iris["species"],  # Make a crosstab
                              columns="count")      # Name the count column

iris_outcome


# In[10]:


iris_setosa=iris.loc[iris["species"]=="Iris-setosa"]
iris_virginica=iris.loc[iris["species"]=="Iris-virginica"]
iris_versicolor=iris.loc[iris["species"]=="Iris-versicolor"]


# **Distribution plots
# **

# plot each flower to a histogram

# In[11]:


sns.FacetGrid(iris,hue="species").map(sns.distplot,"petal_length").add_legend()
sns.FacetGrid(iris,hue="species").map(sns.distplot,"petal_width").add_legend()
sns.FacetGrid(iris,hue="species").map(sns.distplot,"sepal_length").add_legend()
plt.show()


# box plot

# In[12]:


sns.boxplot(x="species",y="petal_length",data=iris)
plt.show()


# violin plot

# In[13]:


sns.violinplot(x="species",y="petal_length",data=iris)
plt.show()


# **Scatter plot**
# 

# In[14]:


sns.set_style("whitegrid")
sns.pairplot(iris,hue="species",size=3);
plt.show()


# # K-Means

# [K-means](http://https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/) is a centroid-based algorithm, or a distance-based algorithm, where we calculate the distances to assign a point to a cluster. In K-Means, each cluster is associated with a centroid.

# # Algorithm
# ![image.png](attachment:image.png)

# # How to Implementing K-Means Clustering ?
# 
# * Choose the number of clusters k
# * Select k random points from the data as centroids
# * Assign all the points to the closest cluster centroid
# * Recompute the centroids of newly formed clusters
# * Repeat steps 3 and 4
# 

# In[16]:


iris


# In[17]:


x=iris.iloc[:,0:4]
x


# In[18]:


# # Standardize the data
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(x)


# In[19]:


#Finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init ='k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    


# In[20]:


wcss


# * kmeans.inertia_ is an attribute in scikit-learn's KMeans model that represents the within-cluster sum of squares (WCSS). 
# * It quantifies the total sum of squared distances between each data point and its assigned cluster center.

# # How to find optimal K value?
# 
# ## Elbow Method
# * The elbow method is a graphical method for finding the optimal K value in a k-means clustering algorithm.
# * The elbow graph shows the within-cluster-sum-of-square (WCSS) values on the y-axis corresponding to the different values of K (on the x-axis). 
# * The optimal K value is the point at which the graph forms an elbow. 
# * When we see an elbow shape in the graph, we pick the K-value where the elbow gets created. 
# * We can call this the elbow point.
# * Beyond the elbow point, increasing the value of ‘K’ does not lead to a significant reduction in WCSS.
# * Plotting inertia against the number of clusters (k) can help determine the optimal number of clusters.

# ![image.png](attachment:image.png)

# In[21]:


plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()


# # Implementing K-Means Clustering

# In[22]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
# kmeans.fit(x)
# y_pred=kmeans.predict(x)


# In[23]:


y_kmeans


# In[68]:


kmeans.labels_


# In[70]:


kmeans.cluster_centers_


# ## The Challenge with Random Initialization
# ## Randomly selecting data points as initial centroids can often lead to:
# * Uneven cluster sizes: Some clusters might end up with many more data points than others.
# * Slow convergence: The algorithm might take longer to converge.
# 
# ## K-Means++: A Smarter Initialization
# * To address these issues, K-Means++ was introduced. It employs a more intelligent approach to selecting initial centroids:
# * The first centroid is chosen randomly from the data points.
# * For each subsequent centroid, a point is chosen with a probability proportional to its squared distance from the nearest existing centroid. This step ensures that the centroids are spread out.
# * Repeat this process until all k centroids are selected.
# 

# In[22]:


sns.scatterplot(x=iris['species'],y=iris['petal_length'])


# In[24]:


# Assuming y_kmeans contains cluster labels (0, 1, 2)
cluster_0_data = x[y_kmeans == 0]
cluster_1_data = x[y_kmeans == 1]
cluster_2_data = x[y_kmeans == 2]

# Use filtered data for plotting
plt.scatter(cluster_0_data.iloc[:, 0], cluster_0_data.iloc[:, 1], s=100, c='red', label='Cluster 1')
plt.scatter(cluster_1_data.iloc[:, 0], cluster_1_data.iloc[:, 1], s=100, c='orange', label='Cluster 2')
plt.scatter(cluster_2_data.iloc[:, 0], cluster_2_data.iloc[:, 1], s=100, c='green', label='Cluster 3')
#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'black', label = 'Centroids');

plt.legend()
plt.show()


# In[25]:


cluster_0_data


# In[27]:


kmeans.cluster_centers_


# In[26]:


cluster_0_data.iloc[:, 0]


# In[29]:


# 3d scatterplot using matplotlib

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
plt.scatter(cluster_0_data.iloc[:, 0], cluster_0_data.iloc[:, 1], s=100, c='red', label='Cluster 1')
plt.scatter(cluster_1_data.iloc[:, 0], cluster_1_data.iloc[:, 1], s=100, c='orange', label='Cluster 2')
plt.scatter(cluster_2_data.iloc[:, 0], cluster_2_data.iloc[:, 1], s=100, c='green', label='Cluster 3')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'black', label = 'Centroids')
plt.show()


# * But, in the majority of the real-world data sets, there’s not a clear elbow inflection point to identify the right ‘K’ using the elbow method. 
# * This makes it easier to find the wrong K.

# # Silhouette Analysis
# * The Silhouette score is a very useful method to find the number of K when the elbow method doesn’t show the elbow point.
# * The Silhouette score is a measure of how similar a point is to its own cluster compared to other clusters.
# * The value of the Silhouette score ranges from -1 to 1. 
# * interpretation of the Silhouette score.
#     * 1: Points are perfectly assigned in a cluster and clusters are easily distinguishable. point is well-clustered
#     * 0: Clusters are overlapping. point is on or very close to the decision boundary between two neighbouring clusters.
#     * -1: Points are wrongly assigned in a cluster.
# 

# In[31]:


from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
silh=[]
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for i in range_n_clusters:
  km = KMeans(n_clusters=i,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
  km.fit(x)
  km_labels = km.labels_
  silh.append(silhouette_score(x,km_labels))


# * SilhouetteVisualizer: This visualizer displays the silhouette coefficient for each sample on a per-cluster basis.
# * It is used to evaluate the goodness of clustering.

# In[29]:


# ! pip install yellowbrick


# In[30]:


# from yellowbrick.cluster import SilhouetteVisualizer
# best_num_clusters = silh.index(max(silh)) + 2
# kmeans = KMeans(n_clusters=best_num_clusters)
# visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
# visualizer.fit(x)
# visualizer.show()


# In[27]:


plt.plot(range_n_clusters, silh, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis For Optimal k')
plt.show()


# In[38]:


# Create a subplot for each silhouette visualizer
fig, ax = plt.subplots(len(range_n_clusters), 1, figsize=(8, 24))

for idx, n_clusters in enumerate(range_n_clusters):
    # Create a KMeans instance with the current number of clusters
    kmeans = KMeans(n_clusters=n_clusters,random_state=42,init = 'k-means++', max_iter = 300, n_init = 10)
    
    # Initialize the SilhouetteVisualizer
    visualizer = SilhouetteVisualizer(kmeans, ax=ax[idx], colors='yellowbrick')
    ax[idx].set_yticks([])
    ax[idx].set_xlim([-0.1, 1]) 
    ax[idx].set_xlabel('Silhouette coefficient values')
    ax[idx].set_ylabel('Cluster labels')
    ax[idx].set_title(f'Silhouette plot for {n_clusters} clusters');
   
    
    # Fit the visualizer
    visualizer.fit(x)

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# * The output will be multiple silhouette plots, one for each number of clusters. 
# * The plot shows the silhouette coefficient for each sample, grouped by cluster. 
# * The average silhouette score for the clustering solution is also displayed, which helps in deciding the optimal number of clusters.

# # How to identify best one?
# * Compare Average Silhouette Scores: Select the number of clusters with the highest average silhouette score.
# * Analyze Cluster Consistency: Ensure the silhouette scores are consistent (wide and positive) across all clusters in the chosen plot.it indicates well-balanced clusters.

# In[39]:


from sklearn.metrics import silhouette_samples
for n_clusters in range_n_clusters:
    # Create a KMeans instance with the current number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', max_iter=300, n_init=10)
    
    # Fit the model to the data
    cluster_labels = kmeans.fit_predict(x)
    
    # Calculate silhouette scores
    silhouette_vals = silhouette_samples(x, cluster_labels)
    
    avg_silhouette_score = silhouette_vals.mean()
    print(f'Number of clusters: {n_clusters}, Average silhouette score: {avg_silhouette_score:.2f}')


# In[40]:


silhouette_vals


# * The silhouette_samples function from sklearn.metrics calculates the silhouette coefficient for each sample in your dataset, given the labels of the clusters. 
# * This is useful if you want to analyze the silhouette scores at a more granular level rather than just the overall silhouette score.

# * silhouette_samples(X, cluster_labels): Returns the silhouette score for each data point in the dataset.
# * silhouette_score(X, cluster_labels): Returns the overall silhouette score, which is the mean of all individual silhouette scores.

# In[75]:


kmeans2 = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
kmeans2.fit(x)
y_kmeans2 = kmeans2.fit_predict(x)


# In[76]:


y_kmeans2


# In[77]:


# Assuming y_kmeans contains cluster labels (0, 1, 2)
cluster_0_data = x[y_kmeans2 == 0]
cluster_1_data = x[y_kmeans2 == 1]
# cluster_2_data = x[y_kmeans2 == 2]

# Use filtered data for plotting
plt.scatter(cluster_0_data.iloc[:, 0], cluster_0_data.iloc[:, 1], s=100, c='red', label='Cluster 1')
plt.scatter(cluster_1_data.iloc[:, 0], cluster_1_data.iloc[:, 1], s=100, c='orange', label='Cluster 2')
# plt.scatter(cluster_2_data.iloc[:, 0], cluster_2_data.iloc[:, 1], s=100, c='green', label='Cluster 3')
#Plotting the centroids of the clusters
plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:,1], s = 100, c = 'black', label = 'Centroids');

plt.legend()
plt.show()


# In[75]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
kmeans.fit(x)
y_kmeans = kmeans.fit_predict(x)


# In[76]:


# Assuming y_kmeans contains cluster labels (0, 1, 2)
cluster_0_data = x[y_kmeans == 0]
cluster_1_data = x[y_kmeans == 1]
cluster_2_data = x[y_kmeans == 2]

# Use filtered data for plotting
plt.scatter(cluster_0_data.iloc[:, 0], cluster_0_data.iloc[:, 1], s=100, c='red', label='Cluster 1')
plt.scatter(cluster_1_data.iloc[:, 0], cluster_1_data.iloc[:, 1], s=100, c='orange', label='Cluster 2')
plt.scatter(cluster_2_data.iloc[:, 0], cluster_2_data.iloc[:, 1], s=100, c='green', label='Cluster 3')
#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'black', label = 'Centroids');

plt.legend()
plt.show()


# In[42]:


# # Calculate the silhouette scores for each sample
# sample_silhouette_values = silhouette_samples(x, y_kmeans)

# # Calculate the overall silhouette score (mean of individual scores)
# overall_silhouette_score = silhouette_score(x, y_kmeans)

# print(f'Overall Silhouette Score: {overall_silhouette_score}')

# # Example: print the silhouette score for the first 10 data points
# for i in range(10):
#     print(f'Sample {i+1}: Silhouette Score = {sample_silhouette_values[i]}')

# # If you want to plot the silhouette scores for each cluster:
# n_clusters = 3
# y_lower = 10
# for i in range(n_clusters):
#     ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
#     ith_cluster_silhouette_values.sort()
#     size_cluster_i = ith_cluster_silhouette_values.shape[0]
#     y_upper = y_lower + size_cluster_i

#     color = plt.cm.nipy_spectral(float(i) / n_clusters)
#     plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

#     y_lower = y_upper + 10  # Add a gap between clusters

# plt.xlabel("Silhouette coefficient values")
# plt.ylabel("Cluster label")
# plt.title(f"Silhouette plot for {n_clusters} clusters")
# plt.show()


# In[ ]:
import pickle

# Save the KMeans model
with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)




