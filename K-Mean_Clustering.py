# UNSUPERVISED LEARNING
# K MEAN CLUSTERING


# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING OUR DATASET

dataset = pd.read_csv('/Users/Apple/KMeans.csv')
X = dataset.iloc[:,[1,2]].values

#USING THE ELBOW METHOD TO FINDT THE OPTIMAL NUMBER OF CLUSTERING 
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0 )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#Ploting our CLUSTERS - Elbow Method
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Now we got number of cluster from above plotting which is 5

# Now lets fit the kmeans to dataset

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0 )
y_kmeans = kmeans.fit_predict(X)

# Visualization of Clustering


plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1], s = 100, c = 'red', label = 'Cluster 1' )
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1], s = 100, c = 'blue', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1], s = 100, c = 'orange', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 200, c = 'black',label = 'centroid')
 
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending score (1 - 100%)')
plt.legend()
plt.show()
