# Heirarichal clustering using Dendogram
# UNSUPERVISED LEARNING



# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING OUR DATASET

dataset = pd.read_csv('/Users/Apple/KMeans.csv')
X = dataset.iloc[:,[1,2]].values

#USING THE ELBOW METHOD TO FINDT THE OPTIMAL NUMBER OF CLUSTERING 
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))


#Ploting our Dendogram

plt.title('Dendogram')
plt.xlabel('Customer')
plt.ylabel('Distance')
plt.show()

# Now we got number of cluster from above plotting which is 5

# Now lets fit the kmeans to dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean",linkage='ward') 
y_hc=hc.fit_predict(X)

# Visualization of Clustering


plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s = 100, c = 'red', label = 'Cluster 1' )
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s = 100, c = 'blue', label = 'Cluster 3')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1], s = 100, c = 'orange', label = 'Cluster 5')

 
plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending score (1 - 100%)')
plt.legend()
