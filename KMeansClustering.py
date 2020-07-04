import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv')

X = dataset.iloc[:,3:].values#2 cols only for visualization purpose and to makes things easier

#using elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=42) #init is initialising kmeans++ for avoiding random initilalization trap
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #intertia calulates wcss the given number of clusters

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#Training the model on the dataset
kmeans=KMeans(n_clusters=5, init='k-means++', random_state=42) #init is initialising kmeans++ for avoiding random initilalization trap
y_kmeans=kmeans.fit_predict(X) #trains and creates the cluster indexes for the data

#Visualizing the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100, c ='red',label='Cluster 1')#here x is annual income an Y is spending score. In X we select only the ones that have specific cluster index
#s is for size, c is for color
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100, c ='blue',label='Cluster 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100, c ='green',label='Cluster 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100, c ='cyan',label='Cluster 4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100, c ='magenta',label='Cluster 5') 
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')   #cluster_centers_ 1st column x axis, 2nd col y-axis
plt.title('Clusters of Customers')
plt.xlabel('Annual income in ($k)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()
