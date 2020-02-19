import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the Iris dataset 
dataset = pd.read_csv('Iris.csv')
print dataset.head()

'''
   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
3   4            4.6           3.1            1.5           0.2  Iris-setosa
4   5            5.0           3.6            1.4           0.2  Iris-setosa'''

# deviding x from iris dataset

x = dataset.iloc[:,1:5].values
print x
'''
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]
 [5.4 3.9 1.7 0.4]
 [4.6 3.4 1.4 0.3]
 [5.  3.4 1.5 0.2]'''


#Finding the optimum number of clusters(k)

from sklearn.cluster import KMeans
wcss = []  # within cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
#Plotting a graph, to observe 'The elbow'
    
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
#plt.savefig('Elbow_graph')
#plt.show()

'''
--> You can clearly see why it is called 'The elbow method' from the above obtained
    grap,the optimum clusters is where the elbow occurs. This is when the within cluster
    sum of squares (WCSS) doesn't decrease significantly with every iteration.
    Now that we have the optimum amount of clusters, we can move on to applying
    K-means clustering to the Iris dataset.
--> from the above elbow graph we can conclude no.of clusters are 3 where exactly
    graph has elbow '''

# Now we can apply the KMeans and create clusters

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
pred = kmeans.fit_predict(x)

print kmeans.cluster_centers_

'''
[[5.9016129  2.7483871  4.39354839 1.43387097]
 [5.006      3.418      1.464      0.244     ]
 [6.85       3.07368421 5.74210526 2.07105263]]'''

#Visualising the clusters

plt.scatter(x[pred == 0, 0], x[pred == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[pred == 1, 0], x[pred == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[pred == 2, 0], x[pred == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()
plt.savefig('Clusters_along_with_centroids')
plt.show()




