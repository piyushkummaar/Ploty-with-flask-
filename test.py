import pandas as pd
from pandas.core.frame import DataFrame
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('Iris.csv')
x = dataset.iloc[:, [1, 2, 3, 4]].values


wcss = []
#Finding the optimum number of clusters for k-means classification
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
#Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#Visualising the clusters
# print(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1])
# print(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1])
# print(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1])

#centroids
print(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1])
df = pd.DataFrame()
df['x'] = kmeans.cluster_centers_[:, 0]   
df['y'] = kmeans.cluster_centers_[:,1]
print(df)