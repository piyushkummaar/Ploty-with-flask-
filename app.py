from flask import Flask, render_template
import pandas as pd
import json
import plotly
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/treemap')
def treemap():
    '''
    Centroids cluster
    '''
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

    #centroids of the clusters
    # print(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1])
    df = pd.DataFrame()
    df['x'] = kmeans.cluster_centers_[:, 0]   
    df['y'] = kmeans.cluster_centers_[:,1]

    fig = px.treemap(df, path=['x', 'y'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Plotting the results"
    description = """Iris dataset centroid clusters."""
    return render_template('index.html', graphJSON=graphJSON, header=header,description=description)

@app.route('/treemapIRIS')
def simpletreemap():
    dataset = pd.read_csv('Iris.csv')
    fig = px.treemap(dataset, path=['SepalLengthCm', 'SepalWidthCm', 'Species'], values='PetalWidthCm')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    header="Plotting the results"
    description = """Iris dataset each data point is represented as a marker point."""
    return render_template('index.html', graphJSON=graphJSON, header=header,description=description)

if __name__ == "__main__":
    app.run()