# todo: K Means Clustering

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Warning Suppression
import warnings

warnings.filterwarnings('ignore')

# Setting Display options to ensure feature name visibility
pd.set_option('display.max_columns', None)

# todo: Load the Data
numerical_std = pd.read_csv(r"C:\Users\priya\PycharmProjects\clustering_project_1\data\model_building_data.csv")
print(numerical_std.head())

# todo: WCSS Graph
# find the appropriate cluster number
plt.figure(figsize=(10, 8))
from sklearn.cluster import KMeans

wcss = []
K = range(1, 10)
for i in K:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(numerical_std)
    wcss.append(kmeans.inertia_)
plt.plot(K, wcss, 'o-')
plt.title('The Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# todo: Building Clusters
# Fitting K-Means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(numerical_std)
numerical_std['cluster_label'] = pd.DataFrame(y_kmeans)

# todo: silhouette score
from sklearn.metrics import silhouette_score
cluster_label=numerical_std['cluster_label']
silhouette_avg = silhouette_score(numerical_std, cluster_label)
print("\nThe Silhouette score is", silhouette_avg)

# todo: Davies Bouldin score
from sklearn.metrics import davies_bouldin_score
db_score = davies_bouldin_score(numerical_std, cluster_label)
print("The Davies Bouldin score is", db_score)

data = pd.read_csv(r"C:\Users\priya\PycharmProjects\clustering_project_1\data\combined_data.csv")
data['cluster_label'] = numerical_std['cluster_label']
print("\n The 4 clusters formed and its value count is")
print(data['cluster_label'].value_counts())

num = data.select_dtypes(include='number')
print("\n", num.groupby('cluster_label').agg(['mean']))
