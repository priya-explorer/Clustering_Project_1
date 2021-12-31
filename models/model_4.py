# todo: Hierarchical Clustering
# Model is build by removing "Total_Revolving_Bal" & "Credit_Limit" features since its proportion  is
# already represented as "Avg_Utilization_Ratio" in the data.
# i.e, Avg_Utilization_Ratio = Total_Revolving_Bal\Credit_Limit ----> Proportion of the limit used

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
numerical_std = numerical_std.drop(["Credit_Limit", "Total_Revolving_Bal"], axis=1)
print(numerical_std.head())

# todo: Run Dendrogram based hierarchical clustering
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(20, 10))
plt.title("The Cluster Dendrogram")
dend_cluster = shc.dendrogram(shc.linkage(numerical_std, method='centroid'))
plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
cluster_label=cluster.fit_predict(numerical_std)
numerical_std['cluster_label']=pd.DataFrame(cluster_label)

from sklearn.metrics import silhouette_score
cluster_label=numerical_std['cluster_label']
silhouette_avg = silhouette_score(numerical_std, cluster_label)
print(silhouette_avg)

from sklearn.metrics import davies_bouldin_score
d=davies_bouldin_score(numerical_std,cluster_label)
print(d)
