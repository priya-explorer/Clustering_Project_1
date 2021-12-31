# todo: Hierarchical Clustering

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

# todo: Run Dendrogram based hierarchical clustering
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(20, 10))
plt.title("The Cluster Dendrogram")
dend_cluster = shc.dendrogram(shc.linkage(numerical_std, method='centroid'))
plt.show()