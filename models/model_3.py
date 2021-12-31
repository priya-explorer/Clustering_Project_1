# Model is build by removing "Total_Revolving_Bal" & "Credit_Limit" features since its proportion  is
# already represented as "Avg_Utilization_Ratio" in the data.
# i.e, Avg_Utilization_Ratio = Total_Revolving_Bal\Credit_Limit ----> Proportion of the limit used
# todo: K Means Clustering

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Warning Suppression
import warnings

warnings.filterwarnings('ignore')

# Setting Display options to ensure feature name visibility
pd.set_option('display.max_columns', None)

# todo: Load the Data
numerical_std = pd.read_csv(r"C:\Users\priya\PycharmProjects\clustering_project_1\data\model_building_data.csv")
numerical_std = numerical_std.drop(["Credit_Limit", "Total_Revolving_Bal"], axis=1)
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

cluster_label = numerical_std['cluster_label']
silhouette_avg = silhouette_score(numerical_std, cluster_label)
print("\nThe Silhouette score is", silhouette_avg)

# todo: Davies Bouldin score
from sklearn.metrics import davies_bouldin_score

db_score = davies_bouldin_score(numerical_std, cluster_label)
print("The Davies Bouldin score is", db_score)

data = pd.read_csv(r"C:\Users\priya\PycharmProjects\clustering_project_1\data\combined_data.csv")
data['cluster_label'] = numerical_std['cluster_label']
print("\n The 4 clusters formed and its value count is")
print(data['cluster_label'].value_counts().sort_values(ascending=False))
print()
print(data.head())

num = data.select_dtypes(include='number')
print("\n", num.groupby('cluster_label').agg(['mean']))

# todo: Cluster Profiling
cluster_profile_1 = pd.crosstab(index=data['cluster_label'], columns=data['Card_Category'],
                              values=data['Card_Category'], aggfunc='count')
print()
print(cluster_profile_1)

cluster_profile_2 = pd.crosstab(index=data['cluster_label'], columns=data['Card_Category'],
                                values=data['Total_Trans_Amt'],aggfunc='mean')
print()
print(cluster_profile_2)

cluster_profile_3 = pd.crosstab(index=data['cluster_label'], columns=data['Card_Category'],
                                values=data['Total_Trans_Ct'],aggfunc='mean')
print()
print(cluster_profile_3)

# todo: Visualizing the Cluster Profiles
plt.figure(figsize=(10, 8))
sns.barplot(x="cluster_label", y="Avg_Utilization_Ratio", data=data, estimator= np.mean)
plt.show()

plt.figure(figsize=(10, 8))
sns.barplot(x="cluster_label", y="Months_on_book", data=data, estimator= np.mean)
plt.show()

plt.figure(figsize=(10, 8))
sns.barplot(x="cluster_label", y="Months_Inactive_12_mon", data=data, estimator= np.mean)
plt.show()

plt.figure(figsize=(10, 8))
sns.barplot(x="cluster_label", y="avg_spend", data=data, estimator= np.mean)
plt.show()
