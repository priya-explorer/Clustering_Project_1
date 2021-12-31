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
df = pd.read_excel(r"C:\Users\priya\PycharmProjects\clustering_project_1\data\raw_data\card_customer_data.xlsx")
# print(df.head(5))

# Dropping the 'CLIENTNUM' column is irrelevant in the analysis and model building
df = df.drop(['CLIENTNUM'], axis=1)

# todo: Split features into Numerical and Categorical features dataframe
numerical_df = df.select_dtypes(include="number")
char_df = df.select_dtypes(include="object")

# todo: Select Numerical Features for Clustering
print("\nThe datatype in numeric features  are \n", numerical_df.dtypes)

# todo: Outlier Treatment - Capping and Flooring of outliers
numeric_columns = list(numerical_df.columns)
for col in numeric_columns:
    percentiles = numerical_df[col].quantile([0.01, 0.99]).values
    numerical_df[col] = np.clip(numerical_df[col], percentiles[0], percentiles[1])
print("\n The Summary Statistics after Outlier Treatment")
print(numerical_df.describe(percentiles=[0.01, 0.99]))

# todo: Feature Engineering
# Average Spend per Transaction
numerical_df['avg_spend'] = numerical_df['Total_Trans_Amt'] / numerical_df['Total_Trans_Ct']
print("\nAfter building a derived feature")
print(numerical_df.head())

# todo: Feature Scaling
from sklearn.preprocessing import StandardScaler

std_scaling = StandardScaler()
numerical_std = pd.DataFrame(std_scaling.fit_transform(numerical_df), index=numerical_df.index,
                             columns=numerical_df.columns)

print("\nThe Transformed data after Standard Scaling \n", numerical_std.head())
print("\n The Summary Statistics of transformed scaled data\n", numerical_df.describe())

# todo: Correlation Check
# To understand the linear relationship between different variables
correlation_matrix = numerical_std.corr()

# todo: To find the highly correlated variables
# Select upper triangle of correlation matrix
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

# unstacking 'upper' correlation matrix and sorting it
upper = upper.unstack().sort_values(ascending=True)

# to view the highly positively correlated variables
high_correlation = upper[(upper >= 0.75) & (upper < 1) & (upper != 'Nan')]
# Threshold is greater than or equal  to 0.75 but less than 1
print()
print("The highly positively correlated variables and its value are:\n\n{}".format(high_correlation))

# to view the highly negatively correlated variables
high_negative_correlation = upper[(upper < -0.5) & (upper != 'Nan')]
print("The highly negatively correlated variables and its value are:\n{}".format(high_negative_correlation))

# todo: View the Heat Map
# shows the pearson correlation coefficient(r) between two variables
f, ax = plt.subplots(figsize=(10, 10))
matrix = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, square=True, mask=matrix, fmt='.2f', cmap='Blues')
plt.show()

# todo: Dropping Highly Correlated Variables
numerical_std = numerical_std.drop(['Total_Trans_Amt', 'Total_Trans_Ct'], axis=1)
# todo: To view the heap map after dropping correlated variables
plt.figure(figsize=(10, 10))
sns.heatmap(numerical_std.corr(), annot=True, square=True, fmt='.2f', cmap='Blues')
plt.show()

# View the dataframe
print("\n The numerical feature dataframe after preprocessing")
print(numerical_std.head())
print("The number of rows and columns in the data for model building are ", numerical_std.shape)

# todo: Converting the data required for model building to csv format
numerical_std.to_csv(r"C:\Users\priya\PycharmProjects\clustering_project_1\data\model_building_data.csv", index=False)


# todo: Joining the Numerical and Categorical dataframes
data_all = pd.concat([numerical_df, char_df], axis=1, join="inner")
print("\n The combined data")
print(data_all.head())
print("The number of rows and columns in the combined data are", data_all.shape)

# todo: Converting the data_all to csv file
data_all.to_csv(r"C:\Users\priya\PycharmProjects\clustering_project_1\data\combined_data.csv", index=False)
