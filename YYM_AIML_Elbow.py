import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv("clustering.csv")

# Display first few rows of the dataset
print(df.head())

# Drop missing values
df_cleaned = df.dropna()

# Selecting numerical columns for clustering
numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns

print("Numerical columns used for clustering:", numerical_cols.tolist())

# Feature selection for clustering
X = df_cleaned[numerical_cols]

# Apply the Elbow Method
wcss = []   # Within-cluster sum of squares

for i in range(1, 11):   # Trying different cluster numbers from 1 to 10

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.plot(
    range(1, 11),
    wcss,
    marker='o',
    linestyle='--'
)

plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')

plt.show()

# Choose optimal k
k_optimal = 3

# Apply K-Means with the optimal number of clusters
kmeans = KMeans(
    n_clusters=k_optimal,
    random_state=42,
    n_init=10
)

df_cleaned['Cluster'] = kmeans.fit_predict(X)

# Display clustered data
print(df_cleaned.head())

# Plot the clusters
plt.scatter(
    df_cleaned[numerical_cols[0]],
    df_cleaned[numerical_cols[1]],
    c=df_cleaned['Cluster'],
    cmap='viridis'
)

plt.xlabel(numerical_cols[0])
plt.ylabel(numerical_cols[1])

plt.title(f'K-Means Clustering (k={k_optimal})')

plt.colorbar(label='Cluster')

plt.show()