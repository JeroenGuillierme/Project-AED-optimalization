import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd


data = pd.read_csv('DATA/updated_aed_df_with_all_distances.csv')
# Load Belgium shapefile
belgium_boundary = gpd.read_file('DATA/België.json')

pd.set_option('display.max_columns', None)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Drop rows where Latitude or Longitude are NaN
data = data.dropna(subset=['Latitude', 'Longitude'])

# Impute missing values for Eventlevel with median
imputer = SimpleImputer(strategy='median')
data['Eventlevel'] = imputer.fit_transform(data[['Eventlevel']])

# Fill NaN values in Occasinal_Permance with a specific value,-1
data['Occasional_Permanence'].fillna(-1, inplace=True)

# Assuming T3-T0 NaN values can be filled with a specific value or median
data['T3-T0_min'] = imputer.fit_transform(data[['T3-T0_min']])

# Feature selection
features = ['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel', 'T3-T0_min', 'AED',
            'Ambulance', 'Mug', 'Occasional_Permanence']

# Standardize the features
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data[features])

print(data[features].isna().sum())
print(data.head())

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ELBOW METHOD
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Determine the optimal number of clusters using the Elbow method
wcss = [] # Within Cluster Sum of Squares
NoC = 50 # max number of clusters
for i in range(1, NoC+1):
    kmeans = KMeans(n_clusters=i, random_state=45)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, NoC+1), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow method 12 clusters were chosen
optimal_clusters = 12


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# KMEANS CLUSTERING
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Apply KMeans clustering
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize the clusters

plt.figure(figsize=(10, 6))
belgium_boundary.plot(color='lightgrey', edgecolor='black')
sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', data=data, palette='viridis')
plt.title('Clusters of AED Placement Locations')
plt.show()

# Create separate plots for each cluster
for cluster in range(optimal_clusters):
    plt.figure(figsize=(10, 6))
    belgium_boundary.plot(color='lightgrey', edgecolor='black')
    clustered_data = data[data['Cluster'] == cluster]
    sns.scatterplot(x='Longitude', y='Latitude', data=clustered_data, color='red', palette='viridis')
    plt.title(f'Cluster {cluster+1}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend([],[], frameon=False)  # Hide the legend to avoid clutter
    plt.show()

# Analyze clusters to identify potential AED placement locations
for cluster in range(optimal_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    # Perform analysis to identify areas lacking AEDs
    # This could be based on counts of AEDs, interventions, and other factors
    print(f'Cluster {cluster+1}:')
    print(cluster_data.describe())
    print()


# For efficiency export clustered dataset so that the KMeans algorithm doesn't have to run every time
#cluster_data.to_csv('DATA/aed_clusters.csv', index=False)

# Now look if there is a clusters which represent the locations where low AED coverage is?
'''
CLUSTER 1
> Represents the interventions locations with mostly no close aed's and a mean response time of 10 min

CLUSTER 2
> Represents the current AED locations in Antwerp

CLUSTER 3
> Represents the Ambulance and MUG locations which are permanently available

CLUSTER 4
> Represents the Ambulance locations which are not permanently available

CLUSTER 5
> Represents the interventions of type CAD9

CLUSTER 6
> Represents current AED locations in Wallonia


'''