import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
import logging
import time

data = pd.read_csv('DATA/aed_locations_extended.csv') # data generated in locations dataset extended

print(len(data))
aed_locations = pd.read_csv('DATA/updated_aed_df_with_all_distances.csv')


features = ['Unnamed: 0', 'Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel',
            'T3-T0_min', 'AED', 'Ambulance', 'Mug', 'Occasional_Permanence', 'distance_to_aed',
            'distance_to_ambulance', 'distance_to_mug', 'score', 'coverage', 'final_score']
print(features)
print(data[features].isna().sum())


# Determine the optimal number of clusters using the Elbow method
# Scale data
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data[['Latitude', 'Longitude']])

# Determine the optimal number of clusters using the Silhouette method
def determine_optimal_clusters(data, max_clusters=10):
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        km = KMeans(n_clusters=i, random_state=42)
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
        visualizer.fit(data)
        visualizer.show()
        silhouette_scores.append(silhouette_score(data, km.labels_))

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score For Optimal Number of Clusters')
    plt.show()

    optimal_clusters = np.argmax(silhouette_scores) + 2  # Add 2 because range starts from 2
    return optimal_clusters

optimal_clusters = determine_optimal_clusters(data[['Latitude', 'Longitude']])
# optimal_clusters = 15
print(f"Optimal number of clusters: {optimal_clusters}")


# Perform KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=45)
kmeans.fit(data[['Latitude', 'Longitude']])
data['cluster'] = kmeans.labels_

# Select the top new AED location in each cluster based on the final score
top_new_aed_locations = data.groupby('cluster').apply(lambda group: group.nlargest(1, 'final_score')[['Latitude', 'Longitude']]).reset_index(drop=True)

# Save new AED locations to a CSV file
top_new_aed_locations.to_csv('DATA/new_aed_locations.csv', index=False)

# Optionally, visualize the new AED locations on a map
belgium_map = folium.Map(location=[50.85, 4.35], zoom_start=8)

# Create marker clusters for better visualization
aed_old_cluster = MarkerCluster(name='Old Locations').add_to(belgium_map)
aed_new_cluster = MarkerCluster(name='New AED Locations').add_to(belgium_map)

# Add existing AED locations to the map
for _, row in aed_locations[aed_locations['AED'] == 1].iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup='Existing AED', icon=folium.Icon(color='green')).add_to(aed_old_cluster)

# Add new AED locations to the map
for _, row in top_new_aed_locations.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup='New AED', icon=folium.Icon(color='red')).add_to(aed_new_cluster)

# Save the map to an HTML file
belgium_map.save('aed_coverage_map.html')


