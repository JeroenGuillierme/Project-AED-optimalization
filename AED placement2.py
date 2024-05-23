import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
import logging
import time
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Set up logging
logging.basicConfig(filename='program.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log start time
start_time = time.ctime(int(time.time()))
logging.info("Program started.")
print(f"Program started at {start_time}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load the Belgium boundary shapefile
belgium_boundary = gpd.read_file('DATA/België.json')

# Generate a grid of potential new locations within the Belgium boundary
minx, miny, maxx, maxy = belgium_boundary.total_bounds
grid_points = []
grid_size = 0.01  # Grid size of 0.01 degrees, approximately 1 km

# Create a grid of points within the bounding box
x_coords = np.arange(minx, maxx, grid_size)
y_coords = np.arange(miny, maxy, grid_size)

for x in x_coords:
    for y in y_coords:
        point = Point(x, y)
        # Check if the point is within the Belgium boundary
        if belgium_boundary.contains(point).any():
            grid_points.append(point)

# Create a GeoDataFrame for the grid points
grid_gdf = gpd.GeoDataFrame(grid_points, columns=['geometry'])
grid_gdf['Latitude'] = grid_gdf.geometry.y
grid_gdf['Longitude'] = grid_gdf.geometry.x

# Load your original dataset
df = pd.read_csv('DATA/updated_aed_df_with_all_distances.csv')

# Fill NaN values in Eventlevel and Occasional_Permanence with -1
df['Eventlevel'].fillna(-1, inplace=True)
df['Occasional_Permanence'].fillna(-1, inplace=True)

# Remove rows with NaN values in Latitude or Longitude0
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

'''
# Log transform right skewed numerical columns before scaling (np.log1p due to the zero values)
df['distance_to_aed'] = np.log1p(df['distance_to_aed'])
df['distance_to_ambulance'] = np.log1p(df['distance_to_ambulance'])
df['distance_to_mug'] = np.log1p(df['distance_to_mug'])
df['T3-T0_min'] = np.log1p(df['T3-T0_min'])
'''

# Normalize the numerical columns using RobustScaler
scaler = RobustScaler()
df[['distance_to_aed', 'distance_to_ambulance', 'distance_to_mug', 'T3-T0_min']] = scaler.fit_transform(
    df[['distance_to_aed', 'distance_to_ambulance', 'distance_to_mug', 'T3-T0_min']])

# Assign weights based on assumption
weights = {
    'Intervention': 0.3,
    'CAD9': 0.3,
    'Eventlevel': 0.5,
    'distance_to_aed': 0.8,
    'distance_to_ambulance': 0.65,
    'distance_to_mug': 0.65,
    'T3-T0_min': 0.90
}

# Calculate a weighted score for each location
df['score'] = (df['Intervention'] * weights['Intervention'] + df['CAD9'] * weights['CAD9'] +
               df['Eventlevel'] * weights['Eventlevel'] + df['distance_to_aed'] * weights['distance_to_aed'] +
               df['distance_to_ambulance'] * weights['distance_to_ambulance'] + df['distance_to_mug'] * weights['distance_to_mug'] +
               df['T3-T0_min'] * weights['T3-T0_min'])

# Filter out locations with existing AEDs
potential_locations = df[df['AED'] == 0].copy() # Use a copy!!

# Calculate distances for grid locations
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in km
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(
        dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 1000  # Distance in meters

def calculate_nearest_distance(lat, lon, coords):
    distances = np.array([haversine(lat, lon, coord_lat, coord_lon) for coord_lat, coord_lon in coords])
    return np.min(distances)

# Calculate distances to the nearest AED, ambulance, and MUG for grid locations
aed_coords = df[df['AED'] == 1][['Latitude', 'Longitude']].dropna().to_numpy()
ambulance_coords = df[df['Ambulance'] == 1][['Latitude', 'Longitude']].dropna().to_numpy()
mug_coords = df[df['Mug'] == 1][['Latitude', 'Longitude']].dropna().to_numpy()

grid_gdf['distance_to_aed'] = grid_gdf.apply(lambda row: calculate_nearest_distance(row['Latitude'], row['Longitude'], aed_coords), axis=1)
grid_gdf['distance_to_ambulance'] = grid_gdf.apply(lambda row: calculate_nearest_distance(row['Latitude'], row['Longitude'], ambulance_coords), axis=1)
grid_gdf['distance_to_mug'] = grid_gdf.apply(lambda row: calculate_nearest_distance(row['Latitude'], row['Longitude'], mug_coords), axis=1)
grid_gdf['T3-T0_min'] = np.nan

'''
# Log transform and normalize the distances for grid locations
grid_gdf['distance_to_aed'] = np.log1p(grid_gdf['distance_to_aed'])
grid_gdf['distance_to_ambulance'] = np.log1p(grid_gdf['distance_to_ambulance'])
grid_gdf['distance_to_mug'] = np.log1p(grid_gdf['distance_to_mug'])
'''

grid_gdf[['distance_to_aed', 'distance_to_ambulance', 'distance_to_mug', 'T3-T0_min']] = scaler.transform(
    grid_gdf[['distance_to_aed', 'distance_to_ambulance', 'distance_to_mug', 'T3-T0_min']])

# Assign a neutral score and calculate coverage for grid locations
grid_gdf['score'] = 0
incident_coords = df[df['Intervention'] == 1][['Latitude', 'Longitude']].dropna().to_numpy()

def evaluate_coverage(location, incident_coords, coverage_radius):
    distances = np.array([haversine(location['Latitude'], location['Longitude'], inc_lat, inc_lon)
                          for inc_lat, inc_lon in incident_coords])
    return np.sum(distances <= coverage_radius)

coverage_radius = 1000  # Assuming a coverage radius of 1000 meters

grid_gdf['coverage'] = grid_gdf.apply(lambda row: evaluate_coverage(row, incident_coords, coverage_radius), axis=1)
grid_gdf['final_score'] = grid_gdf['score'] + grid_gdf['coverage']

# Combine existing potential locations with new grid locations
all_potential_locations = pd.concat([potential_locations, grid_gdf[['Latitude', 'Longitude', 'score', 'coverage', 'final_score']]], ignore_index=True)

# Determine the optimal number of clusters using the Elbow method
def determine_optimal_clusters(data, max_clusters=40):
    silhouette_scores = []
    distortions = []
    for i in range(1, max_clusters + 1):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(data)
        distortions.append(km.inertia_)
        silhouette_scores.append(silhouette_score(data, km.labels_))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal Number of Clusters')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score For Optimal Number of Clusters')
    plt.show()

    optimal_clusters = np.argmax(silhouette_scores) + 2  # Add 2 because range started from 2
    return optimal_clusters


optimal_clusters = determine_optimal_clusters(all_potential_locations[['Latitude', 'Longitude']])
print(f"Optimal number of clusters: {optimal_clusters}")


# Perform KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=45)
kmeans.fit(all_potential_locations[['Latitude', 'Longitude']])
all_potential_locations['cluster'] = kmeans.labels_

# Select the top new AED location in each cluster based on the final score
top_new_aed_locations = all_potential_locations.groupby('cluster').apply(lambda group: group.nlargest(1, 'final_score')[['Latitude', 'Longitude']]).reset_index(drop=True)

# Save new AED locations to a CSV file
top_new_aed_locations.to_csv('DATA/new_aed_locations.csv', index=False)

# Optionally, visualize the new AED locations on a map
belgium_map = folium.Map(location=[50.85, 4.35], zoom_start=8)

# Create marker clusters for better visualization
aed_old_cluster = MarkerCluster(name='Old Locations').add_to(belgium_map)
aed_new_cluster = MarkerCluster(name='New AED Locations').add_to(belgium_map)

# Add existing AED locations to the map
for _, row in df[df['AED'] == 1].iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup='Existing AED', icon=folium.Icon(color='green')).add_to(aed_old_cluster)

# Add new AED locations to the map
for _, row in top_new_aed_locations.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup='New AED', icon=folium.Icon(color='red')).add_to(aed_new_cluster)

# Save the map to an HTML file
belgium_map.save('aed_coverage_map.html')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Log end time
end_time = time.time()
total_runtime_minutes = (end_time - start_time) / 60  # Convert seconds to minutes
logging.info("Program completed. Total runtime: {:.2f} minutes".format(total_runtime_minutes))
print("Program completed. Total runtime: {:.2f} minutes".format(total_runtime_minutes))
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
