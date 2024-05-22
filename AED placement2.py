import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from scipy.spatial.distance import cdist, pdist, squareform

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

# Remove rows with NaN values in Latitude or Longitude
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Normalize the numerical columns using MinMaxScaler
scaler = MinMaxScaler()
df[['distance_to_aed', 'distance_to_ambulance', 'distance_to_mug']] = scaler.fit_transform(
    df[['distance_to_aed', 'distance_to_ambulance', 'distance_to_mug']])

# Calculate a weighted score for each location
df['score'] = (df['Intervention'] * 0.00001 + df['CAD9'] * 0.00450788 + df['Eventlevel'] * 0.00450788 +
               df['distance_to_aed'] * 0.26576536 + df['distance_to_ambulance'] * 0.63171176 + df['distance_to_mug'] * 0.03274299)

# Filter out locations with existing AEDs
potential_locations = df[df['AED'] == 0].copy()

# Define a function to calculate the Haversine distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in km
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(
        dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 1000  # Distance in meters

# Calculate distances from each location to the nearest existing AED using Haversine distance
aed_coords = df[df['AED'] == 1][['Latitude', 'Longitude']].dropna().to_numpy()

def calculate_nearest_aed_distance(lat, lon, aed_matrix):
    # Calculate the Haversine distance from the given point to all AED points
    distances = np.array([haversine(lat, lon, aed_lat, aed_lon) for aed_lat, aed_lon in aed_matrix])
    return np.min(distances)

potential_locations['distance_to_nearest_aed'] = potential_locations.apply(
    lambda row: calculate_nearest_aed_distance(row['Latitude'], row['Longitude'], aed_coords), axis=1)

# Determine the coverage radius based on incident distances
incident_coords = df[df['Intervention'] == 1][['Latitude', 'Longitude']].dropna()
incident_distances = potential_locations['distance_to_nearest_aed'].dropna()

# Ensure there are no NaN values before calculating the coverage radius
if not incident_distances.empty:
    # Plot a histogram of distances to the nearest AED
    plt.hist(incident_distances, bins=30)
    plt.xlabel('Distance to Nearest AED (meters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances to Nearest AED')
    plt.show()

    # Calculate the 90th percentile of the incident distances
    coverage_radius = np.percentile(incident_distances, 90)
    print(f"Suggested coverage radius: {coverage_radius} meters")
else:
    print("No valid incident distances available for calculating the coverage radius.")
    coverage_radius = 1000  # Default to 1000 meters if no valid distances are found

# Evaluate coverage for each potential AED location
def evaluate_coverage(location, incident_coords, coverage_radius):
    # Calculate the Haversine distance from the location to all incident coordinates
    distances = np.array([haversine(location['Latitude'], location['Longitude'], inc_lat, inc_lon)
                          for inc_lat, inc_lon in incident_coords])
    return np.sum(distances <= coverage_radius)

incident_matrix = incident_coords.to_numpy()
potential_locations['coverage'] = potential_locations.apply(
    lambda row: evaluate_coverage(row, incident_matrix, coverage_radius), axis=1)

# Combine score and coverage for final ranking
potential_locations['final_score'] = potential_locations['score'] + potential_locations['coverage']

# Integrate new potential locations from the grid
grid_gdf['score'] = 0  # New locations start with a neutral score
grid_gdf['coverage'] = grid_gdf.apply(lambda row: evaluate_coverage(row, incident_matrix, coverage_radius), axis=1)
grid_gdf['final_score'] = grid_gdf['score'] + grid_gdf['coverage']

# Combine existing potential locations with new grid locations
all_potential_locations = pd.concat([potential_locations, grid_gdf[['Latitude', 'Longitude', 'final_score']]],
                                    ignore_index=True)

# Determine the optimal number of clusters using the Elbow method
def determine_optimal_clusters(data, max_clusters=40):
    distortions = []
    for i in range(1, max_clusters + 1):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(data)
        distortions.append(km.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal Number of Clusters')
    plt.show()

    # Find the elbow point in the distortions curve
    diff = np.diff(distortions)
    second_diff = np.diff(diff)
    optimal_clusters = np.argmax(second_diff) + 2  # +2 because of the two diff steps
    return optimal_clusters

# Automatically determine the optimal number of clusters
optimal_clusters = determine_optimal_clusters(all_potential_locations[['Latitude', 'Longitude']])
print(f"Optimal number of clusters: {optimal_clusters}")

# Perform KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(all_potential_locations[['Latitude', 'Longitude']])
all_potential_locations['cluster'] = kmeans.labels_

# Select the top new AED location in each cluster based on the final score
top_new_aed_locations = all_potential_locations.groupby('cluster').apply(
    lambda group: group.nlargest(1, 'final_score')[['Latitude', 'Longitude']]).reset_index(drop=True)

# Save new AED locations to a CSV file
top_new_aed_locations.to_csv('new_aed_locations.csv', index=False)

# Optionally, visualize the new AED locations on a map
belgium_map = folium.Map(location=[50.85, 4.35], zoom_start=8)

# Create marker clusters for better visualization
aed_old_cluster = MarkerCluster(name='Old Locations').add_to(belgium_map)
aed_new_cluster = MarkerCluster(name='New AED Locations').add_to(belgium_map)

# Add existing AED locations to the map
for _, row in df[df['AED'] == 1].iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup='Existing AED', icon=folium.Icon(color='green')).add_to(
        aed_old_cluster)

# Add new AED locations to the map
for _, row in top_new_aed_locations.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup='New AED', icon=folium.Icon(color='red')).add_to(
        aed_new_cluster)

# Save the map to an HTML file
belgium_map.save('aed_coverage_map.html')
