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
# Start time
start_time = time.ctime(int(time.time()))

print(f"Program started at {start_time}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

url = 'https://raw.githubusercontent.com/JeroenGuillierme/Project-AED-optimalization/main/DATA/'

# Load the Belgium boundary shapefile
belgium_boundary = gpd.read_file(f'{url}Belgi%C3%AB.json')

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

grid_gdf['distance_to_aed'] = grid_gdf.apply(
    lambda row: calculate_nearest_distance(row['Latitude'], row['Longitude'], aed_coords), axis=1)
grid_gdf['distance_to_ambulance'] = grid_gdf.apply(
    lambda row: calculate_nearest_distance(row['Latitude'], row['Longitude'], ambulance_coords), axis=1)
grid_gdf['distance_to_mug'] = grid_gdf.apply(
    lambda row: calculate_nearest_distance(row['Latitude'], row['Longitude'], mug_coords), axis=1)
grid_gdf['T3-T0_min'] = np.nan

'''
# Log transform and normalize the distances for grid locations
grid_gdf['distance_to_aed'] = np.log1p(grid_gdf['distance_to_aed'])
grid_gdf['distance_to_ambulance'] = np.log1p(grid_gdf['distance_to_ambulance'])
grid_gdf['distance_to_mug'] = np.log1p(grid_gdf['distance_to_mug'])
'''

'''
grid_gdf[['distance_to_aed', 'distance_to_ambulance', 'distance_to_mug', 'T3-T0_min']] = scaler.transform(
    grid_gdf[['distance_to_aed', 'distance_to_ambulance', 'distance_to_mug', 'T3-T0_min']])
'''

'''
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
'''

grid_gdf['Intervention'] = 0
grid_gdf['CAD9'] = 0
grid_gdf['Eventlevel'] = -1
grid_gdf['AED'] = 0
grid_gdf['Ambulance'] = 0
grid_gdf['Mug'] = 0
grid_gdf['Occasional_Permanence'] = 0
'''
# Combine existing potential locations with new grid locations
all_potential_locations = pd.concat([df,
                                     grid_gdf[['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel',
                                               'AED', 'Ambulance', 'Mug', 'Occasional_Permanence',
                                               'distance_to_aed', 'distance_to_ambulance',
                                               'distance_to_mug']]], ignore_index=True)
'''

grid_gdf.to_csv('DATA/gird_locations.csv', index=False)  # ongeveer 24.000 locaties toegevoegd

print(grid_gdf.head())
print(len(grid_gdf))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# End time
end_time = time.ctime(int(time.time()))
print(f'Program ended at {end_time}')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
