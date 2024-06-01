import pandas as pd
from scipy.spatial import KDTree
import numpy as np
import folium
from folium.plugins import MarkerCluster


# Load your dataset
data = pd.read_csv('DATA/aed_placement_df.csv')

pd.set_option('display.max_columns', None)


# Haversine formula to calculate distance in meters between two points given in degrees
def haversine(coord1, coord2):
    # Radius of the Earth in meters
    R = 6371000
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


# Function to find the minimum distance to a location
def get_min_distance(row, locations, column_name):
    if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
        return np.nan
    location_coord = (row['Latitude'], row['Longitude'])
    if row[column_name] == 1:
        return 0
    # Query the KDTree for the closest location
    _, idx = KDTree(locations).query(location_coord)
    closest_coord = locations.iloc[idx]
    return haversine(location_coord, closest_coord)


# Filter AED locations
aed_locations = data[data['AED'] == 1][['Latitude', 'Longitude']].dropna().reset_index(drop=True)

# Check if there are any AED locations in the dataset
if not aed_locations.empty:
    # Create a KDTree for AED locations
    aed_tree = KDTree(aed_locations)

    # Function to find the minimum distance to AED
    def get_min_distance_to_aed(row):
        return get_min_distance(row, aed_locations, 'AED')


    # Apply the function to each row
    data['distance_to_aed'] = data.apply(get_min_distance_to_aed, axis=1)
else:
    data['distance_to_aed'] = np.nan


# Calculate distances to ambulance and mug locations
ambulance_locations = data[data['Ambulance'] == 1][['Latitude', 'Longitude']].dropna().reset_index(drop=True)
mug_locations = data[data['Mug'] == 1][['Latitude', 'Longitude']].dropna().reset_index(drop=True)

data['distance_to_ambulance'] = data.apply(get_min_distance, args=(ambulance_locations, 'Ambulance'), axis=1)
data['distance_to_mug'] = data.apply(get_min_distance, args=(mug_locations, 'Mug'), axis=1)


features = ['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel', 'T3-T0_min', 'AED',
            'Ambulance', 'Mug', 'Occasional_Permanence', 'distance_to_aed', 'distance_to_ambulance', 'distance_to_mug']
print(list(data.columns))
print(data[features].isna().sum())
print(data.head())
print(data[data['AED'] == 1])
print(data[data['Ambulance'] == 1])
print(data[data['Mug'] == 1])
print(data[data['CAD9'] == 1])



# Visual of closest AED location of previous Intervention locations

# Create a base map centered around Belgium
belgium_center = [50.8503, 4.3517]
map_ = folium.Map(location=belgium_center, zoom_start=8)

# Filter for necessary locations
aed_locations = data[data['AED'] == 1][['Latitude', 'Longitude']]
# Filter ambulance locations with occasional permanence
ambulance_locations = data[(data['Ambulance'] == 1) & (data['Occasional_Permanence'] == 1)][['Latitude', 'Longitude']].dropna().reset_index(drop=True)
# Filter mug locations with occasional permanence
mug_locations = data[(data['Mug'] == 1) & (data['Occasional_Permanence'] == 1)][['Latitude', 'Longitude']].dropna().reset_index(drop=True)
intervention_locations = data[data['Intervention'] == 1][['Latitude', 'Longitude', 'distance_to_aed', 'distance_to_ambulance', 'distance_to_mug']].dropna()

# Create marker clusters for better visualization
aed_cluster = MarkerCluster(name='AED Locations').add_to(map_)
ambulance_cluster = MarkerCluster(name='Ambulance Locations').add_to(map_)
mug_cluster = MarkerCluster(name='Mug Locations').add_to(map_)
intervention_cluster = MarkerCluster(name='Intervention Locations').add_to(map_)

# Add AED locations to the map
for idx, row in aed_locations.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.Icon(color='green', icon='heart'),
        popup=f'AED Location ({row["Latitude"]}, {row["Longitude"]})'
    ).add_to(aed_cluster)

# Add ambulance locations to the map
for idx, row in ambulance_locations.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.Icon(color='blue', icon='ambulance'),
        popup=f'Ambulance Location ({row["Latitude"]}, {row["Longitude"]})'
    ).add_to(ambulance_cluster)

# Add mug locations to the map
for idx, row in mug_locations.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        icon=folium.Icon(color='orange', icon='coffee'),
        popup=f'Mug Location ({row["Latitude"]}, {row["Longitude"]})'
    ).add_to(mug_cluster)

# Add intervention locations to the map and draw lines to the nearest AED, ambulance, and mug
for idx, row in intervention_locations.iterrows():
    intervention_point = [row['Latitude'], row['Longitude']]
    folium.Marker(
        location=intervention_point,
        icon=folium.Icon(color='red', icon='info-sign'),
        popup=f'Intervention Location ({row["Latitude"]}, {row["Longitude"]})'
    ).add_to(intervention_cluster)

    # Find the nearest mug, ambulance, and AED
    nearest_mug_idx = KDTree(mug_locations).query(intervention_point)[1]
    nearest_ambulance_idx = KDTree(ambulance_locations).query(intervention_point)[1]
    nearest_aed_idx = KDTree(aed_locations).query(intervention_point)[1]

    if not pd.isna(row['distance_to_aed']):
        # Draw a line from intervention to the nearest AED
        folium.PolyLine(
            locations=[intervention_point, aed_locations.iloc[nearest_aed_idx]],
            color='blue',
            tooltip=f'AED Distance: {row["distance_to_aed"]:.2f} meters'
        ).add_to(map_)

    '''
    if not pd.isna(row['distance_to_ambulance']):
        # Draw a line from intervention to the nearest ambulance
        folium.PolyLine(
            locations=[intervention_point, ambulance_locations.iloc[nearest_ambulance_idx]],
            color='purple',
            tooltip=f'Ambulance Distance: {row["distance_to_ambulance"]:.2f} meters'
        ).add_to(map_)

    if not pd.isna(row['distance_to_mug']):
        # Draw a line from intervention to the nearest mug
        folium.PolyLine(
            locations=[intervention_point, mug_locations.iloc[nearest_mug_idx]],
            color='green',
            tooltip=f'Mug Distance: {row["distance_to_mug"]:.2f} meters'
        ).add_to(map_)'''

# Add layer control to toggle visibility of clusters
folium.LayerControl().add_to(map_)
# Save the map to an HTML file and display it
map_.save('aed_interventions_map.html')
map_

# Save the updated dataset
# data.to_csv('DATA/updated_aed_df_with_all_distances.csv', index=False)
