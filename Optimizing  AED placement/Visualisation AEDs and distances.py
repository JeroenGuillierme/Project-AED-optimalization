import pandas as pd
from scipy.spatial import KDTree
import numpy as np
import folium
from folium.plugins import MarkerCluster

# Visual of closest AED location of previous Intervention locations
# Load your dataset
data = pd.read_csv(
    'C:/Users/Admin/Documents/GitHub/Project-AED-optimalization/DATA/updated_aed_df_with_all_distances.csv')

# Create a base map centered around Belgium
belgium_center = [50.8503, 4.3517]
map_ = folium.Map(location=belgium_center, zoom_start=8)

# Filter for necessary locations
aed_locations = data[data['AED'] == 1][['Latitude', 'Longitude']]
# Filter ambulance locations with occasional permanence
ambulance_locations = data[(data['Ambulance'] == 1) & (data['Occasional_Permanence'] == 1)][
    ['Latitude', 'Longitude']].dropna().reset_index(drop=True)
# Filter mug locations with occasional permanence
mug_locations = data[(data['Mug'] == 1) & (data['Occasional_Permanence'] == 1)][
    ['Latitude', 'Longitude']].dropna().reset_index(drop=True)
intervention_locations = data[data['Intervention'] == 1][
    ['Latitude', 'Longitude', 'distance_to_aed', 'distance_to_ambulance', 'distance_to_mug']].dropna()

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
