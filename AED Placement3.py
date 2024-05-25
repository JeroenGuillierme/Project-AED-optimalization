import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import cKDTree
import time

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Start time
start_time = time.ctime(int(time.time()))

print(f"Program started at {start_time}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


aed_data = pd.read_csv('DATA/updated_aed_df_with_all_distances.csv')
grid_data = pd.read_csv('DATA/gird_locations.csv')
# Load Belgium shapefile
belgium_boundary = gpd.read_file('DATA/België.json')

pd.set_option('display.max_columns', None)
print(aed_data.head())
print(grid_data.head())

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DATA PREPARATION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print(aed_data.columns)
print(grid_data.columns)

# Combine existing potential locations with new grid locations
all_potential_locations = pd.concat([aed_data,
                                     grid_data[
                                         ['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel', 'T3-T0_min',
                                          'AED', 'Ambulance', 'Mug', 'Occasional_Permanence',
                                          'distance_to_aed', 'distance_to_ambulance',
                                          'distance_to_mug']]])

# Check for missing values
print(all_potential_locations.isnull().sum())
print(len(all_potential_locations))

# Setting the style for the plots
sns.set(style="whitegrid")

# Create a figure and a grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

# Plot histogram for response times
sns.histplot(all_potential_locations['T3-T0_min'], bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Response Times')
axes[0, 0].set_xlabel('Response Time (minutes)')
axes[0, 0].set_ylabel('Frequency')

# Plot histogram for latitude
sns.histplot(all_potential_locations['Latitude'], bins=30, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Latitudes')
axes[0, 1].set_xlabel('Latitude')
axes[0, 1].set_ylabel('Frequency')

# Plot histogram for longitude
sns.histplot(all_potential_locations['Longitude'], bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Longitudes')
axes[1, 0].set_xlabel('Longitude')
axes[1, 0].set_ylabel('Frequency')

# Plot histogram for distance to the closest AED
sns.histplot(all_potential_locations['distance_to_aed'], bins=30, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Distances to the Closest AED')
axes[1, 1].set_xlabel('Distance to closest AED')
axes[1, 1].set_ylabel('Frequency')

# Plot histogram for distance to the closest ambulance
sns.histplot(all_potential_locations['distance_to_ambulance'], bins=30, kde=True, ax=axes[2, 0])
axes[2, 0].set_title('Distribution of Distances to the Closest Ambulance Location')
axes[2, 0].set_xlabel('Distance to closest Ambulance location')
axes[2, 0].set_ylabel('Frequency')

# Plot histogram for distance to the closest Mug
sns.histplot(all_potential_locations['distance_to_mug'], bins=30, kde=True, ax=axes[2, 1])
axes[2, 1].set_title('Distribution of Distances to the Closest Mug Location')
axes[2, 1].set_xlabel('Distance to closest Mug location')
axes[2, 1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Only check Response times for intervention locations (longer than 25 min)
fig, ax = plt.subplots(figsize=(12, 8))
scatter = sns.scatterplot(data=all_potential_locations[all_potential_locations['T3-T0_min'] > 25], x='Longitude',
                          y='Latitude',
                          hue='T3-T0_min', palette='coolwarm',
                          size='T3-T0_min',
                          sizes=(20, 200), legend='brief', ax=ax)
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
ax.set_title('Response Times for previous Intervention Locations')
# Customize the legend
handles, labels = scatter.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:], title='Response Time (min)', loc='upper right',
          bbox_to_anchor=(1, 1))
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# REPLACING MISSING RESPONSE TIMES FROM GRID LOCATIONS WITH (KNN) IMPUTED VALUES BASED ON LOCATION AND DISTANCES
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Assign response time to added locations in the dataset
# Using KNN Imputer
'''
Feature Selection: Selected Latitude, Longitude, and T3-T0_min for imputation.
Scaling: Scaled Latitude and Longitude using StandardScaler. 
            This ensures that these features contribute equally to the distance calculation in the KNN Imputer.
KNN Imputation: KNN Imputer applied to the scaled data to fill in the missing values.
Inverse Transform: After imputation, the scaled features were inverse transformed back to their original scale 
                    and reassigned the imputed response times (T3-T0_min) back to the original dataframe.
'''
# Select features for imputation
features_for_imputation = all_potential_locations[['Latitude', 'Longitude', 'distance_to_ambulance',
                                                   'distance_to_mug', 'T3-T0_min']]
# Separate features to scale
features_to_scale = features_for_imputation[['Latitude', 'Longitude', 'distance_to_ambulance',
                                             'distance_to_mug']]
# Initialize the scaler
# IMPORTANT: IS THIS THE CORRECT SCALER?? => MOST OF THE VARIABLES ARE RIGHT SKEWED
scaler = StandardScaler()
# Fit and transform the features to scale
scaled_features = scaler.fit_transform(features_to_scale)
# Combine scaled features with the original dataframe
scaled_data = pd.DataFrame(scaled_features,
                           columns=['Latitude_scaled', 'Longitude_scaled', 'distance_to_ambulance_scaled',
                                    'distance_to_mug_scaled'])
scaled_data['T3-T0_min'] = features_for_imputation['T3-T0_min'].values
# Initialize the KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)
# Perform KNN Imputation
imputed_values = knn_imputer.fit_transform(scaled_data)
# Create a dataframe with the imputed values
imputed_data = pd.DataFrame(imputed_values, columns=scaled_data.columns)
# Inverse transform the scaled features back to original scale
imputed_data[
    ['Latitude_scaled', 'Longitude_scaled', 'distance_to_ambulance', 'distance_to_mug']] = scaler.inverse_transform(
    imputed_data[['Latitude_scaled', 'Longitude_scaled', 'distance_to_ambulance_scaled', 'distance_to_mug_scaled']])
imputed_data.rename(columns={'Latitude_scaled': 'Latitude', 'Longitude_scaled': 'Longitude',
                             'distance_to_ambulance': 'distance_to_ambulance',
                             'distance_to_mug': 'distance_to_mug'}, inplace=True)
# Assign the imputed values back to the original dataframe
all_potential_locations['T3-T0_min'] = imputed_data['T3-T0_min']

# Verify the imputation
print(all_potential_locations.isnull().sum())

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IDENTIFYING HIGH-RISK AREAS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plotting response times => high response times are problematic
plt.figure(figsize=(10, 6))
sns.histplot(all_potential_locations['T3-T0_min'], bins=30, kde=True)
plt.title('Distribution of Response Times after KNN Imputing')
plt.xlabel('Response Time (minutes)')
plt.ylabel('Frequency')
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CURRENT AED-COVERAGE
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# dataframe `current_aeds` with AED locations
current_aeds = all_potential_locations[all_potential_locations['AED'] == 1]
print(len(current_aeds))
# dataframe `incidents` with intervention locations
incidents = all_potential_locations[all_potential_locations['Intervention'] == 1]

# Create a GeoDataFrame for the incidents
gdf_incidents = gpd.GeoDataFrame(
    all_potential_locations,
    geometry=gpd.points_from_xy(all_potential_locations.Longitude, all_potential_locations.Latitude)
)
print(gdf_incidents.head())
# Plotting existing  incident locations
fig, ax = plt.subplots(figsize=(10, 10))
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
gdf_incidents[gdf_incidents['Intervention'] == 1].plot(ax=ax, marker='o', color='red', markersize=5, label='Incidents')
plt.legend()
plt.show()

# Plotting existing AED locations
fig, ax = plt.subplots(figsize=(10, 10))
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
gdf_incidents[gdf_incidents['AED'] == 1].plot(ax=ax, marker='x', color='blue', markersize=50, label='Current AEDs')
plt.legend()
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DEMAND ANALYSIS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Calculating frequency of incidents by location
incident_frequency = gdf_incidents.groupby(['Latitude', 'Longitude']).size().reset_index(name='count')
# Join frequency back to the GeoDataFrame
gdf_incidents = gdf_incidents.merge(incident_frequency, on=['Latitude', 'Longitude'])

print(gdf_incidents.head())

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HEATMAP OF INCIDENT DENSITY
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 8))
# kernel density estimate (KDE) plot

kde = sns.kdeplot(data=gdf_incidents[gdf_incidents['Intervention'] == 1], x='Longitude', y='Latitude', cmap='Reds',
                  fill=True, ax=ax, cbar=True)

belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
ax.set_title('Heatmap of Incident Density')
'''# Create a colorbar
cbar = fig.colorbar(kde.collections[0], ax=ax)
cbar.set_label('Density')'''

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RESPONSE TIME ANALYSIS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Only check Response times longer then 25 minutes (otherwise plot overfull and not clear anymore)
fig, ax = plt.subplots(figsize=(12, 8))
scatter = sns.scatterplot(data=gdf_incidents[gdf_incidents['T3-T0_min'] > 25], x='Longitude', y='Latitude',
                          hue='T3-T0_min', palette='coolwarm',
                          size='T3-T0_min',
                          sizes=(20, 200), legend='brief', ax=ax)
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
ax.set_title('Response Times by Location')
# Customize the legend
handles, labels = scatter.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:], title='Response Time (min)', loc='upper right',
          bbox_to_anchor=(1, 1))
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# End time
end_time = time.ctime(int(time.time()))
print(f'Program ended at {end_time}')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
