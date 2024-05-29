import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from scipy.stats import boxcox
import time
from geopy.geocoders import ArcGIS

pd.options.mode.copy_on_write = True
pd.set_option('display.max_columns', None)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Start time
start_time = time.ctime(int(time.time()))

print(f"Program started at {start_time}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


aed_data = pd.read_csv('DATA/updated_aed_df_with_all_distances.csv')
# grid_data = pd.read_csv('DATA/gird_locations.csv')
# Load Belgium shapefile
belgium_boundary = gpd.read_file('DATA/België.json')

pd.set_option('display.max_columns', None)
# print(aed_data.head())
# print(grid_data.head())

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DATA PREPARATION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# print(aed_data.columns)
# print(grid_data.columns)

# Combine existing potential locations with new grid locations
'''all_potential_locations = pd.concat([aed_data,
                                     grid_data[
                                         ['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel', 'T3-T0_min',
                                          'AED', 'Ambulance', 'Mug', 'Occasional_Permanence',
                                          'distance_to_aed', 'distance_to_ambulance',
                                          'distance_to_mug']]])'''

# Check for missing values
interventions_data = aed_data[aed_data['Intervention'] == 1]
print(interventions_data.isnull().sum())
print(len(interventions_data))

# Setting the style for the plots
sns.set(style="whitegrid")

# Create a figure and a grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

# Plot histogram for response times
sns.histplot(interventions_data['T3-T0_min'], bins=100, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Response Times')
axes[0, 0].set_xlabel('Response Time (minutes)')
axes[0, 0].set_ylabel('Frequency')

# Plot histogram for latitude
sns.histplot(interventions_data['Latitude'], bins=100, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Latitudes')
axes[0, 1].set_xlabel('Latitude')
axes[0, 1].set_ylabel('Frequency')

# Plot histogram for longitude
sns.histplot(interventions_data['Longitude'], bins=100, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Longitudes')
axes[1, 0].set_xlabel('Longitude')
axes[1, 0].set_ylabel('Frequency')

# Plot histogram for distance to the closest AED
sns.histplot(interventions_data['distance_to_aed'], bins=100, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Distances to the Closest AED')
axes[1, 1].set_xlabel('Distance to closest AED')
axes[1, 1].set_ylabel('Frequency')

# Plot histogram for distance to the closest ambulance
sns.histplot(interventions_data['distance_to_ambulance'], bins=100, kde=True, ax=axes[2, 0])
axes[2, 0].set_title('Distribution of Distances to the Closest Ambulance Location')
axes[2, 0].set_xlabel('Distance to closest Ambulance location')
axes[2, 0].set_ylabel('Frequency')

# Plot histogram for distance to the closest Mug
sns.histplot(interventions_data['distance_to_mug'], bins=100, kde=True, ax=axes[2, 1])
axes[2, 1].set_title('Distribution of Distances to the Closest Mug Location')
axes[2, 1].set_xlabel('Distance to closest Mug location')
axes[2, 1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Only check Response times for intervention locations (longer than 12 min)
fig, ax = plt.subplots(figsize=(12, 8))
scatter = sns.scatterplot(
    data=interventions_data[interventions_data['T3-T0_min'] > 12],
    x='Longitude',
    y='Latitude',
    hue='distance_to_aed', palette='YlOrRd', hue_norm=(1500, 6000),
    size='T3-T0_min',
    sizes=(20, 200), legend='auto', ax=ax)
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
ax.set_title('Response Times for previous Intervention Locations')
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
features_for_imputation = interventions_data[['Latitude', 'Longitude', 'distance_to_ambulance',
                                              'distance_to_mug', 'T3-T0_min']]

features_for_imputation['T3-T0_min'] = features_for_imputation['T3-T0_min'].values
features_for_imputation = pd.DataFrame(features_for_imputation,
                                       columns=['Latitude', 'Longitude', 'distance_to_ambulance',
                                                'distance_to_mug', 'T3-T0_min'])
# print(features_for_imputation.head())

# Initialize the KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)
# Perform KNN Imputation
imputed_values = knn_imputer.fit_transform(features_for_imputation)
# Create a dataframe with the imputed values
imputed_data = pd.DataFrame(imputed_values, columns=['Latitude', 'Longitude', 'distance_to_ambulance',
                                                     'distance_to_mug', 'T3-T0_min'])
print(imputed_data.head())
print(interventions_data.head())

# Assign the imputed values back to the original dataframe
interventions_data['T3-T0_min'] = imputed_data['T3-T0_min']

print('------------- Before removing duplicates -------------')
# Verify the imputation
# Check for missing values
print(interventions_data.isnull().sum())
print(len(interventions_data))

interventions_data.drop_duplicates(inplace=True)

print('------------- After removing duplicates -------------')
# Verify the imputation
# Check for missing values
print(interventions_data.isnull().sum())
print(len(interventions_data))

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HEATMAP OF INCIDENT DENSITY
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 8))
# kernel density estimate (KDE) plot

kde = sns.kdeplot(data=interventions_data, x='Longitude', y='Latitude', cmap='Reds',
                  fill=True, ax=ax, cbar=True)

belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
ax.set_title('Heatmap of Incident Density')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

# Create a GeoDataFrame from interventions data
gdf_interventions = gpd.GeoDataFrame(
    interventions_data,
    geometry=gpd.points_from_xy(interventions_data.Longitude, interventions_data.Latitude)
)
# Filter out invalid geometries
gdf_interventions = gdf_interventions[gdf_interventions['geometry'].is_valid]
print(len(gdf_interventions))
print(len(interventions_data))
# Create a grid over the study area
xmin, ymin, xmax, ymax = gdf_interventions.total_bounds
cell_size = 0.03  # Adjust the cell size as needed
grid_cells = []
for x0 in np.arange(xmin, xmax + cell_size, cell_size):
    for y0 in np.arange(ymin, ymax + cell_size, cell_size):
        x1 = x0 - cell_size
        y1 = y0 + cell_size
        grid_cells.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'])
# Filter out invalid geometries
grid = grid[grid['geometry'].is_valid]
# Spatial join the grid with the intervention points
joined = gpd.sjoin(gdf_interventions, grid, how='left')
# Count incidents in each grid cell
incident_counts = joined.groupby('index_right').size()
# Map incidents counts to the grid
grid['incident_count'] = np.nan
grid.loc[incident_counts.index, 'incident_count'] = incident_counts.values
print(grid['incident_count'].value_counts())

# Perform a spatial join to add the incident_count to the interventions data
gdf_interventions_with_incident_count = gpd.sjoin(gdf_interventions, grid[['geometry', 'incident_count']], how='left')

# Drop the geometry column if it's not needed
gdf_interventions_with_incident_count = gdf_interventions_with_incident_count.drop(columns='geometry')

print(gdf_interventions_with_incident_count.head())

# gdf_interventions.to_csv('DATA/interventions_with_densities.csv', index=False)

# Plot the incident density
fig, ax = plt.subplots(figsize=(12, 8))
grid.plot(column='incident_count', ax=ax, cmap='coolwarm', edgecolor='k', legend=True)
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
ax.set_title('Incident Density Grid')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RESPONSE TIME ANALYSIS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Only check Response times longer then 12 minutes (According to literature waiting longer ten 10 minutes could be life threatening)

fig, ax = plt.subplots(figsize=(12, 8))
scatter = sns.scatterplot(
    data=gdf_interventions_with_incident_count[gdf_interventions_with_incident_count['T3-T0_min'] > 12],
    x='Longitude', y='Latitude',
    hue='distance_to_aed', palette='YlOrRd', hue_norm=(1500, 6000),
    size='T3-T0_min',
    sizes=(20, 200), legend='auto', ax=ax)
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
ax.set_title('Response Times by Location')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IDENTIFYING HIGH-RISK AREAS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Define high-risk based on response time and incident frequency
response_time_threshold = 12  # minutes
incident_density_threshold = 1  # more than one previous intervention on that location

# Identify high-risk areas
high_risk_areas = gdf_interventions_with_incident_count[
    (gdf_interventions_with_incident_count['T3-T0_min'] > response_time_threshold) &
    (gdf_interventions_with_incident_count['incident_count'] > incident_density_threshold)
    ]

print('------------- Unique values and amounts -------------')
print(gdf_interventions_with_incident_count['incident_count'].value_counts())

# Visualise high-risk areas
fig, ax = plt.subplots(figsize=(12, 8))
scatter = sns.scatterplot(data=high_risk_areas,
                          x='Longitude', y='Latitude',
                          size='incident_count', hue='T3-T0_min', palette='coolwarm',
                          sizes=(20, 200), legend='auto', ax=ax)
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
ax.set_title('High-Risk Areas based on Response Time and Incident Frequency')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()  # You can see a lot of cardiac arrests in Antwerp, Brussels and Luik (and also on the lin Bergen, Charleroi, Namen)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CLUSTERING HIGH RISK AREAS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Identify high-risk areas to identify hotspots or highly concentrated high risk areas
high_risk_areas = high_risk_areas.dropna(subset=['Latitude', 'Longitude'])

# Elbow Method
wcss = []
K = range(1, 41)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(high_risk_areas[['Latitude', 'Longitude']])
    wcss.append(kmeans.inertia_)

# Plot WCSS
plt.figure(figsize=(10, 6))
plt.plot(K, wcss, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares (WCSS)')
plt.title('Elbow Plot to Determine Optimal Number of Clusters')
plt.show()

# Silhouette Method
silhouette_scores = []
K = range(2, 41)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(high_risk_areas[['Latitude', 'Longitude']])
    silhouette_scores.append(silhouette_score(high_risk_areas[['Latitude', 'Longitude']], labels))

# Plot Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')  # should be maximised
plt.title('Silhouette Plot to Determine Optimal Number of Clusters')
plt.show()

# Perform clustering
num_clusters = 15  # Adjust this number as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
high_risk_areas['cluster'] = kmeans.fit_predict(high_risk_areas[['Latitude', 'Longitude']])

# Determine new AED locations (cluster centers)
new_aed_locations = pd.DataFrame(kmeans.cluster_centers_, columns=['Latitude', 'Longitude'])
print('New AED Locations: \n', new_aed_locations)

# Plot clusters
fig, ax = plt.subplots(figsize=(12, 8))
# Plot the high-risk areas and color by cluster
scatter = sns.scatterplot(
    data=high_risk_areas,
    x='Longitude', y='Latitude',
    hue='cluster', palette='tab20',
    size='incident_count',
    sizes=(20, 200), legend=False, ax=ax)
# Overlay the Belgium boundary
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')

# Overlay the proposed AED locations
new_aed_gdf = gpd.GeoDataFrame(new_aed_locations, geometry=gpd.points_from_xy(new_aed_locations.Longitude, new_aed_locations.Latitude))
new_aed_gdf.plot(ax=ax, marker='x', color='blue', markersize=100, label='Proposed AED Locations')

ax.set_title('High-Risk Areas Clusters and Proposed AED Locations')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.show()


# Plot new AED locations
fig, ax = plt.subplots(figsize=(10, 10))
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
new_aed_gdf = gpd.GeoDataFrame(new_aed_locations,
                               geometry=gpd.points_from_xy(new_aed_locations.Longitude, new_aed_locations.Latitude))
new_aed_gdf.plot(ax=ax, marker='x', color='blue', markersize=100, label='Proposed AED Locations')
plt.legend()
plt.show()

# Merge with indicator
merged_df = new_aed_locations.merge(aed_data, on=['Latitude', 'Longitude'], how='left', indicator=True)

# Get the rows that are present in both DataFrames
common_rows = merged_df[merged_df['_merge'] == 'both']

# Display the common rows
print('These are the common rows: \n', common_rows)  # No common rows


'''
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CONVERT NEW LOCATIONS TO ADRESSES
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def reverse_geocode(latitude, longitude):
    geolocator = ArcGIS()
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    return location.address


# Apply reverse geocoding to each row in the dataset
new_aed_locations['Address'] = new_aed_locations.apply(lambda row: reverse_geocode(row['Latitude'], row['Longitude']),
                                                       axis=1)

print(new_aed_locations)
'''
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# End time
end_time = time.ctime(int(time.time()))
print(f'Program ended at {end_time}')
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
