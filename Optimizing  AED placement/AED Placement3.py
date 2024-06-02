import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, Polygon
import time
from geopy.geocoders import ArcGIS

pd.options.mode.copy_on_write = True
pd.set_option('display.max_columns', None)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Start time
start_time = time.ctime(int(time.time()))

print(f"Program started at {start_time}")
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


aed_data = pd.read_csv(
    'C:/Users/Admin/Documents/GitHub/Project-AED-optimalization/DATA/updated_aed_df_with_all_distances.csv')
# grid_data = pd.read_csv('DATA/gird_locations.csv')
# Load Belgium shapefile
belgium_boundary = gpd.read_file('C:/Users/Admin/Documents/GitHub/Project-AED-optimalization/DATA/België.json')
# Load Belgium with regions shapefile
belgium_with_regions_boundary = gpd.read_file(
    'C:/Users/Admin/Documents/GitHub/Project-AED-optimalization/DATA/belgium-with-regions_.geojson')

aed_locations = aed_data[aed_data['AED'] == 1]
print(f'There are currently {len(aed_locations)} AEDs in Belgium')

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

interventions_data.drop_duplicates(inplace=True)  # 2512 duplicates removed

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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ADDING INCIDENT DENSITY TO DATASET
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create a GeoDataFrame from interventions data
gdf_interventions = gpd.GeoDataFrame(
    interventions_data,
    geometry=gpd.points_from_xy(interventions_data.Longitude, interventions_data.Latitude)
)
# Filter out invalid geometries
gdf_interventions = gdf_interventions[gdf_interventions['geometry'].is_valid]
print(len(gdf_interventions))
print(len(interventions_data))


# Function to calculate cell size in decimal degrees based on desired grid area and latitude
def calculate_cell_size(grid_area_km2, latitude_degrees):
    '''
    This function calculates the cell size in degrees based on a given grid size in km² and the latitude coordinate.
    :param grid_area_km2: Wanted grid are in km².
    :param latitude_degrees: The latitude coordinate at which the grid size in degrees need to be calculated.
    :return: The degrees of the cell size used to split Belgium up in different grids of given area in km².
    '''
    # Convert latitude to radians for trigonometric functions
    latitude_rad = np.radians(latitude_degrees)

    # Conversion factor from meters to decimal degrees for latitude
    meters_to_degrees = 1 / (111.32 * 1000 * np.cos(latitude_rad))

    # Calculate width of the square grid cell in meters
    width_meters = np.sqrt(grid_area_km2 * 1000000)

    # Calculate cell size in decimal degrees
    cell_size_degrees = width_meters * meters_to_degrees

    return cell_size_degrees


# Example parameters
grid_area_km2 = 3  # Desired grid area in km²
# Calculate mean latitude from gdf_interventions
mean_latitude = gdf_interventions.geometry.centroid.y.mean()
print(f'The mean latitude is: {mean_latitude}')
# Calculate cell size in decimal degrees
cell_size_degrees = calculate_cell_size(grid_area_km2, mean_latitude)
print(f'The cell size in degrees is: {cell_size_degrees}')
# Create a grid over the study area
xmin, ymin, xmax, ymax = gdf_interventions.total_bounds
cell_size = 0.03  # Adjust the cell size as needed
grid_cells = []
for x0 in np.arange(xmin, xmax + cell_size_degrees, cell_size_degrees):
    for y0 in np.arange(ymin, ymax + cell_size_degrees, cell_size_degrees):
        x1 = x0 - cell_size_degrees
        y1 = y0 + cell_size_degrees
        grid_cells.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'])
# Filter out invalid geometries
grid = grid[grid['geometry'].is_valid]
# Spatial join the grid with the intervention points
joined_interventions = gpd.sjoin(gdf_interventions, grid, how='left')
# Count incidents in each grid cell
incident_counts = joined_interventions.groupby('index_right').size()
# Map incidents counts to the grid
grid['incident_count'] = np.nan
grid.loc[incident_counts.index, 'incident_count'] = incident_counts.values
# print(grid['incident_count'].sort_values(ascending=True).value_counts())

# Perform a spatial join to add the incident_count to the interventions data
gdf_interventions_with_incident_count = gpd.sjoin(gdf_interventions, grid[['geometry', 'incident_count']], how='left')

# Plot the incident density
fig, ax = plt.subplots(figsize=(12, 8))
grid.plot(column='incident_count', ax=ax, cmap='OrRd', edgecolor='k', legend=True)
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
ax.set_title('Incident Density Grid')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

print(f'COLUMNS: {gdf_interventions_with_incident_count.columns}')
# Drop the index right column if it's not needed
gdf_interventions_with_incident_count = gdf_interventions_with_incident_count.drop(columns='index_right')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HEATMAP OF AED DENSITY
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 8))
# kernel density estimate (KDE) plot
kde = sns.kdeplot(data=aed_locations, x='Longitude', y='Latitude', cmap='Reds',
                  fill=True, ax=ax, cbar=True)

belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
ax.set_title('Heatmap of AED Density')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ADDING AED DENSITY TO DATASET
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Convert to GeoDataFrame
gdf_aed_locations = gpd.GeoDataFrame(
    aed_locations,
    geometry=gpd.points_from_xy(aed_locations.Longitude, aed_locations.Latitude)
)
# Spatial join the grid with the aed points
joined_aed = gpd.sjoin(gdf_aed_locations, grid, how='left')
# Count incidents in each grid cell
aed_counts = joined_aed.groupby('index_right').size()
# Map incidents counts to the grid
grid['aed_count'] = np.nan
grid.loc[aed_counts.index, 'aed_count'] = aed_counts.values
print(grid[['incident_count', 'aed_count']].value_counts())

# Perform a spatial join to add the incident_count to the interventions data
gdf_interventions_with_both_count = gpd.sjoin(gdf_interventions_with_incident_count, grid[['geometry', 'aed_count']],
                                              how='left')

# Plot the AED density
fig, ax = plt.subplots(figsize=(12, 8))
grid.plot(column='aed_count', ax=ax, cmap='OrRd', edgecolor='k', legend=True)
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
ax.set_title('AED Density Grid')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RESPONSE TIME ANALYSIS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Only check Response times longer then 10 minutes
# (According to literature waiting longer ten 10 minutes could be life-threatening)
ResponseTimeFilterd = gdf_interventions_with_both_count[gdf_interventions_with_both_count['T3-T0_min'] > 10]
ResponseTimeFilterd_sorted = ResponseTimeFilterd.sort_values(by='T3-T0_min', ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
scatter = sns.scatterplot(
    data=ResponseTimeFilterd_sorted,
    x='Longitude', y='Latitude',
    hue='T3-T0_min', palette='YlOrRd',
    size='T3-T0_min',
    sizes=(20, 200), legend='brief', ax=ax)
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
ax.set_title('Response Times by Location')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IDENTIFYING HIGH-RISK AREAS WITH LOW AED COVERAGE
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

sns.histplot(data=gdf_interventions_with_both_count['T3-T0_min'], kde=True, label='Response Time')
sns.histplot(data=gdf_interventions_with_both_count['incident_count'], kde=True,
             label='Incident Density (#incidents/3km²)')
sns.histplot(data=gdf_interventions_with_both_count['aed_count'], kde=True,
             label='AED Density (#AEDs/3km²)')
plt.legend()
plt.title('Density Response Time, Incident Density and AED Density')
plt.xlabel('')
plt.show()

'''
These thresholds are chosen a little arbitrary.
We defined a high risk area as an area where there have been more than 5 cardiac arrests per 3km² 
with a response time larger than 10 minutes in public places.

Literature says (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7892213/): 

'We found that a short Emergency Medical Service (EMS) response time was associated with a high rate of survival 
to hospital discharge after Out-of-Hospital Cardiac Arrest (OHCA). 
The optimal response time threshold for survival to hospital discharge was 6.2min. 
In the case of OHCA in public areas or with bystander CPR, the threshold was prolonged to 7.2min and 6.3min, 
respectively; and in the absence of a witness, the threshold was shortened to 4.2min.'

About the incident density it says:

'Incident density, or the frequency of events in a specific area, can impact ambulance response times and survival 
outcomes for out-of-hospital cardiac arrest (OHCA) patients.
High incident density areas may lead to longer response times due to increased demand on emergency medical services(EMS)
resources.'
'''
# Define high-risk based on response time and incident frequency
response_time_threshold = 10  # minutes
incident_density_threshold = 5  # more than five previous interventions on that location area
aed_density_threshold = 5  # less than 5 aeds in the 3 km³ grid for that location
# Identify high-risk areas
high_risk_areas = gdf_interventions_with_both_count[
    (gdf_interventions_with_both_count['T3-T0_min'] > response_time_threshold) &
    (gdf_interventions_with_both_count['incident_count'] > incident_density_threshold) &
    (gdf_interventions_with_both_count['aed_count'] < aed_density_threshold)
    ]

print(
    'Number of high risk areas in Belgium according to chosen thresholds: ',
    len(high_risk_areas[['Latitude', 'Longitude']].drop_duplicates()))  # 457

# Visualise high-risk areas
fig, ax = plt.subplots(figsize=(12, 8))
belgium_with_regions_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
scatter = sns.scatterplot(data=high_risk_areas,
                          x='Longitude', y='Latitude',
                          size='incident_count', hue='T3-T0_min', palette='coolwarm',
                          sizes=(20, 200), legend='auto', ax=ax)
ax.set_title('High-Risk Areas based on Response Time and Incident Frequency')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()
# You can see a lot of cardiac arrests in Antwerp, Brussels and Luik (and also on the lin Bergen, Charleroi, Namen)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CLUSTERING HIGH RISK AREAS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Identify high-risk areas to identify hotspots or highly concentrated high risk areas
high_risk_areas = high_risk_areas.dropna(subset=['Latitude', 'Longitude'])

# Calculate the average distance between each point in the data set and its 20 nearest neighbors
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(high_risk_areas[['Latitude', 'Longitude']])
distances, indices = neighbors_fit.kneighbors(high_risk_areas[['Latitude', 'Longitude']])
# Sort distance values by ascending value and plot
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.title('Distance to 5 nearest neighbours')
plt.plot(distances)

# Choose eps based on the k-distance graph
eps = cell_size_degrees  # See earlier for determining grid size: eps = 0.02463... this is approximately 3km
# Choose min_samples based on the data density
# A point is considered a core point if at least 5 points are within a distance of 3km
min_samples = 5

# Initialize DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Perform clustering
high_risk_areas['cluster'] = dbscan.fit_predict(high_risk_areas[['Latitude', 'Longitude']])

# Drop noise points (cluster == -1)
# Filter out noise points (cluster == -1)
high_risk_areas_filtered = high_risk_areas[high_risk_areas['cluster'] != -1]

# Determine new AED locations (cluster centers)
cluster_centers = high_risk_areas_filtered.groupby('cluster').mean()[['Latitude', 'Longitude']]
cluster_sizes = high_risk_areas_filtered['cluster'].value_counts()
print('New AED Locations: \n', cluster_centers)
labels = dbscan.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
# Plot clusters
fig, ax = plt.subplots(figsize=(12, 8))
custom_palette = sns.color_palette("Paired", len(cluster_sizes) + 1)
# Plot the high-risk areas and color by cluster
scatter = sns.scatterplot(
    data=high_risk_areas,
    x='Longitude', y='Latitude',
    hue='cluster', palette=custom_palette,
    size='incident_count',
    sizes=(20, 200), legend=False, ax=ax)
# Overlay the Belgium boundary
belgium_boundary.plot(ax=ax, facecolor='none', edgecolor='black')

# Overlay the proposed AED locations
new_aed_gdf = gpd.GeoDataFrame(cluster_centers,
                               geometry=gpd.points_from_xy(cluster_centers['Longitude'], cluster_centers['Latitude']))
new_aed_gdf.plot(ax=ax, marker='x', color='blue', markersize=100, label='Proposed AED Locations')

ax.set_title('High-Risk Areas Clusters and Proposed AED Locations')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

# New plot for cluster sizes
fig, ax = plt.subplots(figsize=(12, 8))

# Prepare cluster centers with sizes
cluster_centers['size'] = cluster_sizes

# Plot only cluster centers with sizes and colors depending on cluster size to determine most 'problematic' clusters
belgium_with_regions_boundary.plot(ax=ax, facecolor='none', edgecolor='black')
scatter = sns.scatterplot(data=cluster_centers,
                          x='Longitude', y='Latitude', size='size', sizes=(20, 200),
                          hue='size', palette='coolwarm', legend='brief')
ax.set_title('Cluster Centers with Sizes Proportional to Cluster Sizes')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()
plt.show()

# Evaluation metrics

sc = silhouette_score(high_risk_areas[['Latitude', 'Longitude']], labels)
print("Silhouette Coefficient:%0.2f" % sc)
# For incident_density_threshold = 10: sc = -0.13 and 11 clusters
# For incident_density_threshold = 5: sc = 0.69  and 82 clusters

'''
new_aed_gdf[['Latitude', 'Longitude']].to_csv(
    'C:/Users/Admin/Documents/GitHub/Project-AED-optimalization/DATA/new_aed_locations.csv', index=True)
'''

'''
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CONVERT NEW LOCATIONS TO ADRESSES
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def reverse_geocode(latitude, longitude):
    geolocator = ArcGIS()
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    return location.address


# Apply reverse geocoding to each row in the dataset
new_aed_gdf['Address'] = new_aed_gdf.apply(lambda row: reverse_geocode(row['Latitude'], row['Longitude']),
                                                 axis=1)

'''

# Save the filtered proposed AED locations
# new_aed_gdf_filtered[['Latitude', 'Longitude']].to_csv('DATA/new_filtered_aed_locations.csv', index=True)

# Save the combined AED locations (existing + new filtered)
# combined_aed_gdf[['Latitude', 'Longitude']].to_csv('DATA/combined_aed_locations.csv', index=True)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# End time
end_time = time.ctime(int(time.time()))
print(f'Program ended at {end_time}')
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
