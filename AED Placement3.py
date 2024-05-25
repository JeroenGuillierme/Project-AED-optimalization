import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

aed_data = pd.read_csv('DATA/updated_aed_df_with_all_distances.csv')
grid_data = pd.read_csv('DATA/grid_locations.csv')
print(aed_data.head())
print(grid_data.head())

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DATA PREPARATION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print(aed_data.columns)
print(grid_data.columns)

# Combine existing potential locations with new grid locations
all_potential_locations = pd.concat([aed_data,
                                     grid_gdf[['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel',
                                               'AED', 'Ambulance', 'Mug', 'Occasional_Permanence',
                                               'distance_to_aed', 'distance_to_ambulance',
                                               'distance_to_mug']]])


# Check for missing values
print(all_potential_locations.isnull().sum())
print(len(all_potential_locations))


# Plotting response times => looking for areas with high response times
plt.figure(figsize=(10, 6))
sns.histplot(all_potential_locations['T3-T0_min'], bins=30, kde=True)
plt.title('Distribution of Response Times')
plt.xlabel('Response Time (minutes)')
plt.ylabel('Frequency')
plt.show()

# Create histogram for the distribution of Latitude
plt.figure(figsize=(10, 6))
sns.histplot(all_potential_locations['Latitude'], bins=30, kde=True)
plt.title('Distribution of Latitudes')
plt.xlabel('Latitude')
plt.ylabel('Frequency')
plt.show()

# Create histogram for the distribution of Longitude
plt.figure(figsize=(10, 6))
sns.histplot(all_potential_locations['Longitude'], bins=30, kde=True)
plt.title('Distribution of Longitudes')
plt.xlabel('Longitude')
plt.ylabel('Frequency')
plt.show()



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
features_for_imputation = all_potential_locations[['Latitude', 'Longitude', 'T3-T0_min']]
# Separate features to scale
features_to_scale = features_for_imputation[['Latitude', 'Longitude']]
# Initialize the scaler
scaler = StandardScaler()
# Fit and transform the features to scale
scaled_features = scaler.fit_transform(features_to_scale)
# Combine scaled features with the original dataframe
scaled_data = pd.DataFrame(scaled_features, columns=['Latitude_scaled', 'Longitude_scaled'])
scaled_data['T3-T0_min'] = features_for_imputation['T3-T0_min'].values
# Initialize the KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)
# Perform KNN Imputation
imputed_values = knn_imputer.fit_transform(scaled_data)
# Create a dataframe with the imputed values
imputed_data = pd.DataFrame(imputed_values, columns=scaled_data.columns)
# Inverse transform the scaled features back to original scale
imputed_data[['Latitude_scaled', 'Longitude_scaled']] = scaler.inverse_transform(imputed_data[['Latitude_scaled', 'Longitude_scaled']])
imputed_data.rename(columns={'Latitude_scaled': 'Latitude', 'Longitude_scaled': 'Longitude'}, inplace=True)
# Assign the imputed values back to the original dataframe
all_potential_locations['T3-T0_min'] = imputed_data['T3-T0_min']

# Verify the imputation
print(all_potential_locations.isnull().sum())


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IDENTIFYING HIGH-RISK AREAS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Plotting response times
plt.figure(figsize=(10, 6))
sns.histplot(aed_data['T3-T0_min'], bins=30, kde=True)
plt.title('Distribution of Response Times')
plt.xlabel('Response Time (minutes)')
plt.ylabel('Frequency')
plt.show()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CURRENT AED-COVERAGE
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# dataframe `current_aeds` with AED locations
current_aeds = all_potential_locations[aed_data['AED'] == 1]
print(len(aed_data[aed_data['AED'] == 1]))
incidents = all_potential_locations[aed_data['Intervention'] == 1]

# Create a GeoDataFrame for the incidents
gdf_incidents = gpd.GeoDataFrame(
    incidents,
    geometry=gpd.points_from_xy(incidents.Longitude, incidents.Latitude)
)

# Plotting existing AEDs and incidents
fig, ax = plt.subplots(figsize=(10, 10))
gdf_incidents.plot(ax=ax, marker='o', color='red', markersize=5, label='Incidents')
current_aeds.plot(ax=ax, marker='x', color='blue', markersize=50, label='Current AEDs')
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