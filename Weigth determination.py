import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from scipy.spatial.distance import cdist

# Load the Belgium boundary shapefile
belgium_boundary = gpd.read_file('DATA/België.json')

# Adjust grid size as needed
grid_size = 0.01  # Adjust this value based on your needs

# Generate a grid of potential new locations within Belgium boundary
minx, miny, maxx, maxy = belgium_boundary.total_bounds
grid_points = []

x_coords = np.arange(minx, maxx, grid_size)
y_coords = np.arange(miny, maxy, grid_size)

for x in x_coords:
    for y in y_coords:
        point = Point(x, y)
        if belgium_boundary.contains(point).any():
            grid_points.append(point)

# Create a GeoDataFrame for the grid points
grid_gdf = gpd.GeoDataFrame(grid_points, columns=['geometry'])
grid_gdf['Latitude'] = grid_gdf.geometry.y
grid_gdf['Longitude'] = grid_gdf.geometry.x

print(grid_gdf.head())

# Load your original dataset
df = pd.read_csv('DATA/updated_aed_df_with_all_distances.csv')

# Fill NaN values in Eventlevel and Occasional_Permanence
df['Eventlevel'].fillna(-1, inplace=True)
df['Occasional_Permanence'].fillna(-1, inplace=True)

# Remove rows with NaN values in Latitude or Longitude
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Normalize the numerical columns
scaler = MinMaxScaler()
df[['distance_to_aed', 'distance_to_ambulance', 'distance_to_mug']] = scaler.fit_transform(
    df[['distance_to_aed', 'distance_to_ambulance', 'distance_to_mug']])

# Train a regression model to determine weights dynamically
features = ['Intervention', 'CAD9', 'Eventlevel', 'distance_to_aed', 'distance_to_ambulance', 'distance_to_mug']
target = 'T3-T0_min'  # Replace with the actual target variable in your dataset

# Ensure there are no NaN values in the features and target
df = df.dropna(subset=features + [target])

# Train a linear regression model
X = df[features]
y = df[target]
reg = LinearRegression().fit(X, y)

# Extract the coefficients and normalize them to sum to 1
coefficients = reg.coef_
weights = coefficients / coefficients.sum()


print(weights) # Result: [0.         0.06526201 0.00450788 0.26577536 0.63171176 0.03274299]

# Apply the weights to calculate the weighted score for each location
df['score'] = (df['Intervention'] * weights[0] +
               df['CAD9'] * weights[1] +
               df['Eventlevel'] * weights[2] +
               df['distance_to_aed'] * weights[3] +
               df['distance_to_ambulance'] * weights[4] +
               df['distance_to_mug'] * weights[5])


