import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import statsmodels.api as sm
import math
from sklearn.cluster import KMeans

# Load your original dataset
df = pd.read_csv('DATA/updated_aed_df_with_all_distances.csv')

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Coverage Radius
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Response times for Ambulances
response_times = df['T3-T0_min'].dropna()
print(response_times.describe())
percentile_90_ambulance = np.percentile(response_times, 90)
print(f"90th percentile ambulance response time: {percentile_90_ambulance} minutes")

# Response times for AED
target_aed_response_time = 5  # Target response time for AEDs in minutes
average_speed_km_per_min = 12 / 60  # 12 km/h in km per minute becaue in distress people will run to AED
coverage_radius_km = target_aed_response_time * average_speed_km_per_min
coverage_radius_m = coverage_radius_km * 1000  # Convert to meters
print(f"Adjusted coverage radius for AEDs: {coverage_radius_m} meters")



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plot Distributions
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create histogram
plt.hist(np.log1p(df['distance_to_aed']), edgecolor='black', bins=20)
plt.show()

plt.hist(np.log1p(df['distance_to_ambulance']+1), edgecolor='black', bins=20)
plt.show()

plt.hist(np.log1p(df['distance_to_mug']+1), edgecolor='black', bins=20)
plt.show()

plt.hist(np.log1p(df['T3-T0_min']), edgecolor='black', bins=20)
plt.show()



# Fill NaN values in Eventlevel and Occasional_Permanence
df['Eventlevel'].fillna(-1, inplace=True)
df['Occasional_Permanence'].fillna(-1, inplace=True)

# Remove rows with NaN values in Latitude or Longitude
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Select relevant variables for PCA
# For example, excluding variables like latitude, longitude, and intervention
# You can adjust this based on your specific dataset
selected_columns = ['Intervention', 'AED', 'Ambulance', 'Mug', 'Occasional_Permanence', 'Eventlevel',
                    'distance_to_aed', 'distance_to_ambulance', 'distance_to_mug', 'T3-T0_min']

# Extract the selected columns
data = df[selected_columns]
print(data.isna().sum())

# Standardize the data
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)

# Perform PCA
pca = PCA()
principal_components = pca.fit_transform(scaled_data)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='-')
plt.title('Explained Variance Ratio')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

# Determine the number of components to retain
# You can also use a scree plot to visually inspect for an "elbow"
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()
num_components = len(cumulative_variance[cumulative_variance <= 0.95]) + 1
print("Number of components to retain: ", num_components)

# Fit PCA with the determined number of components
pca = PCA(n_components=num_components)
principal_components = pca.fit_transform(scaled_data)

# Get the loadings of each variable on each principal component
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Display the loadings
loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(num_components)], index=selected_columns)
print("Loadings of variables on principal components:")
print(loadings_df)
