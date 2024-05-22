import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your original dataset
df = pd.read_csv('DATA/updated_aed_df_with_all_distances.csv')
df2 = df

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

print("Weights from Linear Regression:")
print(weights) # Result: [0.         0.06526201 0.00450788 0.26577536 0.63171176 0.03274299]

# Apply the weights to calculate the weighted score for each location
df['score'] = (df['Intervention'] * weights[0] +
               df['CAD9'] * weights[1] +
               df['Eventlevel'] * weights[2] +
               df['distance_to_aed'] * weights[3] +
               df['distance_to_ambulance'] * weights[4] +
               df['distance_to_mug'] * weights[5])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Drop any rows with NaN values
df2.dropna(inplace=True)

# Select relevant variables for PCA
# For example, excluding variables like latitude, longitude, and intervention
# You can adjust this based on your specific dataset
selected_columns = ['Intervention', 'AED', 'Ambulance', 'Mug', 'Occasional_Permanence',
                    'distance_to_aed', 'distance_to_ambulance', 'distance_to_mug', 'T3-T0_min']

# Extract the selected columns
data = df2[selected_columns]

# Standardize the data
scaler = StandardScaler()
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
