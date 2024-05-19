import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# Load dataset
aed = pd.read_csv('DATA/aed_placement_df.csv')

# Impute NaN values
imputer_continuous = SimpleImputer(strategy='mean')
aed['T3-T0_min'] = imputer_continuous.fit_transform(pd.DataFrame(aed['T3-T0_min']))
aed['Eventlevel'].fillna(aed['Eventlevel'].median(), inplace=True)

# Encode NaN values in categorical variables
aed['Occasional_Permanence'].fillna(-1, inplace=True)

# Normalise continuous variable
scaler = StandardScaler()
aed[['T3-T0_min']] = scaler.fit_transform(aed[['T3-T0_min']])

# Remove rows with NaN values in latitude
aed = aed.dropna(subset=['Latitude'])

# geospatial encoding
aed['Coordinates'] = aed.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
geo_aed = gpd.GeoDataFrame(aed, geometry='Coordinates')

print(aed[['Latitude', 'Longitude', 'T3-T0_min', 'Eventlevel', 'AED', 'Intervention']].isna().sum())

# Clustering
kmeans = KMeans(n_clusters = 10, random_state=45)
aed['Cluster'] = kmeans.fit_predict(aed[['Latitude', 'Longitude', 'T3-T0_min', 'Eventlevel', 'AED', 'Intervention']])

# visualise clusters

geo_aed['Cluster'] = aed['Cluster']
geo_aed.plot(column='Cluster', cmap='viridis', legend=True)
plt.show()

#Identify clusters lacking AEDs and analyse them for potential AED placement
high_need_clusters = aed[aed['AED'] == 0].groupby('Cluster').agg({
    'Eventlevel': 'mean',
    'T3-T0_min': 'mean'}).sort_values(by=['Eventlevel', 'T3-T0_min'],
                                      ascending=[False, True])
print(high_need_clusters)