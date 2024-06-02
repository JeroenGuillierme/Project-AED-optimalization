import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IMPORTING DATASETS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ambulance = pd.read_parquet('DATA/ambulance_locations.parquet.gzip')
mug = pd.read_parquet('DATA/mug_locations.parquet.gzip')
pit = pd.read_parquet('DATA/pit_locations.parquet.gzip')
interventions1 = pd.read_parquet('DATA/interventions1.parquet.gzip')
interventions2 = pd.read_parquet('DATA/interventions2.parquet.gzip')
interventions3 = pd.read_parquet('DATA/interventions3.parquet.gzip')
interventions4 = pd.read_parquet('DATA/interventions_bxl.parquet.gzip')
interventions5 = pd.read_parquet('DATA/interventions_bxl2.parquet.gzip')
cad = pd.read_parquet('DATA/cad9.parquet.gzip')
aed = pd.read_parquet('DATA/aed_locations.parquet.gzip')

aed_1 = pd.read_csv('DATA/aed1.csv')
aed_2 = pd.read_csv('DATA/aed2.csv')
aed_3 = pd.read_csv('DATA/aed3.csv')
aed_4 = pd.read_csv('DATA/aed4.csv')
aed_5 = pd.read_csv('DATA/aed5.csv')

mug1 = pd.read_csv('DATA/mug1.csv')

pd.set_option('display.max_columns', None)
pd.options.mode.copy_on_write = True


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to correct latitude values
def correct_latitude(lat):
    '''
    Corrects and standardizes latitude values by ensuring they are numeric and properly formatted.
    :param lat: str, float or int
    The latitude value to be corrected.
    :return: float
    The corrected latitude value, or NaN if the input is NaN.
    '''
    if pd.isna(lat):
        return lat  # Return NaN as it is
    if isinstance(lat, (str, float, int)):
        lat_str = str(lat)
        # Remove any existing non-numeric characters (except -)
        lat_str = re.sub(r'[^0-9-]', '', lat_str)
        # Move the decimal point to ensure two digits before the decimal point
        if len(lat_str) > 2:
            lat_str = lat_str[:2] + '.' + lat_str[2:]
        return float(lat_str)
    return lat


# Function to correct longitude values
def correct_longitude(lon):
    '''
    Corrects and standardizes longitude values by ensuring they are numeric and properly formatted.
    :param lon: str, float or int
    The longitude value to be corrected.
    :return:float
    The corrected longitude value, or NaN if the input is NaN.
    '''
    if pd.isna(lon):
        return lon  # Return NaN as it is
    if isinstance(lon, (str, float, int)):
        lon_str = str(lon)
        # Remove any existing non-numeric characters (except -)
        lon_str = re.sub(r'[^0-9-]', '', lon_str)
        # Move the decimal point to ensure one digit before the decimal point
        if len(lon_str) > 1:
            lon_str = lon_str[:1] + '.' + lon_str[1:]
        return float(lon_str)
    return lon


# Define a function to extract the numeric part using regex
def extract_numeric(text):
    '''
    Extracts the first numeric part from a given text using regular expressions.
    :param text: str
    The text from which to extract the numeric part.
    :return: int or float
    The extracted numeric value, or NaN if no numeric part is found.
    '''
    match = re.search(r'\d+', text)
    return int(match.group()) if match else np.nan


# Function to filter rows based on coordinates falling within Belgium
def is_within_belgium(lat, lon):
    '''
    Checks if given latitude and longitude coordinates fall within Belgium's geographical boundaries.
    :param lat: float
    The latitude value to be checked.
    :param lon: float
    The longitude value to be checked.
    :return: bool
    True if the coordinates are within Belgium's boundaries, False otherwise.
    '''
    # Define the geographical boundaries of Belgium
    belgium_boundaries = {
        'min_latitude': 48,
        'max_latitude': 55,
        'min_longitude': 2,
        'max_longitude': 8
    }

    return (belgium_boundaries['min_latitude'] <= lat <= belgium_boundaries['max_latitude']) and \
        (belgium_boundaries['min_longitude'] <= lon <= belgium_boundaries['max_longitude'])


# Function to convert Timedelta to minutes
def timedelta_to_minutes(td):
    '''
    Converts a pandas Timedelta object to minutes.
    :param td: pd.TimeDelta
    The Timedelta object to be converted.
    :return: float
    The total duration in minutes.
    '''
    return td.total_seconds() / 60


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREPROCESSING INTERVENTIONS DATASETS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Append interventions dataset (1-3)
interventions1["T0"] = pd.to_datetime(interventions1["T0"],
                                      format='%d%b%y:%H:%M:%S')  # Convert the first time format to datetime
interventions1["T3"] = pd.to_datetime(interventions1["T3"],
                                      format='%Y-%m-%d %H:%M:%S.%f')  # Convert the column back to datetime with the new format
interventions2["T0"] = pd.to_datetime(interventions2["T0"],
                                      format='%d%b%y:%H:%M:%S')  # Convert the first time format to datetime
interventions2["T3"] = pd.to_datetime(interventions2["T3"],
                                      format='%Y-%m-%d %H:%M:%S.%f')  # Convert the column back to datetime with the new format
interventions3["T0"] = pd.to_datetime(interventions3["T0"],
                                      format='%d%b%y:%H:%M:%S')  # Convert the first time format to datetime
interventions3["T3"] = pd.to_datetime(interventions3["T3"],
                                      format='%Y-%m-%d %H:%M:%S.%f')  # Convert the column back to datetime with the new format
interventions_total = pd.concat([interventions1, interventions2, interventions3], axis=0)

interventions_total['CAD9'] = 0
interventions_total["T3-T0"] = interventions_total["T3"] - interventions_total["T0"]
interventions_total = interventions_total[interventions_total["EventType Firstcall"] == "P003 - Cardiac arrest"]
interventions_total['Eventlevel'] = interventions_total["EventLevel Firstcall"]
interventions_total['Latitude'] = interventions_total['Latitude intervention'].apply(correct_latitude)
interventions_total['Longitude'] = interventions_total['Longitude intervention'].apply(correct_longitude)
interventions_total['Intervention'] = 1
interventions_total = interventions_total[
    ['Latitude', 'Longitude', "Intervention", "CAD9", "Eventlevel", "T3-T0"]]

# Append CAD9 dataset
CAD9_expanded = cad
CAD9_expanded['CAD9'] = 1  # add extra column CAD9
CAD9_expanded['Intervention'] = 1  # add extra column Intervention
CAD9_expanded["T0"] = pd.to_datetime(CAD9_expanded["T0"],
                                     format='%Y-%m-%d %H:%M:%S.%f')  # Convert the column back to datetime with the new format
CAD9_expanded["T3"] = pd.to_datetime(CAD9_expanded["T3"],
                                     format='%Y-%m-%d %H:%M:%S.%f')  # Convert the column back to datetime with the new format
CAD9_expanded["T3-T0"] = CAD9_expanded["T3"] - CAD9_expanded["T0"]
CAD9_expanded.loc[
    CAD9_expanded['T3-T0'] < pd.Timedelta(0), 'T3-T0'] = pd.NaT  # Transform negative time difference to NaT
CAD9_expanded = CAD9_expanded[CAD9_expanded["EventType Trip"] == "P003 - HARTSTILSTAND - DOOD - OVERLEDEN"]
CAD9_expanded['Eventlevel'] = CAD9_expanded['EventLevel Trip']
CAD9_expanded['Latitude'] = CAD9_expanded['Latitude intervention'].apply(correct_latitude)
CAD9_expanded['Longitude'] = CAD9_expanded['Longitude intervention'].apply(correct_longitude)
CAD9_expanded = CAD9_expanded[['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel', 'T3-T0']]

# Append datasets interventions Brussels
# Interventions 4
interventions4["t0"] = pd.to_datetime(interventions4["t0"],
                                      format='%Y-%m-%d %H:%M:%S.%f %z')  # Convert the first time format to datetime
interventions4["t3"] = pd.to_datetime(interventions4["t3"],
                                      format='%Y-%m-%d %H:%M:%S.%f %z')  # Convert the column back to datetime with the new format
interventions4['CAD9'] = 0
interventions4['Intervention'] = 1
interventions4 = interventions4[interventions4['eventtype_firstcall'] == 'P003 - Cardiac arrest']
interventions4['Eventlevel'] = interventions4['eventLevel_firstcall']
interventions4['T3-T0'] = interventions4['t3'] - interventions4['t0']
interventions4['Latitude'] = interventions4['latitude_intervention'].apply(correct_latitude)
interventions4['Longitude'] = interventions4['longitude_intervention'].apply(correct_longitude)
interventions4 = interventions4[['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel', 'T3-T0']]

# Interventions 5
interventions5["T0"] = pd.to_datetime(interventions5["T0"],
                                      format='%d%b%y:%H:%M:%S')  # Convert the first time format to datetime
interventions5["T3"] = pd.to_datetime(interventions5["T3"],
                                      format='%d%b%y:%H:%M:%S')  # Convert the column back to datetime with the new format
interventions5['CAD9'] = 0
interventions5['Intervention'] = 1
interventions5 = interventions5[
    (interventions5['EventType and EventLevel'] == 'P003  N01 - HARTSTILSTAND - DOOD - OVERLEDEN') | (
            interventions5['EventType and EventLevel'] == 'P003  N05 - HARTSTILSTAND - DOOD - OVERLEDEN')]
interventions5["Eventlevel"] = interventions5["EventType and EventLevel"].str.split(" ").str[1].str.replace("0", "")
interventions5['T3-T0'] = interventions5['T3'] - interventions5['T0']
interventions5['Latitude'] = interventions5['Latitude intervention'].apply(correct_latitude)
interventions5['Longitude'] = interventions5['Longitude intervention'].apply(correct_longitude)
interventions5 = interventions5[['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel', 'T3-T0']]

# Putting everything together
interventions_TOTAL = pd.concat([interventions_total, CAD9_expanded, interventions4, interventions5], axis=0)
interventions_TOTAL['AED'] = 0  # Adding extra AED column
interventions_TOTAL['Eventlevel'] = interventions_TOTAL['Eventlevel'].apply(extract_numeric)
interventions_TOTAL['Ambulance'] = 0
interventions_TOTAL['Mug'] = 0
interventions_TOTAL['Occasional_Permanence'] = np.nan

# Filter out rows where T3-T0 is greater than one day => Changed by IsolationForest algorithm
# one_day = pd.Timedelta(days=1)
# one_hour = pd.Timedelta(minutes=55)
# interventions_TOTAL = interventions_TOTAL[interventions_TOTAL['T3-T0'] < one_hour] # We lose approx. 4000 interventions with a response tie longer then one day

interventions_TOTAL['T3-T0_min'] = interventions_TOTAL['T3-T0'].apply(timedelta_to_minutes)
interventions_TOTAL = interventions_TOTAL[
    ['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel', 'T3-T0_min', 'AED', 'Ambulance', 'Mug',
     'Occasional_Permanence']]
print(list(interventions_TOTAL.columns))
print(interventions_TOTAL['T3-T0_min'])

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREPROCESSING AED DATASETS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

aed_total = pd.concat([aed_1, aed_2, aed_3, aed_4, aed_5], axis=0)
yes_values = ['Y', 'y', 'Oui-Ja', 'Ja', 'Oui', 'J', np.nan]
aed_total = aed_total[
    aed_total['public'].isin(yes_values)]  # discard non-public aed's and also when no data is available

aed_total['Latitude'] = aed_total['latitude'].apply(correct_latitude)
aed_total['Longitude'] = aed_total['longitude'].apply(correct_longitude)
aed_total['CAD9'] = 0
aed_total['Intervention'] = 0
aed_total['AED'] = 1
aed_total['Eventlevel'] = np.nan
aed_total['T3-T0_min'] = pd.NaT
aed_total['Ambulance'] = 0
aed_total['Mug'] = 0
aed_total['Occasional_Permanence'] = np.nan
aed_total = aed_total[
    ['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel', 'T3-T0_min', 'AED', 'Ambulance', 'Mug',
     'Occasional_Permanence']]


# Apply the filter function to the aed_total
aed_total2 = aed_total[aed_total.apply(lambda row: is_within_belgium(row['Latitude'], row['Longitude']), axis=1)]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREPROCESSING AMBULANCE DATASETS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ambulance['Latitude'] = ambulance['latitude'].apply(correct_latitude)
ambulance['Longitude'] = ambulance['longitude'].apply(correct_longitude)
ambulance['Occasional_Permanence'] = ambulance['occasional_permanence'].replace({'N': 0, 'Y': 1})
ambulance['CAD9'] = 0
ambulance['Intervention'] = 0
ambulance['AED'] = 0
ambulance['Eventlevel'] = np.nan
ambulance['T3-T0_min'] = pd.NaT
ambulance['Ambulance'] = 1
ambulance['Mug'] = 0
ambulance = ambulance[
    ['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel', 'T3-T0_min', 'AED', 'Ambulance', 'Mug',
     'Occasional_Permanence']]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREPROCESSING MUG DATASETS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

mug1['Latitude'] = mug1['latitude'].apply(correct_latitude)
mug1['Longitude'] = mug1['longitude'].apply(correct_longitude)
mug1['Occasional_Permanence'] = 1  # we assume the mug comes from the hospital and is permanently available.
mug1['CAD9'] = 0
mug1['Intervention'] = 0
mug1['AED'] = 0
mug1['Eventlevel'] = np.nan
mug1['T3-T0_min'] = pd.NaT
mug1['Ambulance'] = 1
mug1['Mug'] = 1
mug1 = mug1[['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel', 'T3-T0_min', 'AED', 'Ambulance', 'Mug',
             'Occasional_Permanence']]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ALL TOGETHER FOR AED OPTIMISATION
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print(list(aed_total2.columns))
print(list(interventions_TOTAL.columns))
print(list(ambulance.columns))
print(list(mug1.columns))
aed_df = pd.concat([interventions_TOTAL, aed_total2, ambulance, mug1], axis=0)

print(aed_df.head())
print(len(aed_df))
print('Unique number of values Latitude values: ', len(list(aed_df['Latitude'].unique())))
print('Unique number of values Longitude values: ', len(list(aed_df['Longitude'].unique())))
print('Unique values eventlevels', aed_df['Eventlevel'].unique())
print('Unique values CAD9: ', aed_df['CAD9'].unique())
cross_tab = pd.crosstab(index=pd.Categorical(aed_df["Eventlevel"]), columns='count')
print('Cross table of Event Levels: \n', cross_tab)

max_responseTime = aed_df['T3-T0_min'].max()
print(f"Maximum Response Time for aed_df: {max_responseTime}")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# OUTLIER DETECTION RESPONSE TIME T3-T0
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IsolationForest (for outliers/anomalies of the response time T3-T0, because some are longer than 1 day)
# => probably incorrectly filled in, in the dataset

print(aed_df['T3-T0_min'].isna().sum())  # 17445 NaN values

# Split the DataFrame into two: one with the NaN and one without in the 'T3-T0_min' column
aed_df_with_nan = aed_df[aed_df['T3-T0_min'].isna()]  # DataFrame with NaN values
aed_df_without_nan = aed_df[~aed_df['T3-T0_min'].isna()]  # DataFrame without NaN values
print('Without NaN: ', len(aed_df_without_nan))
print('With NaN: ', len(aed_df_with_nan))

Time = aed_df_without_nan['T3-T0_min']

# IsolationForest algorithm
IsoFo = IsolationForest(n_estimators=100, contamination='auto',
                        random_state=45)  # Random state added for reproducibility
y_labels = IsoFo.fit_predict(np.array(Time).reshape(-1, 1))

# Only including the inliers
aed_df_filtered = aed_df_without_nan[y_labels == 1]  # DataFrame with inliers
discarded_rows = aed_df_without_nan[y_labels == -1]  # DataFrame with outliers

min_timedelta = discarded_rows['T3-T0_min'].min()  # Time deltas larger than 44.27 minutes are discarded
max_timedelta = discarded_rows['T3-T0_min'].max()
min_timedelta2 = aed_df_filtered['T3-T0_min'].min()
max_timedelta2 = aed_df_filtered['T3-T0_min'].max()
print(f"min_timedelta of outliers: {min_timedelta}")  # Minimum outlier value = 44.27 minutes
print(f"max_timedelta of outliers: {max_timedelta}")  # Maximum outlier value = 80267.83 minutes
print(f"min_timedelta of inliers: {min_timedelta2}")  # Minimum outlier value = 0.62 minutes
print(f"max_timedelta of inliers: {max_timedelta2}")  # Maximum outlier value = 44.17 minutes
print(f"Number of discarded rows: {len(discarded_rows)}")  # 1040
print(f"Number of filtered rows: {len(aed_df_filtered)}")  # 9306

print("\nFiltered DataFrame (Inliers):")
# print(aed_df_filtered)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TOTAL DATASET WITHOUT OUTLIERS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Add rows with NaN values again:
print(list(aed_df_filtered.columns))
print(list(aed_df_with_nan.columns))
aed_ready = pd.concat([aed_df_filtered, aed_df_with_nan], axis=0)

x = range(0, 9306)
plt.scatter(x, aed_df_filtered['T3-T0_min'])
plt.show()

# Last check if no weird values are included in the dataset

# Get the minimum and maximum values of the 'latitude' column
min_latitude = aed_ready['Latitude'].min()
max_latitude = aed_ready['Latitude'].max()
# Get the minimum and maximum values of the 'longitude' column
min_longitude = aed_ready['Longitude'].min()
max_longitude = aed_ready['Longitude'].max()
# Get the minimum and maximum values of the response time column
min_timedelta = aed_ready['T3-T0_min'].min()
max_timedelta = aed_ready['T3-T0_min'].max()

print('Length of dataset: ', len(aed_ready))
print('Number of missing values per column: \n', print(aed_ready[['Latitude', 'Longitude', 'Intervention', 'CAD9',
                                                                  'Eventlevel', 'T3-T0_min', 'AED', 'Ambulance', 'Mug',
                                                                  'Occasional_Permanence']].isna().sum())) # 1210 missing values for latitude from intervention dataset
print(f"Minimum latitude of dataset: {min_latitude}")
print(f"Maximum latitude of dataset: {max_latitude}")
print(f"Minimum longitude of dataset: {min_longitude}")
print(f"Maximum longitude of dataset: {max_longitude}")
print(f"min_timedelta of dataset: {min_timedelta}")  # Minimum outlier value = 44.27 minutes
print(f"max_timedelta of dataset: {max_timedelta}")  # Maximum outlier value = 80267.83 minutes

#aed_ready.to_csv('DATA/aed_placement_df.csv', index=False)
