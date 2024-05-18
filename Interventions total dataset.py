import pandas as pd
from datetime import datetime
import numpy as np
import re

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

pd.set_option('display.max_columns', None)
pd.options.mode.copy_on_write = True


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to correct latitude values
def correct_latitude(lat):
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
    match = re.search(r'\d+', text)
    return int(match.group()) if match else ''

# Function to filter rows based on coordinates falling within Belgium
def is_within_belgium(lat, lon):
    return (belgium_boundaries['min_latitude'] <= lat <= belgium_boundaries['max_latitude']) and \
        (belgium_boundaries['min_longitude'] <= lon <= belgium_boundaries['max_longitude'])


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
CAD9_expanded['CAD9'] = 1 # add extra column CAD9
CAD9_expanded['Intervention'] = 1 # add extra column Intervention
CAD9_expanded["T0"] = pd.to_datetime(CAD9_expanded["T0"],
                                     format='%Y-%m-%d %H:%M:%S.%f')  # Convert the column back to datetime with the new format
CAD9_expanded["T3"] = pd.to_datetime(CAD9_expanded["T3"],
                                     format='%Y-%m-%d %H:%M:%S.%f')  # Convert the column back to datetime with the new format
CAD9_expanded["T3-T0"] = CAD9_expanded["T3"] - CAD9_expanded["T0"]
CAD9_expanded.loc[CAD9_expanded['T3-T0'] < pd.Timedelta(0), 'T3-T0'] = pd.NaT # Transform negative time difference to NaT
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
interventions_TOTAL['AED'] = 0 # Adding extra AED column
interventions_TOTAL['Eventlevel'] = interventions_TOTAL['Eventlevel'].apply(extract_numeric)



# Get the minimum and maximum values of the 'latitude' column
min_latitude = interventions_TOTAL['Latitude'].min()
max_latitude = interventions_TOTAL['Latitude'].max()
# Get the minimum and maximum values of the 'longitude' column
min_longitude = interventions_TOTAL['Longitude'].min()
max_longitude = interventions_TOTAL['Longitude'].max()

print(f"Minimum latitude: {min_latitude}")
print(f"Maximum latitude: {max_latitude}")
print(f"Minimum longitude: {min_longitude}")
print(f"Maximum longitude: {max_longitude}")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# AED DATASETS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


aed_total = pd.concat([aed_1, aed_2, aed_3, aed_4, aed_5], axis=0)
yes_values = ['Y', 'y', 'Oui-Ja', 'Ja', 'Oui', 'J', '']
aed_total = aed_total[
    aed_total['public'].isin(yes_values)]  # discard non-public aed's and also when no data is available


aed_total['Latitude'] = aed_total['latitude']
aed_total['Longitude'] = aed_total['longitude']
aed_total['CAD9'] = 0
aed_total['Intervention'] = 0
aed_total['AED'] = 1
aed_total['Eventlevel'] = ''
aed_total['T3-T0'] = pd.NaT
aed_total = aed_total[['Latitude', 'Longitude', 'Intervention', 'CAD9', 'Eventlevel', 'T3-T0', 'AED']]


# Get the minimum and maximum values of the 'latitude' column
min_latitude = aed_total['Latitude'].min()
max_latitude = aed_total['Latitude'].max()
# Get the minimum and maximum values of the 'longitude' column
min_longitude = aed_total['Longitude'].min()
max_longitude = aed_total['Longitude'].max()

print(f"Minimum latitude: {min_latitude}")
print(f"Maximum latitude: {max_latitude}")
print(f"Minimum longitude: {min_longitude}")
print(f"Maximum longitude: {max_longitude}")
print(len(aed_total))

# Define the geographical boundaries of Belgium
belgium_boundaries = {
    'min_latitude': 48,
    'max_latitude': 55,
    'min_longitude': 2,
    'max_longitude': 8
}


# Apply the filter function to the aed_total
aed_total2 = aed_total[aed_total.apply(lambda row: is_within_belgium(row['Latitude'], row['Longitude']), axis=1)]


# Get the minimum and maximum values of the 'latitude' column
min_latitude = aed_total2['Latitude'].min()
max_latitude = aed_total2['Latitude'].max()
# Get the minimum and maximum values of the 'longitude' column
min_longitude = aed_total2['Longitude'].min()
max_longitude = aed_total2['Longitude'].max()

print(f"Minimum latitude aed_total2: {min_latitude}")
print(f"Maximum latitude aed_total2: {max_latitude}")
print(f"Minimum longitude aed_total2: {min_longitude}")
print(f"Maximum longitude aed_total2: {max_longitude}")
print(len(aed_total))

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ALL TOGETHER FOR AED OPTIMISATION
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print(list(aed_total2.columns))
print(list(interventions_TOTAL.columns))

aed_df = pd.concat([interventions_TOTAL, aed_total2], axis=0)

print(aed_df.head())
print(len(aed_df))
print(len(list(aed_df['Latitude'].unique())))
print(len(list(aed_df['Longitude'].unique())))
print(aed_df['Eventlevel'].unique())
print(aed_df['CAD9'].unique())
cross_tab = pd.crosstab(index=pd.Categorical(aed_df["Eventlevel"]), columns='count')
print(cross_tab)


# Get the minimum and maximum values of the 'latitude' column
min_latitude = aed_df['Latitude'].min()
max_latitude = aed_df['Latitude'].max()
# Get the minimum and maximum values of the 'longitude' column
min_longitude = aed_df['Longitude'].min()
max_longitude = aed_df['Longitude'].max()

print(f"Minimum latitude aed_df: {min_latitude}")
print(f"Maximum latitude aed_df: {max_latitude}")
print(f"Minimum longitude aed_df: {min_longitude}")
print(f"Maximum longitude aed_df: {max_longitude}")

# df.to_parquet('DATA/aed_df.parquet', index=False)
