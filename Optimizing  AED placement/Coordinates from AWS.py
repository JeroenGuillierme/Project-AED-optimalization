import pandas as pd
from pandas.compat import pyarrow
import boto3

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IMPORTING DATASETS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

mug = pd.read_parquet('DATA/mug_locations.parquet.gzip')
aed = pd.read_parquet('DATA/aed_locations.parquet.gzip')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#  CONVERTING ADDRESSES INTO COORDINATES
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

aed["municipality"] = aed["municipality"].replace("KalloBeveren-Waas)","Beveren")

# I initially broke it down into five steps, thinking it would be faster, but I realized it could probably be done in one go
aed5 = aed.iloc[14001:15227].copy()

# Aws region in which the Amazon locator service is configured, it probably would have been better to replace it with eu-central-1
# region_name = 'us-east-1'

# Configuring AWS credentials after creating an IAM user
# session = boto3.Session(
#    aws_access_key_id='AKIA6GBMAUQCXW4TPTUE',
#    aws_secret_access_key='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',
#    region_name=region_name)

# client = boto3.client('location')

def get_coordinates(street, house_number, city):
    try:
        response = client.search_place_index_for_text(
            IndexName='explore.place.Esri',
            Text=f'{street} {house_number}, {city}, Belgium')

        latitude = response['Results'][0]['Place']['Geometry']['Point'][1]
        longitude = response['Results'][0]['Place']['Geometry']['Point'][0]

        return latitude, longitude

    except Exception as e:
        print(f"Error occurred for {street} {house_number}, {city}: {str(e)}")
        return None, None

latitudes = []
longitudes = []

for index, row in aed5.iterrows():
    street = row['address']
    house_number = row['number']
    city = row['municipality']

    latitude, longitude = get_coordinates(street, house_number, city)
    latitudes.append(latitude)
    longitudes.append(longitude)

aed5.loc[:, 'latitude'] = latitudes
aed5.loc[:, 'longitude'] = longitudes

# Save dataset with added coordinates
aed5.to_csv('DATA/aed5.csv', index=False)

# This is done for the datasets aed and mug
