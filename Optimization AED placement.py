
import pandas as pd
from pandas.compat import pyarrow

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

# (Helena) de locatie van de oude AED's ahv adres = straat (address) + nummer (number) + postcode (postal_code) of stad (municipality)
print(aed.address)
print(aed.number)
print(aed.postal_code)
print(aed.municipality) # is de stad dus zegt eigenlijk hetzelfde als de postcode

# de locatie van de interventies, ambulance (vertrek) ahv latitude en longitude, let op ze hebben verschillende benamingen voor de verschillende datasets van interventies
print(interventions4.latitude_intervention)
print(interventions4.longitude_intervention)

print(ambulance.latitude)
print(ambulance.longitude)

# test om de locaties van de ambulances in kaart te brengen; is gelukt maar weet nog niet goed wat we hier precies mee zijn haha
import folium
import pandas as pd

def main():
    # controleer of de kolommen 'Latitude' en 'Longitude' bestaan
    if 'latitude' not in ambulance.columns or 'longitude' not in ambulance.columns:
        print("Fout: 'Latitude' en/of 'Longitude' kolommen niet gevonden in de dataset.")
        return

    # verwijder rijen met ontbrekende waarden voor 'Latitude' en 'Longitude'
    df = ambulance.dropna(subset=['latitude', 'longitude'])

    # maak een kaart met het midden ergens in België
    my_map = folium.Map(location=[50.8503, 4.3517], zoom_start=8)

    # voeg markers toe voor elke locatie in de DataFrame
    for index, row in ambulance.iterrows():
        folium.Marker([row['latitude'], row['longitude']]).add_to(my_map)

    # bewaar de kaart naar een HTML-bestand
    my_map.save("map.html")

if __name__ == "__main__":
    main()

# test om adres om te zetten in latitude en longitude
# gelukt om 1 adres om te zetten, bij het omzetten van alle adressen ging dit heel traag vandaar proberen met eerste 50 adressen
import pandas as pd
import boto3

aed["municipality"] = aed["municipality"].replace("KalloBeveren-Waas)","Beveren")

aed5 = aed.iloc[14001:15227].copy()


# AWS-regio waarin Amazon Location Service is geconfigureerd
#region_name = 'us-east-1'

# AWS-credentials configureren
#session = boto3.Session(
#    aws_access_key_id='AKIA6GBMAUQCXW4TPTUE',
#    aws_secret_access_key='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',
#    region_name=region_name)

# Creëer een client voor Amazon Location Service
client = boto3.client('location')

# Functie om adres om te zetten naar coördinaten
def get_coordinates(street, house_number, city):
    try:
        # Adres omzetten in longitudinale en latitudinale coördinaten
        response = client.search_place_index_for_text(
            IndexName='explore.place.Esri',
            Text=f'{street} {house_number}, {city}, Belgium'
        )

        # Latitude en longitude verkrijgen uit de respons
        latitude = response['Results'][0]['Place']['Geometry']['Point'][1]
        longitude = response['Results'][0]['Place']['Geometry']['Point'][0]

        return latitude, longitude

    except Exception as e:
        print(f"Error occurred for {street} {house_number}, {city}: {str(e)}")
        return None, None

# Maak lege lijsten om coördinaten op te slaan
latitudes = []
longitudes = []

# Doorloop elk adres in de dataset en haal de coördinaten op
for index, row in aed5.iterrows():
    street = row['address']
    house_number = row['number']
    city = row['municipality']

    latitude, longitude = get_coordinates(street, house_number, city)
    latitudes.append(latitude)
    longitudes.append(longitude)

# Voeg de coördinaten toe aan de dataset
aed5.loc[:, 'latitude'] = latitudes
aed5.loc[:, 'longitude'] = longitudes

print(aed5)

# Sla de dataset op met de toegevoegde coördinaten
#<<<<<<< HEAD
aed5.to_csv('DATA/aed5.csv', index=False)
#=======
#aed.to_csv('updated_dataset.csv', index=False)

#>>>>>>> d8d7207b39d670cdaf633689ecfcc141df8e4c20
