import pandas as pd

aed = pd.read_parquet('DATA/aed_locations.parquet.gzip')

print(aed)

aed1 = aed.head(50)
print(aed1)

naam = "helena)"
print(naam.replace(")",""))

print(sum(aed["address"] == "None"))

import os
import pandas as pd

# Maak een lijst van alle CSV-bestanden met volledige paden
csv_files = [
    'data/aed1.csv',
    'data/aed2.csv',
    'data/aed3.csv',
    'data/aed4.csv',
    'data/aed5.csv'
]

# Controleer of de bestanden bestaan en lees elk CSV-bestand in een DataFrame
dfs = []
for csv_file in csv_files:
    full_path = os.path.abspath(csv_file)
    if os.path.exists(full_path):
        try:
            df = pd.read_csv(full_path)
            dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"Leeg bestand of geen gegevens: {full_path}")
    else:
        print(f"Bestand niet gevonden: {full_path}")

# Voeg alle DataFrames samen tot één dataset als er DataFrames zijn
if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    # Schrijf de samengevoegde dataset naar een nieuw CSV-bestand
    output_path = os.path.abspath('data/aed_samengevoegd.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"CSV-bestanden succesvol samengevoegd tot '{output_path}'")
else:
    print("Geen bestanden samengevoegd omdat geen van de CSV-bestanden succesvol kon worden gelezen.")

aedLoc = pd.read_csv('data/aed_samengevoegd.csv')
print(aedLoc)
print(aed)
print(aedLoc['public'])

# Maak een nieuwe dataset waarbij 'public' gelijk is aan Y, Yes, Ja of Oui
aedLocPub = aedLoc[aedLoc['public'].isin(['Y', 'Oui-Ja'])]

# Schrijf de gefilterde dataset naar een nieuw CSV-bestand
aedLocPub.to_csv('data/aedLocPub.csv', index=False)

print(aedLoc['public'].unique()) # Respons = ['Y' nan 'N' 'y' 'Non-Nee' 'Oui-Ja' 'Ja' 'Oui' 'J']; dus bovenstaande is niet volledig en wat doen we met nan (NotANumber)?

# Bepaal de gemeenschappelijke kolommen
common_columns = [col for col in aed.columns if col in aedLoc.columns]

# Maak een set van tuples voor de gemeenschappelijke kolommen
set1 = set([tuple(row) for row in aed[common_columns].to_numpy()])
set2 = set([tuple(row) for row in aedLoc[common_columns].to_numpy()])

# Vind de verschillen
missing_in_aed = set2 - set1
missing_in_aedLoc = set1 - set2

print("Rijen die ontbreken in dataset1:")
for row in missing_in_aed:
    print(row)

print("Rijen die ontbreken in dataset2:")
for row in missing_in_aedLoc:
    print(row)

num_nan_longitude = aedLoc['longitude'].isna().sum()
num_nan_latitude = aedLoc['latitude'].isna().sum()

print(f"Aantal nan waarden in longitude: {num_nan_longitude}")
print(f"Aantal nan waarden in latitude: {num_nan_latitude}")

num_nan_address = aed['address'].isna().sum()
num_nan_number = aed['number'].isna().sum()
num_nan_municipality= aed['municipality'].isna().sum()

print(f"Aantal nan waarden in address: {num_nan_address}")
print(f"Aantal nan waarden in number: {num_nan_number}")
print(f"Aantal nan waarden in municipality: {num_nan_municipality}")

print(aed)

import folium
import pandas as pd

def main():
    # controleer of de kolommen 'latitude' en 'longitude' bestaan
    if 'latitude' not in aedLoc.columns or 'longitude' not in aedLoc.columns:
        print("Fout: 'Latitude' en/of 'Longitude' kolommen niet gevonden in de dataset.")
        return

    # verwijder rijen met ontbrekende waarden voor 'Latitude' en 'Longitude'
    df = aedLoc.dropna(subset=['latitude', 'longitude'])

    # maak een kaart met het midden ergens in België
    my_map = folium.Map(location=[50.8503, 4.3517], zoom_start=8)

    # voeg markers toe voor elke locatie in de DataFrame
    for index, row in aedLoc.iterrows():
        folium.Marker([row['latitude'], row['longitude']]).add_to(my_map)

    # bewaar de kaart naar een HTML-bestand
   # my_map.save("mapAED.html")

#if __name__ == "__main__":
    main()

from geopy.geocoders import Nominatim

# Maak een Nominatim geolocator object
#geolocator = Nominatim(user_agent="myGeocoder")

# Het adres dat je wilt omzetten
#adres = input("Voer het adres in: ")

# Gebruik de geolocator om het adres om te zetten naar coördinaten
#locatie = geolocator.geocode(adres)

#if locatie:
    #print(f"Adres: {adres}")
    #print(f"Breedtegraad: {locatie.latitude}")
    #print(f"Lengtegraad: {locatie.longitude}")
#else:
    #print("Adres niet gevonden")

interventions1 = pd.read_parquet('DATA/interventions1.parquet.gzip')
interventions2 = pd.read_parquet('DATA/interventions2.parquet.gzip')
interventions3 = pd.read_parquet('DATA/interventions3.parquet.gzip')
interventions4 = pd.read_parquet('DATA/interventions_bxl.parquet.gzip')
interventions5 = pd.read_parquet('DATA/interventions_bxl2.parquet.gzip')
interventions = pd.read_csv('DATA/interventions.csv')

# Filter de dataset voor rijen waar 'EventLevel Firstcall' gelijk is aan 'N5'
N5_subset = interventions1[interventions1['EventLevel Firstcall'] == 'N5']

# Print de gefilterde subset
print(N5_subset['EventType Firstcall'])

# Filter de dataset voor rijen waar 'EventLevel Firstcall' gelijk is aan 'N1'
N1_subset = interventions1[interventions1['EventLevel Firstcall'] == 'N1']

# Print de gefilterde subset
print(N1_subset['EventType Firstcall'])

print(interventions1['EventLevel Firstcall'].unique())
print(interventions2['EventLevel Firstcall'].unique())
print(interventions3['EventLevel Firstcall'].unique())


N8_subset = interventions[interventions['Eventlevel'] == 'N8']
print(N8_subset['Time1'])
print(max(interventions['Time1']))
print(max(N8_subset['Time1']))