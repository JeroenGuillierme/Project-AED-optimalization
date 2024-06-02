import pandas as pd

### Create dataset "interventions" and save as csv file

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IMPORTING DATASETS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ambulance = pd.read_parquet('https://raw.githubusercontent.com/JeroenGuillierme/Project-AED-optimalization/main/DATA/ambulance_locations.parquet.gzip')
mug = pd.read_parquet('https://raw.githubusercontent.com/JeroenGuillierme/Project-AED-optimalization/main/DATA/mug_locations.parquet.gzip')
pit = pd.read_parquet('https://raw.githubusercontent.com/JeroenGuillierme/Project-AED-optimalization/main/DATA/pit_locations.parquet.gzip')
interventions1 = pd.read_parquet('https://raw.githubusercontent.com/JeroenGuillierme/Project-AED-optimalization/main/DATA/interventions1.parquet.gzip')
interventions2 = pd.read_parquet('https://raw.githubusercontent.com/JeroenGuillierme/Project-AED-optimalization/main/DATA/interventions2.parquet.gzip')
interventions3 = pd.read_parquet('https://raw.githubusercontent.com/JeroenGuillierme/Project-AED-optimalization/main/DATA/interventions3.parquet.gzip')
interventions4 = pd.read_parquet('https://raw.githubusercontent.com/JeroenGuillierme/Project-AED-optimalization/main/DATA/interventions_bxl.parquet.gzip')
interventions5 = pd.read_parquet('https://raw.githubusercontent.com/JeroenGuillierme/Project-AED-optimalization/main/DATA/interventions_bxl2.parquet.gzip')
cad = pd.read_parquet('https://raw.githubusercontent.com/JeroenGuillierme/Project-AED-optimalization/main/DATA/cad9.parquet.gzip')
aed = pd.read_parquet('https://raw.githubusercontent.com/JeroenGuillierme/Project-AED-optimalization/main/DATA/aed_locations.parquet.gzip')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CREATE DATASET "INTERVENTIONS"
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

pd.set_option('display.max_columns', None)

## The datasets ambulance_locations, mug_locations, pit_locations and ead_locations don't contain time variables and are not used


## interventions1
interventions1["T0"] = pd.to_datetime(interventions1["T0"], format='%d%b%y:%H:%M:%S')
interventions1["T3"] = pd.to_datetime(interventions1["T3"], format='%Y-%m-%d %H:%M:%S.%f')
interventions1["T5"] = pd.to_datetime(interventions1["T5"], format='%Y-%m-%d %H:%M:%S.%f')
df1 = pd.DataFrame({
    "Province": interventions1["Province intervention"].str.upper(),
    "City Permanence": interventions1["CityName permanence"].str.upper(),
    "City Intervention": interventions1["CityName intervention"].str.upper(),
    "Vector": interventions1["Vector type"].str.upper(),
    "Eventlevel": interventions1["EventLevel Firstcall"].str.upper(),
    "Time1": interventions1["T3"]-interventions1["T0"],
    "Time2": interventions1["T5"]-interventions1["T0"]})
df1["City Permanence"] = df1["City Permanence"].str.extract(r'\((.*?)\)')
df1["City Intervention"] = df1["City Intervention"].str.extract(r'\((.*?)\)')
df1["Eventlevel"] = df1["Eventlevel"].str.replace("A","")
df1["Eventlevel"] = df1["Eventlevel"].str.replace("B","")


## interventions2
interventions2["T0"] = pd.to_datetime(interventions2["T0"], format='%d%b%y:%H:%M:%S')
interventions2["T3"] = pd.to_datetime(interventions2["T3"], format='%Y-%m-%d %H:%M:%S.%f')
interventions2["T5"] = pd.to_datetime(interventions2["T5"], format='%Y-%m-%d %H:%M:%S.%f')
df2 = pd.DataFrame({
                    "Province": interventions2["Province intervention"].str.upper(),
                    "City Permanence": interventions2["CityName permanence"].str.upper(),
                    "City Intervention": interventions2["CityName intervention"].str.upper(),
                    "Vector": interventions2["Vector type"].str.upper(),
                    "Eventlevel": interventions2["EventLevel Firstcall"].str.upper(),
                    "Time1": interventions2["T3"]-interventions2["T0"],
                    "Time2": interventions2["T5"]-interventions2["T0"]})
df2["City Permanence"] = df2["City Permanence"].str.extract(r'\((.*?)\)')
df2["City Intervention"] = df2["City Intervention"].str.extract(r'\((.*?)\)')
df2["Eventlevel"] = df2["Eventlevel"].str.replace("A","")
df2["Eventlevel"] = df2["Eventlevel"].str.replace("B","")


## interventions3
interventions3["T0"] = pd.to_datetime(interventions3["T0"], format='%d%b%y:%H:%M:%S')
interventions3["T3"] = pd.to_datetime(interventions3["T3"], format='%Y-%m-%d %H:%M:%S.%f')
interventions3["T5"] = pd.to_datetime(interventions3["T5"], format='%Y-%m-%d %H:%M:%S.%f')
df3 = pd.DataFrame({
                    "Province": interventions3["Province intervention"].str.upper(),
                    "City Permanence": interventions3["CityName permanence"].str.upper(),
                    "City Intervention": interventions3["CityName intervention"].str.upper(),
                    "Vector": interventions3["Vector type"].str.upper(),
                    "Eventlevel": interventions3["EventLevel Firstcall"].str.upper(),
                    "Time1": interventions3["T3"]-interventions3["T0"],
                    "Time2": interventions3["T5"]-interventions3["T0"]})
df3["City Permanence"] = df3["City Permanence"].str.extract(r'\((.*?)\)')
df3["City Intervention"] = df3["City Intervention"].str.extract(r'\((.*?)\)')
df3["Eventlevel"] = df3["Eventlevel"].str.replace("A","")
df3["Eventlevel"] = df3["Eventlevel"].str.replace("B","")


## interventions4
interventions4["t0"] = pd.to_datetime(interventions4["t0"], format='%Y-%m-%d %H:%M:%S.%f %z')
interventions4["t3"] = pd.to_datetime(interventions4["t3"], format='%Y-%m-%d %H:%M:%S.%f %z')
interventions4["t5"] = pd.to_datetime(interventions4["t5"], format='%Y-%m-%d %H:%M:%S.%f %z')
df4 = pd.DataFrame({
                    "Province": ["BRUSSEL"]*len(interventions4),
                    "City Permanence": interventions4["cityname_permanence"].str.upper(),
                    "City Intervention": interventions4["cityname_intervention"].str.upper(),
                    "Vector": interventions4["vector_type"].str.upper(),
                    "Eventlevel": interventions4["eventLevel_firstcall"].str.upper(),
                    "Time1": interventions4["t3"]-interventions4["t0"],
                    "Time2": interventions4["t5"]-interventions4["t0"]})
df4["City Permanence"] = df4["City Permanence"].str.split(" \(").str[0]
df4["City Permanence"] = df4["City Permanence"].replace("BRUXELLES", "BRUSSEL")
df4["City Intervention"] = df4["City Intervention"].str.split(" \(").str[0]
df4["City Intervention"] = df4["City Intervention"].replace("BRUXELLES", "BRUSSEL")


## interventions5
interventions5["T0"] = pd.to_datetime(interventions5["T0"], format='%d%b%y:%H:%M:%S')
interventions5["T3"] = pd.to_datetime(interventions5["T3"], format='%d%b%y:%H:%M:%S')
interventions5["T5"] = pd.to_datetime(interventions5["T5"], format='%d%b%y:%H:%M:%S')
df5 = pd.DataFrame({
                    "Province": ["BRUSSEL"]*len(interventions5),
                    "City Permanence": interventions5["Cityname Permanence"].str.upper(),
                    "City Intervention": interventions5["Cityname Intervention"].str.upper(),
                    "Vector": interventions5["Vector type NL"].str.upper(),
                    "Eventlevel": interventions5["EventType and EventLevel"].str.upper(),
                    "Time1": interventions5["T3"]-interventions5["T0"],
                    "Time2": interventions5["T5"]-interventions5["T0"]})
df5["City Permanence"] = df5["City Permanence"].str.extract(r'\((.*?)\)')
df5["City Permanence"] = df5["City Permanence"].replace("BRUXELLES", "BRUSSEL")
df5["City Intervention"] = df5["City Intervention"].str.extract(r'\((.*?)\)')
df5["City Intervention"] = df5["City Intervention"].replace("BRUXELLES", "BRUSSEL")
df5["Vector"] = df5["Vector"].replace("AMB", "AMBULANCE")
df5["Eventlevel"] = df5["Eventlevel"].str.split(" ").str[1]
df5["Eventlevel"] = df5["Eventlevel"].str.replace("BUITENDIENSTSTELLING", "")
df5["Eventlevel"] = df5["Eventlevel"].str.replace("INTERVENTIEPLAN", "")
df5["Eventlevel"] = df5["Eventlevel"].str.replace("0", "")


## cad
cad["T0"] = pd.to_datetime(cad["T0"], format='%Y-%m-%d %H:%M:%S.%f')
cad["T3"] = pd.to_datetime(cad["T3"], format='%Y-%m-%d %H:%M:%S.%f')
cad["T5"] = pd.to_datetime(cad["T5"], format='%Y-%m-%d %H:%M:%S.%f')
df6 = pd.DataFrame({
                    "Province": cad["province"].str.upper(),
                    "City Permanence": cad["Permanence long name"].str.upper(),
                    "City Intervention": cad["CityName intervention"].str.upper(),
                    "Vector": cad["Vector Type"].str.upper(),
                    "Eventlevel": cad["EventLevel Trip"].str.upper(),
                    "Time1": cad["T3"]-cad["T0"],
                    "Time2": cad["T5"]-cad["T0"]})
df6["City Permanence"] = df6["City Permanence"].str.split(" ").str[1]
valid_categories = ["N1","N2", "N3", "N4", "N5", "N6", "N7", "N8"]
df6.loc[~df6['Eventlevel'].isin(valid_categories), 'Eventlevel'] = 'OTHER'

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CONCATENATE DATAFRAMES
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

frames = [df1, df2, df3, df4, df5, df6]
data = pd.concat(frames)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MATCH NAMES OF CATEGORIES
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

data["Province"] = data["Province"].replace("ANT","ANTWERPEN")
data["Province"] = data["Province"].replace("OVL","OOST-VLAANDEREN")
data["Province"] = data["Province"].replace("WVL","WEST-VLAANDEREN")
data["Province"] = data["Province"].replace("HAI","HENEGOUWEN")
data["Province"] = data["Province"].replace("BRW","WAALS_BRABANT")
data["Province"] = data["Province"].replace("VBR","VLAAMS-BRABANT")
data["Province"] = data["Province"].replace("NAM","NAMEN")
data["Province"] = data["Province"].replace("LIE","LUIK")
data["Province"] = data["Province"].replace("LIM","LIMBURG")
data["Province"] = data["Province"].replace("LUX","LUXEMBURG")
data["Vector"] = data["Vector"].str.split(" ").str[0]
data["Time1"] = data["Time1"].dt.total_seconds().round()
data["Time1"][data["Time1"] < 0] = pd.NaT
data["Time2"] = data["Time2"].dt.total_seconds().round()
data["Time2"][data["Time2"] < 0] = pd.NaT

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SAVE DATASET AS CSV FILE
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

data.to_csv('https://raw.githubusercontent.com/JeroenGuillierme/Project-AED-optimalization/main/DATA/interventions.csv', index=False)
