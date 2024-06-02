import pandas as pd
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

pd.set_option('display.max_columns', None)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INTERVENTION DATASET IS MADE (CSV)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Ambulance, mug, pit en aed bevatten geen responstijd variabelen dus gebruiken we niet


# interventions1: CityName permanence, CityName intervention, Vector type, EventLevel Firstcall
interventions1["T0"] = pd.to_datetime(interventions1["T0"], format='%d%b%y:%H:%M:%S')
interventions1["T3"] = pd.to_datetime(interventions1["T3"], format='%Y-%m-%d %H:%M:%S.%f')

df1 = pd.DataFrame({
    "CityName permanence": interventions1["CityName permanence"].str.upper(),
    "CityName intervention": interventions1["CityName intervention"].str.upper(),
    "Vector type": interventions1["Vector type"].str.upper(),
    "EventLevel Firstcall": interventions1["EventLevel Firstcall"].str.upper(),
    "Time1": interventions1["T3"]-interventions1["T0"]})
df1["CityName permanence"] = df1["CityName permanence"].str.extract(r'\((.*?)\)')
df1["CityName intervention"] = df1["CityName intervention"].str.extract(r'\((.*?)\)')
df1["Time1"] = df1['Time1'].dt.total_seconds().round()
#print(df1["Time1"])

# interventions2: CityName permanence, CityName intervention, Vector type, EventLevel Firstcall
interventions2["T0"] = pd.to_datetime(interventions2["T0"], format='%d%b%y:%H:%M:%S')
interventions2["T3"] = pd.to_datetime(interventions2["T3"], format='%Y-%m-%d %H:%M:%S.%f')

df2 = pd.DataFrame({"CityName permanence": interventions2["CityName permanence"].str.upper(),
                    "CityName intervention": interventions2["CityName intervention"].str.upper(),
                    "Vector type": interventions2["Vector type"].str.upper(),
                    "EventLevel Firstcall": interventions2["EventLevel Firstcall"].str.upper(),
                    "Time1": interventions2["T3"]-interventions2["T0"]})
df2["CityName permanence"] = df2["CityName permanence"].str.extract(r'\((.*?)\)')
df2["CityName intervention"] = df2["CityName intervention"].str.extract(r'\((.*?)\)')
df2["Time1"] = df2['Time1'].dt.total_seconds().round()
#print(df2["Time2"])


# interventions3: CityName permanence, CityName intervention, Vector type, EventLevel Firstcall
interventions3["T0"] = pd.to_datetime(interventions3["T0"], format='%d%b%y:%H:%M:%S')
interventions3["T3"] = pd.to_datetime(interventions3["T3"], format='%Y-%m-%d %H:%M:%S.%f')

df3 = pd.DataFrame({"CityName permanence": interventions3["CityName permanence"].str.upper(),
                    "CityName intervention": interventions3["CityName intervention"].str.upper(),
                    "Vector type": interventions3["Vector type"].str.upper(),
                    "EventLevel Firstcall": interventions3["EventLevel Firstcall"].str.upper(),
                    "Time1": interventions3["T3"]-interventions3["T0"]})
df3["CityName permanence"] = df3["CityName permanence"].str.extract(r'\((.*?)\)')
df3["CityName intervention"] = df3["CityName intervention"].str.extract(r'\((.*?)\)')
df3["Time1"] = df3['Time1'].dt.total_seconds().round()
#print(df3["Time2"])

# interventions4: cityname_permanence, CityName intervention, vector_type, eventLevel_firstcall
interventions4["t0"] = pd.to_datetime(interventions4["t0"], format='%Y-%m-%d %H:%M:%S.%f %z')
interventions4["t3"] = pd.to_datetime(interventions4["t3"], format='%Y-%m-%d %H:%M:%S.%f %z')

df4 = pd.DataFrame({"CityName permanence": interventions4["cityname_permanence"].str.upper(),
                    "CityName intervention": interventions4["cityname_intervention"].str.upper(),
                    "Vector type": interventions4["vector_type"].str.upper(),
                    "EventLevel Firstcall": interventions4["eventLevel_firstcall"].str.upper(),
                    "Time1": interventions4["t3"]-interventions4["t0"]})
df4["CityName permanence"] = df4["CityName permanence"].str.split(" \(").str[0]
df4["CityName permanence"] = df4["CityName permanence"].replace("BRUXELLES", "BRUSSEL")
df4["CityName intervention"] = df4["CityName intervention"].str.split(" \(").str[0]
df4["CityName intervention"] = df4["CityName intervention"].replace("BRUXELLES", "BRUSSEL")
df4["Time1"] = df4['Time1'].dt.total_seconds().round()
#print(df4["Time2"])


#interventions5
interventions5["T0"] = pd.to_datetime(interventions5["T0"], format='%d%b%y:%H:%M:%S')
interventions5["T3"] = pd.to_datetime(interventions5["T3"], format='%d%b%y:%H:%M:%S')

df5 = pd.DataFrame({"CityName permanence": interventions5["Cityname Permanence"].str.upper(),
                    "CityName intervention": interventions5["Cityname Intervention"].str.upper(),
                    "Vector type": interventions5["Vector type NL"].str.upper(),
                    "EventLevel Firstcall": interventions5["EventType and EventLevel"].str.upper(),
                    "Time1": interventions5["T3"]-interventions5["T0"]})
df5["CityName permanence"] = df5["CityName permanence"].str.extract(r'\((.*?)\)')
df5["CityName permanence"] = df5["CityName permanence"].replace("BRUXELLES", "BRUSSEL")
df5["CityName intervention"] = df5["CityName intervention"].str.extract(r'\((.*?)\)')
df5["CityName intervention"] = df5["CityName intervention"].replace("BRUXELLES", "BRUSSEL")
df5["Vector type"] = df5["Vector type"].replace("AMB", "AMBULANCE")
df5["EventLevel Firstcall"] = df5["EventLevel Firstcall"].str.split(" ").str[1]
df5["EventLevel Firstcall"] = df5["EventLevel Firstcall"].str.replace("0", "")
df5["Time1"] = df5['Time1'].dt.total_seconds().round()
#print(df5["Time2"])

#cad
cad["T0"] = pd.to_datetime(cad["T0"], format='%Y-%m-%d %H:%M:%S.%f')
cad["T3"] = pd.to_datetime(cad["T3"], format='%Y-%m-%d %H:%M:%S.%f')

df6 = pd.DataFrame({"CityName permanence": cad["Permanence long name"].str.upper(),
                    "CityName intervention": cad["CityName intervention"].str.upper(),
                    "Vector type": cad["Vector Type"].str.upper(),
                    "EventLevel Firstcall": cad["EventLevel Trip"].str.upper(),
                    "Time1": cad["T3"]-cad["T0"]})
df6["CityName permanence"] = df6["CityName permanence"].str.split(" ").str[1]
df6["Time1"][df6["Time1"] < pd.Timedelta(0)] = pd.NaT
df6["Time1"] = df6['Time1'].dt.total_seconds().round()
#print(df6["Time2"])

frames = [df1, df2, df3, df4, df5, df6]

data = pd.concat(frames)

data.to_csv('DATA/interventions.csv', index=False)
print(data["CityName intervention"])
