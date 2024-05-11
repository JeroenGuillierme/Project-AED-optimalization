import pandas as pd

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



## ambulance, mug, pit en aed bevatten geen responstijden dus gebruiken we niet


# interventions1: CityName permanence, CityName intervention, Vector type, EventLevel Firstcall
df1 = pd.DataFrame({
    "CityName permanence": interventions1["CityName permanence"].str.upper(),
    "CityName intervention": interventions1["CityName intervention"].str.upper(),
    "Vector type": interventions1["Vector type"].str.upper(),
    "EventLevel Firstcall": interventions1["EventLevel Firstcall"].str.upper()})
df1["CityName permanence"] = df1["CityName permanence"].str.extract(r'\((.*?)\)')
df1["CityName intervention"] = df1["CityName intervention"].str.extract(r'\((.*?)\)')
#print(df1["EventLevel Firstcall"])

# interventions2: CityName permanence, CityName intervention, Vector type, EventLevel Firstcall
df2 = pd.DataFrame({"CityName permanence": interventions2["CityName permanence"].str.upper(),
                    "CityName intervention": interventions2["CityName intervention"].str.upper(),
                    "Vector type": interventions2["Vector type"].str.upper(),
                    "EventLevel Firstcall": interventions2["EventLevel Firstcall"].str.upper()})
df2["CityName permanence"] = df2["CityName permanence"].str.extract(r'\((.*?)\)')
df2["CityName intervention"] = df2["CityName intervention"].str.extract(r'\((.*?)\)')
#print(df2["EventLevel Firstcall"])


# interventions3: CityName permanence, CityName intervention, Vector type, EventLevel Firstcall
df3 = pd.DataFrame({"CityName permanence": interventions3["CityName permanence"].str.upper(),
                    "CityName intervention": interventions3["CityName intervention"].str.upper(),
                    "Vector type": interventions3["Vector type"].str.upper(),
                    "EventLevel Firstcall": interventions3["EventLevel Firstcall"].str.upper()})
df3["CityName permanence"] = df3["CityName permanence"].str.extract(r'\((.*?)\)')
df3["CityName intervention"] = df3["CityName intervention"].str.extract(r'\((.*?)\)')
#print(df3["EventLevel Firstcall"])

# interventions4: cityname_permanence, CityName intervention, vector_type, eventLevel_firstcall
df4 = pd.DataFrame({"CityName permanence": interventions4["cityname_permanence"].str.upper(),
                    "CityName intervention": interventions4["cityname_intervention"].str.upper(),
                    "Vector type": interventions4["vector_type"].str.upper(),
                    "EventLevel Firstcall": interventions4["eventLevel_firstcall"].str.upper()})
df4["CityName permanence"] = df4["CityName permanence"].str.split(" \(").str[0]
df4["CityName permanence"] = df4["CityName permanence"].replace("BRUXELLES", "BRUSSEL")
df4["CityName intervention"] = df4["CityName intervention"].str.split(" \(").str[0]
df4["CityName intervention"] = df4["CityName intervention"].replace("BRUXELLES", "BRUSSEL")
#print(df4["EventLevel Firstcall"])


#interventions5
df5 = pd.DataFrame({"CityName permanence": interventions5["Cityname Permanence"].str.upper(),
                    "CityName intervention": interventions5["Cityname Intervention"].str.upper(),
                    "Vector type": interventions5["Vector type NL"].str.upper(),
                    "EventLevel Firstcall": interventions5["EventType and EventLevel"].str.upper()})
df5["CityName permanence"] = df5["CityName permanence"].str.extract(r'\((.*?)\)')
df5["CityName permanence"] = df5["CityName permanence"].replace("BRUXELLES", "BRUSSEL")
df5["CityName intervention"] = df5["CityName intervention"].str.extract(r'\((.*?)\)')
df5["CityName intervention"] = df5["CityName intervention"].replace("BRUXELLES", "BRUSSEL")
df5["Vector type"] = df5["Vector type"].replace("AMB", "AMBULANCE")
df5["EventLevel Firstcall"] = df5["EventLevel Firstcall"].str.split(" ").str[1]
df5["EventLevel Firstcall"] = df5["EventLevel Firstcall"].str.replace("0", "")
#print(df5["EventLevel Firstcall"])

#cad
df6 = pd.DataFrame({"CityName permanence": cad["Permanence long name"].str.upper(),
                    "CityName intervention": cad["CityName intervention"].str.upper(),
                    "Vector type": cad["Vector Type"].str.upper(),
                    "EventLevel Firstcall": cad["EventLevel Trip"].str.upper()})
df6["CityName permanence"] = df6["CityName permanence"].str.split(" ").str[1]
#print(df6["EventLevel Firstcall"])

frames = [df1, df2, df3, df4, df5, df6]

result = pd.concat(frames)
print(result)

