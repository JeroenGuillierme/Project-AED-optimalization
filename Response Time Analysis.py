import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.ensemble import IsolationForest
from plotly.offline import iplot
import plotly.graph_objects as go
import plotly.offline as pyo
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta



"""
#Open data
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



#Ambulance, mug, pit en aed bevatten geen responstijd variabelen dus gebruiken we niet


# interventions1: CityName permanence, CityName intervention, Vector type, EventLevel Firstcall
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
df1["Province"] = df1["Province"].replace("ANT","ANTWERPEN")
df1["Province"] = df1["Province"].replace("OVL","OOST-VLAANDEREN")
df1["Province"] = df1["Province"].replace("WVL","WEST-VLAANDEREN")
df1["Province"] = df1["Province"].replace("HAI","HENEGOUWEN")
df1["Province"] = df1["Province"].replace("BRW","WAALS_BRABANT")
df1["Province"] = df1["Province"].replace("VBR","VLAAMS-BRABANT")
df1["City Permanence"] = df1["City Permanence"].str.extract(r'\((.*?)\)')
df1["City Intervention"] = df1["City Intervention"].str.extract(r'\((.*?)\)')
df1["Eventlevel"] = df1["Eventlevel"].str.replace("A","")
df1["Eventlevel"] = df1["Eventlevel"].str.replace("B","")
df1["Time1"] = df1['Time1'].dt.total_seconds().round()
df1["Time2"] = df1['Time2'].dt.total_seconds().round()


# interventions2: CityName permanence, CityName intervention, Vector type, EventLevel Firstcall
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
df2["Province"] = df2["Province"].replace("HAI","HENEGOUWEN")
df2["Province"] = df2["Province"].replace("LIE","LUIK")
df2["City Permanence"] = df2["City Permanence"].str.extract(r'\((.*?)\)')
df2["City Intervention"] = df2["City Intervention"].str.extract(r'\((.*?)\)')
df2["Time1"] = df2['Time1'].dt.total_seconds().round()
df2["Time2"] = df2['Time2'].dt.total_seconds().round()


# interventions3: CityName permanence, CityName intervention, Vector type, EventLevel Firstcall
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
df3["Province"] = df3["Province"].replace("LIE","LUIK")
df3["Province"] = df3["Province"].replace("LIM","LIMBURG")
df3["Province"] = df3["Province"].replace("NAM","NAMEN")
df3["Province"] = df3["Province"].replace("LUX","LUXEMBURG")
df3["City Permanence"] = df3["City Permanence"].str.extract(r'\((.*?)\)')
df3["City Intervention"] = df3["City Intervention"].str.extract(r'\((.*?)\)')
df3["Time1"] = df3['Time1'].dt.total_seconds().round()
df3["Time2"] = df3['Time2'].dt.total_seconds().round()


# interventions4: cityname_permanence, CityName intervention, vector_type, eventLevel_firstcall
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
df4["Time1"] = df4['Time1'].dt.total_seconds().round()
df4["Time2"] = df4['Time2'].dt.total_seconds().round()


#interventions5
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
df5["Eventlevel"] = df5["Eventlevel"].str.replace("0", "")
df5["Time1"] = df5['Time1'].dt.total_seconds().round()
df5["Time2"] = df5['Time2'].dt.total_seconds().round()


#cad
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
df6["Province"] = df6["Province"].replace("NAM","NAMEN")
df6["Province"] = df6["Province"].replace("WVL","WEST-VLAANDEREN")
df6["Province"] = df6["Province"].replace("OVL","OOST-VLAANDEREN")
df6["Province"] = df6["Province"].replace("VBR","VLAAMS-BRABANT")
df6["City Permanence"] = df6["City Permanence"].str.split(" ").str[1]
df6["Time1"] = df6['Time1'].dt.total_seconds().round()
df6["Time2"] = df6['Time2'].dt.total_seconds().round()
df6["Time1"][df6['Time1'] < 0] = pd.NaT
df6["Time2"][df6['Time2'] < 0] = pd.NaT


frames = [df1, df2, df3, df4, df5, df6]
data = pd.concat(frames)
data.to_csv('DATA/interventions.csv', index=False)
"""




##### DATA PREPROCESSING #####
data = pd.read_csv("DATA/interventions.csv")


# Vanaf hier data opsplitsen in train en test
train, test = train_test_split(data, random_state=1)

## Nagaan hoeveel data er ontbreekt
city_permanence_count = data['CityName permanence'].isna().sum()
#print(city_permanence_count/1045549) # 0.10775774258308314
city_intervention_count = data['CityName intervention'].isna().sum()
#print(city_intervention_count/1045549) # 0.005094930988408961
vector_count = data['Vector type'].isna().sum()
#print(vector_count/1045549) # 0.018477374087680253
level_count = data['EventLevel Firstcall'].isna().sum()
#print(level_count/1045549) # 0.026061906232993384
time1_count = data['Time1'].isna().sum()
#print(time1_count/1045549) # 0.2206171112018662
time2_count = data['Time2'].isna().sum()
#print(time2_count/1045549) # 0.3882247508246864


#missing data verwijderen
train = train.dropna(subset=[
    'CityName permanence',
    'CityName intervention',
    'Vector type',
    'EventLevel Firstcall',
    'Time1',
    'Time2'
])



y1_train = train['Time1']
y2_train = train['Time2']
X_train = train[['CityName permanence', 'CityName intervention', 'Vector type', 'EventLevel Firstcall']]


# Maak een OneHotEncoder
encoder = OneHotEncoder()
# Encode de categorische variabelen
#print(len(list(data["CityName intervention"].unique())))
encoder.fit(X_train)
X_train = encoder.transform(X_train).toarray()




## Nagaan hoeveel data er ontbreekt
city_permanence_count = data['CityName permanence'].isna().sum()
#print(city_permanence_count/1045549) # 0.10775774258308314
city_intervention_count = data['CityName intervention'].isna().sum()
#print(city_intervention_count/1045549) # 0.005094930988408961
vector_count = data['Vector type'].isna().sum()
#print(vector_count/1045549) # 0.018477374087680253
level_count = data['EventLevel Firstcall'].isna().sum()
#print(level_count/1045549) # 0.026061906232993384
time1_count = data['Time1'].isna().sum()
#print(time1_count/1045549) # 0.2206171112018662
time2_count = data['Time2'].isna().sum()
#print(time2_count/1045549) # 0.3882247508246864


#isolationforest (voor outliers)

#print(y1_train) # lengte = 404108
#print(y2_train) # lengte = 404108

IsoFo = IsolationForest(n_estimators=100, contamination= 'auto')
y1_labels = IsoFo.fit_predict(np.array(y1_train).reshape(-1,1))
y2_labels = IsoFo.fit_predict(np.array(y2_train).reshape(-1,1))

#Only including the inliers

#voor time1
y1_train_filtered = y1_train[y1_labels == 1]
X1_train = np.array(X_train[y1_labels == 1]).reshape(-1,1)

#voor time2
y2_train_filtered = y2_train[y2_labels == 1]
X2_train = np.array(X_train[y2_labels == 1]).reshape(-1,1)


#print(y1_train_filtered) # lengte = 374498 dus er zijn  outliers verwijderd
#print(y2_train_filtered) # lengte = 374365 dus er zijn  outliers verwijderd

print(y1_train[y1_labels == -1])

##nu klaar om regressie te doen
