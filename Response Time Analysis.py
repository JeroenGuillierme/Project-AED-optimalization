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


##### DATA PRE-PREPROCESSING #####
""" Ambulance, mug, pit en aed bevatten geen responstijd variabelen dus gebruiken we niet """


# interventions1: CityName permanence, CityName intervention, Vector type, EventLevel Firstcall
interventions1["T0"] = pd.to_datetime(interventions1["T0"], format='%d%b%y:%H:%M:%S')
interventions1["T3"] = pd.to_datetime(interventions1["T3"], format='%Y-%m-%d %H:%M:%S.%f')
interventions1["T5"] = pd.to_datetime(interventions1["T5"], format='%Y-%m-%d %H:%M:%S.%f')

df1 = pd.DataFrame({
    "CityName permanence": interventions1["CityName permanence"].str.upper(),
    "CityName intervention": interventions1["CityName intervention"].str.upper(),
    "Vector type": interventions1["Vector type"].str.upper(),
    "EventLevel Firstcall": interventions1["EventLevel Firstcall"].str.upper(),
    "Time1": interventions1["T3"]-interventions1["T0"],
    "Time2": interventions1["T5"]-interventions1["T0"]})
df1["CityName permanence"] = df1["CityName permanence"].str.extract(r'\((.*?)\)')
df1["CityName intervention"] = df1["CityName intervention"].str.extract(r'\((.*?)\)')
#print(df1["Time2"])

# interventions2: CityName permanence, CityName intervention, Vector type, EventLevel Firstcall
interventions2["T0"] = pd.to_datetime(interventions2["T0"], format='%d%b%y:%H:%M:%S')
interventions2["T3"] = pd.to_datetime(interventions2["T3"], format='%Y-%m-%d %H:%M:%S.%f')
interventions2["T5"] = pd.to_datetime(interventions2["T5"], format='%Y-%m-%d %H:%M:%S.%f')

df2 = pd.DataFrame({"CityName permanence": interventions2["CityName permanence"].str.upper(),
                    "CityName intervention": interventions2["CityName intervention"].str.upper(),
                    "Vector type": interventions2["Vector type"].str.upper(),
                    "EventLevel Firstcall": interventions2["EventLevel Firstcall"].str.upper(),
                    "Time1": interventions2["T3"]-interventions2["T0"],
                    "Time2": interventions2["T5"]-interventions2["T0"]})
df2["CityName permanence"] = df2["CityName permanence"].str.extract(r'\((.*?)\)')
df2["CityName intervention"] = df2["CityName intervention"].str.extract(r'\((.*?)\)')
#print(df2["Time2"])


# interventions3: CityName permanence, CityName intervention, Vector type, EventLevel Firstcall
interventions3["T0"] = pd.to_datetime(interventions3["T0"], format='%d%b%y:%H:%M:%S')
interventions3["T3"] = pd.to_datetime(interventions3["T3"], format='%Y-%m-%d %H:%M:%S.%f')
interventions3["T5"] = pd.to_datetime(interventions3["T5"], format='%Y-%m-%d %H:%M:%S.%f')

df3 = pd.DataFrame({"CityName permanence": interventions3["CityName permanence"].str.upper(),
                    "CityName intervention": interventions3["CityName intervention"].str.upper(),
                    "Vector type": interventions3["Vector type"].str.upper(),
                    "EventLevel Firstcall": interventions3["EventLevel Firstcall"].str.upper(),
                    "Time1": interventions3["T3"]-interventions3["T0"],
                    "Time2": interventions3["T5"]-interventions3["T0"]})
df3["CityName permanence"] = df3["CityName permanence"].str.extract(r'\((.*?)\)')
df3["CityName intervention"] = df3["CityName intervention"].str.extract(r'\((.*?)\)')
#print(df3["Time2"])

# interventions4: cityname_permanence, CityName intervention, vector_type, eventLevel_firstcall
interventions4["t0"] = pd.to_datetime(interventions4["t0"], format='%Y-%m-%d %H:%M:%S.%f %z')
interventions4["t3"] = pd.to_datetime(interventions4["t3"], format='%Y-%m-%d %H:%M:%S.%f %z')
interventions4["t5"] = pd.to_datetime(interventions4["t5"], format='%Y-%m-%d %H:%M:%S.%f %z')

df4 = pd.DataFrame({"CityName permanence": interventions4["cityname_permanence"].str.upper(),
                    "CityName intervention": interventions4["cityname_intervention"].str.upper(),
                    "Vector type": interventions4["vector_type"].str.upper(),
                    "EventLevel Firstcall": interventions4["eventLevel_firstcall"].str.upper(),
                    "Time1": interventions4["t3"]-interventions4["t0"],
                    "Time2": interventions4["t5"]-interventions4["t0"]})
df4["CityName permanence"] = df4["CityName permanence"].str.split(" \(").str[0]
df4["CityName permanence"] = df4["CityName permanence"].replace("BRUXELLES", "BRUSSEL")
df4["CityName intervention"] = df4["CityName intervention"].str.split(" \(").str[0]
df4["CityName intervention"] = df4["CityName intervention"].replace("BRUXELLES", "BRUSSEL")
#print(df4["Time2"])


#interventions5
interventions5["T0"] = pd.to_datetime(interventions5["T0"], format='%d%b%y:%H:%M:%S')
interventions5["T3"] = pd.to_datetime(interventions5["T3"], format='%d%b%y:%H:%M:%S')
interventions5["T5"] = pd.to_datetime(interventions5["T5"], format='%d%b%y:%H:%M:%S')

df5 = pd.DataFrame({"CityName permanence": interventions5["Cityname Permanence"].str.upper(),
                    "CityName intervention": interventions5["Cityname Intervention"].str.upper(),
                    "Vector type": interventions5["Vector type NL"].str.upper(),
                    "EventLevel Firstcall": interventions5["EventType and EventLevel"].str.upper(),
                    "Time1": interventions5["T3"]-interventions5["T0"],
                    "Time2": interventions5["T5"]-interventions5["T0"]})
df5["CityName permanence"] = df5["CityName permanence"].str.extract(r'\((.*?)\)')
df5["CityName permanence"] = df5["CityName permanence"].replace("BRUXELLES", "BRUSSEL")
df5["CityName intervention"] = df5["CityName intervention"].str.extract(r'\((.*?)\)')
df5["CityName intervention"] = df5["CityName intervention"].replace("BRUXELLES", "BRUSSEL")
df5["Vector type"] = df5["Vector type"].replace("AMB", "AMBULANCE")
df5["EventLevel Firstcall"] = df5["EventLevel Firstcall"].str.split(" ").str[1]
df5["EventLevel Firstcall"] = df5["EventLevel Firstcall"].str.replace("0", "")
#print(df5["Time2"])

#cad
cad["T0"] = pd.to_datetime(cad["T0"], format='%Y-%m-%d %H:%M:%S.%f')
cad["T3"] = pd.to_datetime(cad["T3"], format='%Y-%m-%d %H:%M:%S.%f')
cad["T5"] = pd.to_datetime(cad["T5"], format='%Y-%m-%d %H:%M:%S.%f')

df6 = pd.DataFrame({"CityName permanence": cad["Permanence long name"].str.upper(),
                    "CityName intervention": cad["CityName intervention"].str.upper(),
                    "Vector type": cad["Vector Type"].str.upper(),
                    "EventLevel Firstcall": cad["EventLevel Trip"].str.upper(),
                    "Time1": cad["T3"]-cad["T0"],
                    "Time2": cad["T5"]-cad["T0"]})
df6["CityName permanence"] = df6["CityName permanence"].str.split(" ").str[1]
df6["Time1"][df6["Time1"] < pd.Timedelta(0)] = pd.NaT
df6["Time2"][df6["Time2"] < pd.Timedelta(0)] = pd.NaT
#print(df6["Time2"])

frames = [df1, df2, df3, df4, df5, df6]

data = pd.concat(frames)
#print(data["CityName intervention"])

##### DATA PREPROCESSING #####

""" Vanaf hier data opsplitsen in train en test"""
train, test = train_test_split(data, random_state=1)
X_train = train[['CityName permanence', 'CityName intervention', 'Vector type', 'EventLevel Firstcall']]
y1_train = train['Time1']
y2_train = train['Time2']
print(X_train)

# Maak een OneHotEncoder
encoder = OneHotEncoder()
#encoder = OneHotEncoder(sparse_output=False)

# Encode de categorische variabelen
encoder.fit(X_train)
X_train = encoder.transform(X_train).toarray()
#X_train = encoder.fit_transform(X_train)
#print(X_train)

"""Moeten we hier geen PCA doen?"""

#### imputers (github les) (missing values veranderen)

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

"""Gingen we de ontbrekende waarden niet verwijderen?"""

#SI_numerical = SimpleImputer(missing_values=pd.NaT, strategy='mean') #missing value is NaT
#SI_categorical = SimpleImputer(missing_values=None, strategy='most_frequent') #missing value is None? maar na encoder denk ik dat er dan gewoon in alle kolommen een 0 staat dus geen None meer

#y1_train = SI_numerical.fit_transform(y1_train.values.reshape(-1,1))
#y2_train = SI_numerical.fit_transform(y2_train.values.reshape(-1,1))
#X_train = SI_categorical.fit_transform(X_train.values.reshape(-1,1)).flatten()
#print(y1_train)
#print(y2_train)
#print(X_train) #geeft rare foutmelding -> geraak er nog niet uit :(

#isolationforest (voor outliers)
"""Hoe kan je spreken van outliers als het gaat over categorische variabelen?"""
IsoFo = IsolationForest(n_estimators=100, contamination= 'auto')
labels_citynameP = IsoFo.fit_predict(np.array(X_train['CityName permanence']).reshape(-1,1))
labels_citynameI = IsoFo.fit_predict(np.array(X_train['CityName intervention']).reshape(-1,1))
labels_vector = IsoFo.fit_predict(np.array(X_train['Vector type']).reshape(-1,1))
labels_eventlevel = IsoFo.fit_predict(np.array(X_Train['EventLevel Firstcall']).reshape(-1,1))

#Only including the inliers
CityName_permanence_filtered = frames.X_Train['CityName permanence'][labels_citynameP == 1] 
CityName_intervention_filtered = frames.X_train['CityName intervention'][labels_citynameI == 1]
Vector_type_filtered = frames.X_train['Vector type'][labels_vector == 1]
EventLevel_Firstcall_filtered = frames.X_train['EventLevel Firstcall'][labels_eventlevel == 1]

y1_train = np.array(frames.y1_train[labels == 1]).reshape(-1,1)
y2_train = np.array(frames.y2_train[labels == 1]).reshape(-1,1)

##nu klaar om regressie te doen
