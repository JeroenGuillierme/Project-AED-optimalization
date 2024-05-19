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
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics



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
df1["City Permanence"] = df1["City Permanence"].str.extract(r'\((.*?)\)')
df1["City Intervention"] = df1["City Intervention"].str.extract(r'\((.*?)\)')
df1["Eventlevel"] = df1["Eventlevel"].str.replace("A","")
df1["Eventlevel"] = df1["Eventlevel"].str.replace("B","")


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
df2["City Permanence"] = df2["City Permanence"].str.extract(r'\((.*?)\)')
df2["City Intervention"] = df2["City Intervention"].str.extract(r'\((.*?)\)')
df2["Eventlevel"] = df2["Eventlevel"].str.replace("A","")
df2["Eventlevel"] = df2["Eventlevel"].str.replace("B","")


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
df3["City Permanence"] = df3["City Permanence"].str.extract(r'\((.*?)\)')
df3["City Intervention"] = df3["City Intervention"].str.extract(r'\((.*?)\)')
df3["Eventlevel"] = df3["Eventlevel"].str.replace("A","")
df3["Eventlevel"] = df3["Eventlevel"].str.replace("B","")


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
df5["Eventlevel"] = df5["Eventlevel"].str.replace("BUITENDIENSTSTELLING", "")
df5["Eventlevel"] = df5["Eventlevel"].str.replace("INTERVENTIEPLAN", "")
df5["Eventlevel"] = df5["Eventlevel"].str.replace("0", "")


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
df6["City Permanence"] = df6["City Permanence"].str.split(" ").str[1]
valid_categories = ["N1","N2", "N3", "N4", "N5", "N6", "N7", "N8"]
df6.loc[~df6['Eventlevel'].isin(valid_categories), 'Eventlevel'] = 'OTHER'


frames = [df1, df2, df3, df4, df5, df6]
data = pd.concat(frames)


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


data.to_csv('DATA/interventions.csv', index=False)
"""


##### DATA PREPROCESSING #####
interventions = pd.read_csv("DATA/interventions.csv")

data = interventions[["Province", "Vector", "Eventlevel", "Time1", "Time2"]]

## Nagaan hoeveel data er ontbreekt
#print(data.isna().sum().sort_values()/len(data))

# Remove missing observations
data = data.dropna(subset=['Province','Vector','Eventlevel','Time1','Time2'])


# Vanaf hier data opsplitsen in train en test
train, test = train_test_split(data, random_state=21) #784161 observaties


y1_train = train['Time1']
y1_test = test['Time1']
y2_train = train['Time2']
y2_test = test['Time2']
X_train = train[['Province', 'Vector', 'Eventlevel']] # observaties na verwijderen van missing data
X_test = test[['Province', 'Vector', 'Eventlevel']]

# Encode de categorische variabelen
#print(data["Province"].value_counts()) #11 regio's met elk voldoende observaties
#print(data["Vector"].value_counts()) #5 vector types
#print(data["Eventlevel"].value_counts()) #10 eventlevels
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_train)
X_train = encoder.transform(X_train).toarray()
X_test = encoder.transform(X_test).toarray()


# Isolationforest (voor outliers)
IsoFo = IsolationForest(n_estimators=100, contamination= 'auto')
y1_labels = IsoFo.fit_predict(np.array(y1_train).reshape(-1,1))
#y2_labels = IsoFo.fit_predict(np.array(y2_train).reshape(-1,1))

# Only including the inliers

#voor time1
y1_train_filtered = y1_train[y1_labels == 1]
X1_train_filtered = np.array(X_train[y1_labels == 1]) # observaties na verwijderen van outliers
print(min(y1_train[y1_labels == -1])) #verwijderde outliers: meer dan 2767sec of 46min


#voor time2
#y2_train_filtered = y2_train[y2_labels == 1]
#X2_train_filtered = np.array(X_train[y2_labels == 1]).reshape(-1,1)



### Random Forest Regression


# Define parameters: these will need to be tuned to prevent overfitting and underfitting
params = {
    "n_estimators": 100,  # Number of trees in the forest
    "max_depth": 10,  # Max depth of the tree
    "min_samples_split": 4,  # Min number of samples required to split a node
    "min_samples_leaf": 2,  # Min number of samples required at a leaf node
    "ccp_alpha": 0,  # Cost complexity parameter for pruning
    "random_state": 123,
}


# Create a RandomForestRegressor object with the parameters above
rf = RandomForestRegressor(**params)

# Train the random forest on the train set
rf = rf.fit(X1_train_filtered, y1_train_filtered)

# Predict the outcomes on the test set
y1_pred = rf.predict(X_test)

# Evaluate performance with error metrics
print("Mean Absolute Error:", metrics.mean_absolute_error(y1_test, y1_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y1_test, y1_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y1_test, y1_pred)))


# Create a sorted Series of features importances
importances_sorted = pd.Series(data=rf.feature_importances_, index=pd.DataFrame(X1_train_filtered).columns).sort_values()

# Plot a horizontal barplot of importances_sorted
importances_sorted.plot(kind="barh")
plt.title("Features Importances")
plt.show()


### Hyperparameter tuning with random search
# Define a parameter grid with distributions of possible parameters to use
rs_param_grid = {
    "n_estimators": list((range(20, 200))),
    "max_depth": list((range(3, 12))),
    "min_samples_split": list((range(2, 5))),
    "min_samples_leaf": list((range(1, 5))),
    "ccp_alpha": [0, 0.001, 0.01, 0.1],
}

# Create a RandomForestRegressor
rf = RandomForestRegressor(random_state=123)

# Instantiate RandomizedSearchCV() with rf and the parameter grid
rf_rs = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rs_param_grid,
    cv=3,  # Number of folds
    n_iter=10,  # Number of parameter candidate settings to sample
    verbose=2,  # The higher this is, the more messages are outputed
    scoring="neg_mean_absolute_error",  # Metric to evaluate performance
    random_state=123
)

# Train the model on the training set
rf_rs.fit(X1_train_filtered, y1_train_filtered)

# Print the best parameters and highest accuracy
print("Best parameters found: ", rf_rs.best_params_)
print("Best performance: ", rf_rs.best_score_)
