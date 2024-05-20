import pandas as pd
from sklearn.model_selection import train_test_split
"""

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
# Ambulance, mug, pit en aed bevatten geen responstijd variabelen dus gebruiken we niet


# interventions1: CityName permanence, CityName intervention, Vector type, EventLevel Firstcall
interventions1["T0"] = pd.to_datetime(interventions1["T0"], format='%d%b%y:%H:%M:%S') # Convert the first time format to datetime
interventions1["T3"] = pd.to_datetime(interventions1["T3"], format='%Y-%m-%d %H:%M:%S.%f') # Convert the column back to datetime with the new format
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
interventions2["T0"] = pd.to_datetime(interventions2["T0"], format='%d%b%y:%H:%M:%S') # Convert the first time format to datetime
interventions2["T3"] = pd.to_datetime(interventions2["T3"], format='%Y-%m-%d %H:%M:%S.%f') # Convert the column back to datetime with the new format
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
interventions3["T0"] = pd.to_datetime(interventions3["T0"], format='%d%b%y:%H:%M:%S') # Convert the first time format to datetime
interventions3["T3"] = pd.to_datetime(interventions3["T3"], format='%Y-%m-%d %H:%M:%S.%f')# Convert the column back to datetime with the new format
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
interventions4["t0"] = pd.to_datetime(interventions4["t0"], format='%Y-%m-%d %H:%M:%S.%f %z')# Convert the first time format to datetime
interventions4["t3"] = pd.to_datetime(interventions4["t3"], format='%Y-%m-%d %H:%M:%S.%f %z') # Convert the column back to datetime with the new format
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
interventions5["T0"] = pd.to_datetime(interventions5["T0"], format='%d%b%y:%H:%M:%S')# Convert the first time format to datetime
interventions5["T3"] = pd.to_datetime(interventions5["T3"], format='%d%b%y:%H:%M:%S') # Convert the column back to datetime with the new format
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
cad["T0"] = pd.to_datetime(cad["T0"], format='%Y-%m-%d %H:%M:%S.%f') # Convert the column back to datetime with the new format
cad["T3"] = pd.to_datetime(cad["T3"], format='%Y-%m-%d %H:%M:%S.%f') # Convert the column back to datetime with the new format
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

#Vanaf hier data opsplitsen in train en test
train, test = train_test_split(data, random_state=1)
print(train)


### onehot enoder (chatgpt)
from sklearn.preprocessing import OneHotEncoder

# Definieer X en y (features en target)
X_train = train[['CityName permanence', 'CityName intervention', 'Vector type', 'EventLevel Firstcall']]
y1 = train['Time1']
y2 = train['Time2']

# Maak een OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Encode de categorische variabelen
X_train_encoded = encoder.fit_transform(X_train)
print(X_train_encoded)

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

from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
SI_numerical = SimpleImputer(missing_values=None, strategy='mean')
SI_categorical = SimpleImputer(missing_values= pd.NaT, strategy='most_frequent')

train.y1 =SI_numerical.fit_transform(train['y1'].values.reshape(-1,1))
train.y2 =SI_numerical.fit_transform(train['y2'].values.reshape(-1,1))
train.X_train_encoded = SI_categorical.fit_transform(train['X_train_encoded'].values.reshape(-1,1)).flatten()
print(train['y1'])
print(train['y2'])
print(train['X_train_encoded'])#geeft rare foutmelding -> geraak er nog niet uit :(


#isolationforest (voor outliers)
#Necessary Imports
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from plotly.offline import iplot
import plotly.graph_objects as go
import plotly.offline as pyo
from sklearn.linear_model import LinearRegression

IsoFo = IsolationForest(n_estimators=100, contamination= 'auto')
labels_citynameP = IsoFo.fit_predict(np.array(train.X_train_encoded['CityName permanence']).reshape(-1,1))
labels_citynameI = IsoFo.fit_predict(np.array(train.X_train_encoded['CityName intervention']).reshape(-1,1))
labels_vector = IsoFo.fit_predict(np.array(train.X_train_encoded['Vector type']).reshape(-1,1))
labels_eventlevel = IsoFo.fit_predict(np.array(train.X_train_encoded['EventLevel Firstcall']).reshape(-1,1))

#Only including the inliers
CityName_permanence_filtered = frames.X_train_encoded['CityName permanence'][labels_citynameP == 1]
CityName_intervention_filtered = frames.X_train_encoded['CityName intervention'][labels_citynameI == 1]
Vector_type_filtered = frames.X_train_encoded['Vector type'][labels_vector == 1]
EventLevel_Firstcall_filtered = frames.X_train_encoded['EventLevel Firstcall'][labels_eventlevel == 1]

y1_filtered = np.array(frames.y1[labels == 1]).reshape(-1,1)
y2_filtered = np.array(frames.y2[labels == 1]).reshape(-1,1)

##nu klaar om regressie te doen
"""

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
import statsmodels as sm



##### DATA PREPROCESSING #####
interventions = pd.read_csv("DATA/interventions.csv")

data = interventions[["Province", "Vector", "Eventlevel", "Time1", "Time2"]] # 1045549 observaties

## Nagaan hoeveel data er ontbreekt
#print(data.isna().sum().sort_values()/len(data))

# Remove missing observations
data = data.dropna(subset=['Province','Vector','Eventlevel','Time1','Time2']) # 622375 observaties na verwijderen van missing data

#print("min(Time1) = ",min(data["Time1"])) # min Time1 = 13s
#print("max(Time1) = ",max(data["Time1"])) # max Time1 = 4820918s

# Vanaf hier data opsplitsen in train en test
train, test = train_test_split(data, random_state=21) #466781 observaties in train (dus 75% van data wordt gebruikt als train)


y1_train = train['Time1']
y1_test = test['Time1']
y2_train = train['Time2']
y2_test = test['Time2']
X_train = train[['Province', 'Vector', 'Eventlevel']]
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
IsoFo = IsolationForest(n_estimators=100, contamination= 'auto', random_state=31)
y1_labels = IsoFo.fit_predict(np.array(y1_train).reshape(-1,1))
#y2_labels = IsoFo.fit_predict(np.array(y2_train).reshape(-1,1))

# Only including the inliers

#voor time1
y1_train_filtered = y1_train[y1_labels == 1]
X1_train_filtered = np.array(X_train[y1_labels == 1]) # 420393 observaties na verwijderen van outliers
#print("smallest removed outlier: ",min(y1_train[y1_labels == -1])) # removed outliers > 2812sec or 47min
#print("mean(Time1) = ",y1_train_filtered.mean()) # mean(Time1) = 826sec or 13,7min
#print("min(Time1): ",min(y1_train_filtered)) # min(Time1) = 18s -> impossible?!
#print("max(Time1): ",max(y1_train_filtered)) # max(Time1) = 2811sec or 46min
#print(y1_train_filtered[y1_train_filtered < 120]) # 1821 observations have Time1 < 120s or 2min

#voor time2
#y2_train_filtered = y2_train[y2_labels == 1]
#X2_train_filtered = np.array(X_train[y2_labels == 1]).reshape(-1,1)


### Random Forest Regression
"""

Het kiezen tussen ANOVA en supervised learning methoden zoals Random Forest of Gradient Boosting hangt sterk af van het doel van je analyse en de aard van je data. Hier is een vergelijking van ANOVA met Random Forest en Gradient Boosting, zodat je een weloverwogen beslissing kunt nemen.

ANOVA (Analysis of Variance)
Voordelen:
Interpretability: ANOVA is een statistische methode die eenvoudig te interpreteren is. Het helpt je te begrijpen of er significante verschillen zijn tussen de middelen van verschillende groepen.
Factor Analysis: Geschikt voor het analyseren van de effecten van categorische onafhankelijke variabelen (factoren) en hun interacties op een continue afhankelijke variabele.
Statistical Significance: ANOVA geeft directe p-waarden die de significantie van effecten aangeven.
Simplicity: Gemakkelijk toe te passen op kleinere datasets en minder complexe modellen.
Nadelen:
Assumpties: ANOVA gaat uit van normaal verdeelde residuen, homoscedasticiteit (gelijke varianties), en onafhankelijkheid van waarnemingen.
Limitations with Non-linear Relationships: Minder geschikt voor complexe, niet-lineaire relaties tussen variabelen.
Fixed Factors: Beperkt tot het analyseren van categorische variabelen; niet geschikt voor continue onafhankelijke variabelen.
Supervised Learning (Random Forest en Gradient Boosting)
Voordelen:
Flexibility: Kan omgaan met zowel categorische als continue onafhankelijke variabelen en kan complexe, niet-lineaire relaties modelleren.
Performance: Vaak superieur in voorspellende prestaties, vooral bij complexe datasets met veel variabelen.
Feature Importance: Kan inzicht geven in de relatieve belangrijkheid van verschillende kenmerken.
Handling of Large Datasets: Geschikt voor grote datasets en hoge-dimensionaliteit.
Nadelen:
Interpretability: Minder intuïtief te interpreteren dan ANOVA. De resultaten zijn complexer en vereisen meer inspanning om te begrijpen.
Hyperparameter Tuning: Vereist uitgebreide hyperparameter tuning om de beste prestaties te bereiken.
Computational Cost: Kan rekenintensief zijn, vooral voor grote datasets en complexe modellen."""

"""
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

#na ongeveer 30 min runnen resultaat beste parameters in volgende code aangepast

"""
# Define parameters: these will need to be tuned to prevent overfitting and underfitting
params = {
    "n_estimators": 110,  # Number of trees in the forest
    "max_depth": 11,  # Max depth of the tree
    "min_samples_split": 4,  # Min number of samples required to split a node
    "min_samples_leaf": 1,  # Min number of samples required at a leaf node
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



