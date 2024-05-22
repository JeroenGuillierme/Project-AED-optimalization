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
from sklearn.metrics import mean_squared_error


##### DATA PREPROCESSING #####
interventions = pd.read_csv("DATA/interventions.csv")

pd.set_option('display.max_columns', None)

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

## Encoding linken aan originele variabelen
#traindata = pd.DataFrame(X_train)
#traindata.to_csv('DATA/XTrain.csv', index=False)
#frame = pd.DataFrame(X_train)
#frame.to_csv('DATA/EncodedXTrain.csv', index=False)
"""
0: Antwerpen
1: Brussel
2: Henegouwen
3: Limburg
4: Luik
5: Luxemburg
6: Namen
7: Oost-Vlaanderen
8: Vlaams-Brabant
9: Waals-Brabant
10: West-Vlaanderen
11: Ambulance
12: Brandziekenwagen
13: Decontaminatiewagen
14: Mug
15: Pit
16: N0
17: N1
18: N2
19: N3
20: N4
21: N5
22: N6
23: N7
24: N8
25: Other 
"""


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

"""
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
"""

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
"""

#na ongeveer 30 min runnen resultaat beste parameters in volgende code aangepast
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
#print("Mean Absolute Error:", metrics.mean_absolute_error(y1_test, y1_pred))
#print("Mean Squared Error:", metrics.mean_squared_error(y1_test, y1_pred))
#print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y1_test, y1_pred)))


# Create a sorted Series of features importances
importances_sorted = pd.Series(data=rf.feature_importances_, index=pd.DataFrame(X1_train_filtered).columns).sort_values()

#importances = model.feature_importances_
province_importances = importances_sorted[0:11]
vector_importances = importances_sorted[11:16]
eventlevel_importances = importances_sorted[16:27]

total_province_importance = sum(province_importances)
total_vector_importance = sum(vector_importances)
total_eventlevel_importance = sum(eventlevel_importances)


#visualisatie per categorie
categories = ['Province', 'Vector', 'Eventlevel']
total_importances = [total_province_importance, total_vector_importance, total_eventlevel_importance]

plt.bar(categories, total_importances)
plt.xlabel('Categorical Variables')
plt.ylabel('Total Importance')
plt.title('Total Feature Importances per Categorical Variable')
plt.show()


feature_names = ["Province_" + str(i) for i in range(11)] + \
                ["Vector_" + str(i) for i in range(5)] + \
                ["Eventlevel_" + str(i) for i in range(11)]

sorted_indices = importances_sorted.argsort()[::-1]
sorted_importances = importances_sorted[sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]


#visualisatie individueel
plt.barh(sorted_feature_names, sorted_importances)
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()


#prediction (end output)
# Stel een drempelwaarde in voor belangrijkheid
threshold = 0.025

# Selecteer de belangrijkste features op basis van de drempelwaarde
important_features = [feature for feature, importances_sorted in zip(sorted_feature_names, sorted_importances) if importances_sorted >= threshold]

print("Belangrijkste features:", important_features)#['Eventlevel_5', 'Eventlevel_4', 'Vector_3', 'Vector_0', 'Province_8', 'Province_0']

# Filter de dataset om alleen de belangrijke features te behouden
X_important = X1_train_filtered[:,[21,20,14,11,8,0]]

# Train een nieuw Random Forest Regressor model met de geselecteerde features
new_model = RandomForestRegressor(**params)
new_model.fit(X_important, y1_train_filtered)

# Evalueer het model
y_test_pred = new_model.predict(X_test[:,[21,20,14,11,8,0]])
test_mse = mean_squared_error(y1_test, y_test_pred)
print("Mean Squared Error op de testset van het nieuwe model:", test_mse)

# Voorspel de responstijd
predicted_response_time = new_model.predict([[0,0,0,0,0,0]])

# Print de voorspelde responstijd
print("Voorspelde responstijd (in seconden):", predicted_response_time[0])


# Converteer de voorspelde responstijd van seconden naar minuten en seconden
def convert_seconds_to_minutes_seconds(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return minutes, remaining_seconds

predicted_seconds = predicted_response_time[0]
minutes, seconds = convert_seconds_to_minutes_seconds(predicted_seconds)

print(f"Voorspelde responstijd: {minutes} minuten en {seconds} seconden")



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
Computational Cost: Kan rekenintensief zijn, vooral voor grote datasets en complexe modellen.
"""