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
import joblib


interventions = pd.read_csv("DATA/interventions.csv")
pd.set_option('display.max_columns', None)
data = interventions[["Province", "Vector", "Eventlevel", "Time1", "Time2"]] # 1045549 observations


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## Check how much data is missing
#print(data.isna().sum().sort_values()/len(data))

## Remove rows with missing values
data = data.dropna(subset=['Province','Vector','Eventlevel','Time1','Time2']) # 622375 observations after removal of missing data

#print("min(Time1) = ",min(data["Time1"])) # min Time1 = 13s
#print("max(Time1) = ",max(data["Time1"])) # max Time1 = 4820918s

## Split data in train and test set
train, test = train_test_split(data, random_state=21) # 466781 observations in train set (75% of data is used as train set)

## Target variable: Time1 (and Time2)
y1_train = train['Time1']
y1_test = test['Time1']
#y2_train = train['Time2']
#y2_test = test['Time2']

## Feature variables: Province, Vector, Eventlevel
X_train = train[['Province', 'Vector', 'Eventlevel']]
X_test = test[['Province', 'Vector', 'Eventlevel']]

## Encoding categorical variables (Province, Vector, Eventlevel) using OneHotEncoder
#print(data["Province"].value_counts()) # 11 regions with each having enough observations
#print(data["Vector"].value_counts()) # 5 vector types
#print(data["Eventlevel"].value_counts()) # 10 eventlevels
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_train)
X_train = encoder.transform(X_train).toarray()
X_test = encoder.transform(X_test).toarray()

## Link names of encoded variables to names of original categories
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

## Outlier detection using Isolation Forest
IsoFo = IsolationForest(n_estimators=100, contamination= 'auto', random_state=31)
y1_labels = IsoFo.fit_predict(np.array(y1_train).reshape(-1,1))

## Only including the inliers
y1_train_filtered = y1_train[y1_labels == 1]
X1_train_filtered = np.array(X_train[y1_labels == 1]) # 420393 observations after removal of outliers
#print("smallest removed outlier: ",min(y1_train[y1_labels == -1])) # removed outliers > 2812sec or 47min
#print("mean(Time1) = ",y1_train_filtered.mean()) # mean(Time1) = 826sec or 13,7min
#print("min(Time1): ",min(y1_train_filtered)) # min(Time1) = 18s
#print("max(Time1): ",max(y1_train_filtered)) # max(Time1) = 2811sec or 46min
#print(y1_train_filtered[y1_train_filtered < 120]) # 1821 observations have Time1 < 120s or 2min

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RANDOM FOREST REGRESSION
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
### Hyperparameter tuning with random search
## Define a parameter grid with distributions of possible parameters to use
rs_param_grid = {
    "n_estimators": list((range(20, 200))),
    "max_depth": list((range(3, 12))),
    "min_samples_split": list((range(2, 5))),
    "min_samples_leaf": list((range(1, 5))),
    "ccp_alpha": [0, 0.001, 0.01, 0.1],
}

## Create a RandomForestRegressor
rf = RandomForestRegressor(random_state = 123)

## Instantiate RandomizedSearchCV() with rf and the parameter grid
rf_rs = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rs_param_grid,
    cv=3,  # Number of folds
    n_iter=10,  # Number of parameter candidate settings to sample
    verbose=2,  # The higher this is, the more messages are outputed
    scoring="neg_mean_absolute_error",  # Metric to evaluate performance
    random_state=123
)

## Train the model on the training set
rf_rs.fit(X1_train_filtered, y1_train_filtered)

## Print the best parameters and highest accuracy
print("Best parameters found: ", rf_rs.best_params_) #"n_estimators": 110, "max_depth": 11, "min_samples_split":4, "min_samples_leaf": 1, "ccp_alpha": 0, "random_state": 123
print("Best performance: ", rf_rs.best_score_)
"""

### Random Forest Regressiong using best parameters
params = {
    "n_estimators": 110,  # Number of trees in the forest
    "max_depth": 11,  # Max depth of the tree
    "min_samples_split": 4,  # Min number of samples required to split a node
    "min_samples_leaf": 1,  # Min number of samples required at a leaf node
    "ccp_alpha": 0,  # Cost complexity parameter for pruning
    "random_state": 123,
}

## Create a RandomForestRegressor object with the parameters above
rf = RandomForestRegressor(**params)

## Train the random forest on the train set
rf = rf.fit(X1_train_filtered, y1_train_filtered)

## Predict the outcomes on the test set
y1_pred = rf.predict(X_test)

## Evaluate performance with error metrics
#print("Mean Absolute Error:", metrics.mean_absolute_error(y1_test, y1_pred)) #Mean Absolute Error: 150195.8634726495
#print("Mean Squared Error:", metrics.mean_squared_error(y1_test, y1_pred)) #Mean Squared Error: 412718371777.74756
#print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y1_test, y1_pred))) #Root Mean Squared Error: 642431.6086384197

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# VISUALISATION RESULT RANDOM FOREST
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Create a sorted series of feature importances
importances_sorted = pd.Series(data=rf.feature_importances_, index=pd.DataFrame(X1_train_filtered).columns).sort_values()

province_importances = importances_sorted[0:11]
vector_importances = importances_sorted[11:16]
eventlevel_importances = importances_sorted[16:27]

total_province_importance = sum(province_importances)/11 # importance of Province = 0.0018099858177990054
total_vector_importance = sum(vector_importances)/5 #  importance of Vector = 0.007812187325386341
total_eventlevel_importance = sum(eventlevel_importances)/10 # importance of Eventlevel = 0.09410292193772793
#print("importance of Province = ", total_province_importance)
#print("importance of Vector = ", total_vector_importance)
#print("importance of Eventlevel = ", total_eventlevel_importance)


## Visualisation of feature importances
categories = ['Province', 'Vector', 'Eventlevel']
total_importances = [total_province_importance, total_vector_importance, total_eventlevel_importance]
plt.bar(categories, total_importances)
plt.xlabel('Feature Variables')
plt.ylabel('Total Importance')
plt.title('Feature Importances')
plt.show()

## Create a sorted series of category importances
category_names = ["Province_" + str(i) for i in range(11)] + \
                ["Vector_" + str(i) for i in range(5)] + \
                ["Eventlevel_" + str(i) for i in range(11)]

sorted_indices = importances_sorted.argsort()[::-1]
sorted_importances = importances_sorted[sorted_indices]
sorted_feature_names = [category_names[i] for i in sorted_indices]
print("sorted_indices: ", sorted_indices)
print("sorted_importances: ", sorted_importances)
print("sorted_feature_names: ", sorted_feature_names)

## Visualisation of category importances
plt.barh(sorted_feature_names, sorted_importances)
plt.xlabel('Importance')
plt.title('Category Importances')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREDICTION - END OUTPUT
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Prediction of Time1 using Random Forest Regression model
## Threshold for importance
threshold = 0.025

## Select categories with importance > threshold
important_categories = [feature for feature, importances_sorted in zip(sorted_feature_names, sorted_importances) if importances_sorted >= threshold]

print("Important categories:", important_categories) # ['Eventlevel_5', 'Eventlevel_4', 'Vector_3', 'Vector_0', 'Province_8', 'Province_0']

## Only keep important categories in dataset
Xtrain_important = X1_train_filtered[:,[21,20,14,11,8,0]]
Xtest_important = X_test[:,[21,20,14,11,8,0]]

## Train a new Random Forest Regressor model using only important categories
new_model = RandomForestRegressor(**params)
new_model.fit(Xtrain_important, y1_train_filtered)

## Evaluate the new model
y_test_pred = new_model.predict(Xtest_important)
test_mse = mean_squared_error(y1_test, y_test_pred)
print("Mean Squared Error of new model:", test_mse)

## Predict Time1 using the new model
predicted_response_time = new_model.predict([[0,0,0,0,0,0]])
print("Predicted response time (in seconds):", predicted_response_time[0])

## Save model to file 'ResponseTimeModel.joblib'
joblib.dump(new_model, 'ResponseTimeModel.joblib')


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# USE IN APP
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Load model from file 'ResponseTimeModel.joblib'
#model = joblib.load('ResponseTimeModel.joblib')
#model.predict([vectorMetLengteZes])[0]

def give_predicted_response_time(vectorWithLengthSix):
    predicted_total_seconds = new_model.predict([vectorWithLengthSix])[0]
    minutes = int(predicted_total_seconds // 60)
    seconds = int(predicted_total_seconds % 60)
    return [minutes, seconds]

vector = [0,0,0,0,0,0]
print(give_predicted_response_time(vector))
print("Predicted response time: ",give_predicted_response_time(vector)[0], " minutes and ",give_predicted_response_time(vector)[1], " seconds")

