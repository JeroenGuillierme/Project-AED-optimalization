import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


# Custom transformer for IsolationForest
class IsolationForestTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, contamination='auto', random_state=None):
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = IsolationForest(contamination=self.contamination, random_state=self.random_state)

    def fit(self, X, y=None):
        self.iso_forest.fit(X)
        return self

    def transform(self, X, y=None):
        is_inlier = self.iso_forest.predict(X) == 1
        return X[is_inlier], is_inlier


# Load data
interventions = pd.read_csv("DATA/interventions.csv")
pd.set_option('display.max_columns', None)

# Select relevant feature variables
data = interventions[["Province", "Vector", "Eventlevel", "Time1", "Time2"]]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Check how much data is missing
#print(data.isna().sum().sort_values()/len(data))

## Remove rows with missing values
data = data.dropna(subset=['Province', 'Vector', 'Eventlevel', 'Time1', 'Time2'])

## Split data into train and test sets
train, test = train_test_split(data, random_state=21)

## Target variable: Time1 (and Time2)
y1_train = train['Time1']
y1_test = test['Time1']
#y2_train = train['Time2']
#y2_test = test['Time2']

## Feature variables: Province, Vector, Eventlevel
X_train = train[['Province', 'Vector', 'Eventlevel']]
X_test = test[['Province', 'Vector', 'Eventlevel']]

## Define preprocessing pipeline for categorical features
categorical_features = ['Province', 'Vector', 'Eventlevel']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess features and outlier removal
X_train_preprocessed = preprocessor.fit_transform(X_train)
outlier_removal = IsolationForestTransformer(contamination='auto', random_state=31)
X_train_filtered, inliers = outlier_removal.fit_transform(X_train_preprocessed, y1_train.values.reshape(-1, 1))
y1_train_filtered = y1_train[inliers]

# Define the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor',
     RandomForestRegressor(n_estimators=110, max_depth=11, min_samples_split=4, min_samples_leaf=1, ccp_alpha=0,
                           random_state=123))
])

# Fit the regressor
pipeline.named_steps['regressor'].fit(X_train_filtered, y1_train_filtered)

# Predict on the test set
X_test_preprocessed = preprocessor.transform(X_test)
y1_pred = pipeline.named_steps['regressor'].predict(X_test_preprocessed)

# Evaluate the model
test_mse = mean_squared_error(y1_test, y1_pred)
print("Mean Squared Error on the test set:", test_mse)

# Get feature importances from the trained model
final_model = pipeline.named_steps['regressor']
importances = final_model.feature_importances_
feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
    categorical_features)
importances_sorted = pd.Series(importances, index=feature_names).sort_values()

# Visualize feature importances per category
province_importances = importances_sorted[:11]
vector_importances = importances_sorted[11:16]
eventlevel_importances = importances_sorted[16:]

total_province_importance = province_importances.sum()
total_vector_importance = vector_importances.sum()
total_eventlevel_importance = eventlevel_importances.sum()

categories = ['Province', 'Vector', 'Eventlevel']
total_importances = [total_province_importance, total_vector_importance, total_eventlevel_importance]

plt.bar(categories, total_importances)
plt.xlabel('Categorical Variables')
plt.ylabel('Total Importance')
plt.title('Total Feature Importances per Categorical Variable')
plt.show()

# Visualize individual feature importances
plt.barh(importances_sorted.index, importances_sorted)
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()

# Threshold-based feature selection for prediction
threshold = 0.025
important_features = importances_sorted[importances_sorted >= threshold].index
X_train_important = pd.DataFrame(X_train_filtered.toarray(), columns=feature_names)[important_features]
X_test_important = pd.DataFrame(X_test_preprocessed.toarray(), columns=feature_names)[important_features]

# Train a new model with selected features
new_model = RandomForestRegressor(n_estimators=110, max_depth=11, min_samples_split=4, min_samples_leaf=1, ccp_alpha=0,
                                  random_state=123)
new_model.fit(X_train_important, y1_train_filtered)

# Predict and evaluate the new model
y_test_pred = new_model.predict(X_test_important)
test_mse_new_model = mean_squared_error(y1_test, y_test_pred)
print("Mean Squared Error on the test set of the new model:", test_mse_new_model)

# Predict the response time
predicted_response_time = new_model.predict(np.zeros((1, len(important_features))))
print("Predicted response time (in seconds):", predicted_response_time[0])

# Voorspel de responstijd
predicted_response_time = new_model.predict([[0,0,0,0,0,0]])
# Print de voorspelde responstijd
print("Voorspelde responstijd (in seconden):", predicted_response_time[0])

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# USED IN APP
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def give_predicted_response_time(vectorMetLengteZes):
    predicted_total_seconds = new_model.predict([vectorMetLengteZes])[0]
    minutes = int(predicted_total_seconds // 60)
    seconds = int(predicted_total_seconds % 60)
    return [minutes, seconds]

vector = [0,0,0,0,0,0]
print(give_predicted_response_time(vector))
print("Voorspelde responstijd: ",give_predicted_response_time(vector)[0], " minuten en ",give_predicted_response_time(vector)[1], " seconden")
