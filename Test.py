
### onehot enoder (chatgpt)
from sklearn.preprocessing import OneHotEncoder

# Definieer X en y (features en target)
X = data[['CityName_permanence', 'CityName_intervention', 'Vector_type', 'eventLevel_Firstcall']]
y = data['Time1']

# Maak een OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Encode de categorische variabelen
X_encoded = encoder.fit_transform(X)


####ANALYSIS
# Split de data in trainings- en testsets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Maak en train het regressiemodel
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Voer voorspellingen uit op de testset
predictions = model.predict(X_test)

# Bereken de Mean Squared Error (MSE) om de prestaties van het model te evalueren
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

#### imputers (github les) (missing values veranderen)
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
df = pd.DataFrame(
    {
        'X': [15,9,np.nan,4],
        'Y': ['Yes','Yes','No', None]
    }
)
df#hier onze eigen dataset gebruiken; x = time 1; y: alle variablen
SI_numerical = SimpleImputer(missing_values=np.nan, strategy='mean')
SI_categorical = SimpleImputer(missing_values=None, strategy='most_frequent')

df.X=SI_numerical.fit_transform(df['X'].values.reshape(-1,1))
df.Y = SI_categorical.fit_transform(df['Y'].values.reshape(-1,1)).flatten()
df


<<<<<<< HEAD
#isolationforest (voor outliers)
#Necessary Imports
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from plotly.offline import iplot
import plotly.graph_objects as go
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)
from sklearn.linear_model import LinearRegression

df = pd.read_csv('../4.1 - Introduction to Scikit-Learn//Data/Phones.csv', index_col=0)
IsoFo = IsolationForest(n_estimators=100, contamination= 'auto')
labels = IsoFo.fit_predict(np.array(df.calls).reshape(-1,1))
#Only including the inliers
calls_filtered = df.calls[labels == 1]
date_filtered = np.array(df.year[labels == 1]).reshape(-1,1)
OLS_filtered = LinearRegression()
OLS_filtered.fit(date_filtered, calls_filtered)
=======
aed1 = aed.head(50)
print(aed1)

naam = "helena)"
print(naam.replace(")",""))

print(sum(aed["address"] == "None"))
>>>>>>> main
