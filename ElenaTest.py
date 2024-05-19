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
import statsmodels.api as sm
from statsmodels.formula.api import ols


##### DATA PREPROCESSING #####
interventions = pd.read_csv("DATA/interventions.csv")

data = interventions[["Province", "Vector", "Eventlevel", "Time1", "Time2"]] # 1045549 observaties

## Nagaan hoeveel data er ontbreekt
#print(data.isna().sum().sort_values()/len(data))

# Remove missing observations
data = data.dropna(subset=['Province','Vector','Eventlevel','Time1','Time2']) # 622375 observaties na verwijderen van missing data


df = pd.DataFrame(data)

model = ols("""Time1 ~ C(Vector) + C(Eventlevel) + C(Province) +
               C(Vector):C(Eventlevel) + C(Vector):C(Province) + C(Eventlevel):C(Province) +
               C(Vector):C(Eventlevel):C(Province)""", data=df).fit()

print(sm.stats.anova_lm(model, typ=2)) # allemaal heel significant
