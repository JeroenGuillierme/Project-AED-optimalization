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
import joblib



model = joblib.load('ResponseTimeModel.joblib')

"""
##### DATA PREPROCESSING #####
interventions = pd.read_csv("DATA/interventions.csv")

data = interventions[["Province", "Vector", "Eventlevel", "Time1", "Time2"]] # 1045549 observaties

## Nagaan hoeveel data er ontbreekt
#print(data.isna().sum().sort_values()/len(data))

# Remove missing observations
data = data.dropna(subset=['Province','Vector','Eventlevel','Time1','Time2']) # 622375 observaties na verwijderen van missing data

#no encoding needed right? (en ook niet gedaan vermoed ik)
df = pd.DataFrame(data)

#model = ols(Time1 ~ C(Vector) + C(Eventlevel) + C(Province) +
        C(Vector):C(Eventlevel) + C(Vector):C(Province) + C(Eventlevel):C(Province) +
             C(Vector):C(Eventlevel):C(Province), data=df).fit()

#print(sm.stats.anova_lm(model, typ=2)) # allemaal heel significant
#print(model.summary())

#gls ipv ols
from statsmodels.regression.linear_model import GLS

# Definieer de gewichten op basis van de voorspelde waarden
weights = 1 / np.sqrt(model.fittedvalues)

# Pas GLS toe op de oorspronkelijke DataFrame met de gewichten
# GLS-model specificeren met exogene en endogene variabelen
gls_model = GLS(endog=df['Time1'], exog=df[['Vector', 'Eventlevel', 'Province']],
                weights=weights).fit()


# ANOVA
print(sm.stats.anova_lm(gls_model, typ=2))

# Samenvatting van het model
print(gls_model.summary())
"""




"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Residuen
residuals = model.resid

# Q-Q plot
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.show()

# Residual plot
plt.scatter(model.fittedvalues, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# Histogram van residuen
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.show()
"""




"""
ANOVA (Analysis of Variance)
Voordelen:
Interpretability: ANOVA is een statistische methode die eenvoudig te interpreteren is. Het helpt je te begrijpen of er significante verschillen zijn tussen de middelen van verschillende groepen.
Factor Analysis: Geschikt voor het analyseren van de effecten van categorische onafhankelijke variabelen (factoren) en hun interacties op een continue afhankelijke variabele.
Statistical Significance: ANOVA geeft directe p-waarden die de significantie van effecten aangeven.
Simplicity: Gemakkelijk toe te passen op kleinere datasets en minder complexe modellen.
Nadelen:
Assumpties: ANOVA gaat uit van normaal verdeelde residuen, homoscedasticiteit (gelijke varianties), en onafhankelijkheid van waarnemingen.
Limitations with Non-linear Relationships: Minder geschikt voor complexe, niet-lineaire relaties tussen variabelen.
Fixed Factors: Beperkt tot het analyseren van categorische variabelen; niet geschikt voor continue onafhankelijke variabelen."""