import pandas as pd
from datetime import datetime

interventions1 = pd.read_parquet('DATA/interventions1.parquet.gzip')
interventions2 = pd.read_parquet('DATA/interventions2.parquet.gzip')
interventions3 = pd.read_parquet('DATA/interventions3.parquet.gzip')
interventions4 = pd.read_parquet('DATA/interventions_bxl.parquet.gzip')
interventions5 = pd.read_parquet('DATA/interventions_bxl2.parquet.gzip')
cad = pd.read_parquet('DATA/cad9.parquet.gzip')


def TimeDifference(t0, t3, t5):
    # Definieer de twee tijdsmomenten
    t0 = datetime.strptime("#05JAN23:04:08:01", "#%d%b%y:%H:%M:%S")
    t3 = datetime.strptime("2023-01-05 04:20:32.558", "%Y-%m-%d %H:%M:%S.%f")
    t5 = datetime.strptime("2023-01-05 04:20:32.558", "%Y-%m-%d %H:%M:%S.%f")
    # Bereken het verschil tussen de twee tijdsmomenten
    Time1 = t3 - t0
    Time2 = t5 - t0

    # Print het tijdsverschil
    return ( Time1, Time2)


pip install -U scikit-learn


