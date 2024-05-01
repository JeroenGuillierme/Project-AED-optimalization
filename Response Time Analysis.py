import pandas as pd

ambulance = pd.read_parquet('ambulance_locations.parquet.gzip')
mug = pd.read_parquet('mug_locations.parquet.gzip')
pit = pd.read_parquet('pit_locations.parquet.gzip')
interventions1 = pd.read_parquet('interventions1.parquet.gzip')
interventions2 = pd.read_parquet('interventions2.parquet.gzip')
interventions3 = pd.read_parquet('interventions3.parquet.gzip')
interventions4 = pd.read_parquet('interventions_bxl.parquet.gzip')
interventions5 = pd.read_parquet('interventions_bxl2.parquet.gzip')
cad = pd.read_parquet('cad9.parquet.gzip')
aed = pd.read_parquet('aed_locations.parquet.gzip')