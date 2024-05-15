import pandas as pd
from datetime import datetime
import numpy as np

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


interventions1["T0"] = pd.to_datetime(interventions1["T0"], format='%d%b%y:%H:%M:%S')# Convert the first time format to datetime
interventions1["T3"] = pd.to_datetime(interventions1["T3"], format='%Y-%m-%d %H:%M:%S.%f') # Convert the column back to datetime with the new format

interventions2["T0"] = pd.to_datetime(interventions2["T0"], format='%d%b%y:%H:%M:%S') # Convert the first time format to datetime
interventions2["T3"] = pd.to_datetime(interventions2["T3"], format='%Y-%m-%d %H:%M:%S.%f') # Convert the column back to datetime with the new format

interventions3["T0"] = pd.to_datetime(interventions3["T0"], format='%d%b%y:%H:%M:%S') # Convert the first time format to datetime
interventions3["T3"] = pd.to_datetime(interventions3["T3"], format='%Y-%m-%d %H:%M:%S.%f') # Convert the column back to datetime with the new format

interventions_total = pd.concat([interventions1, interventions2, interventions3], axis=0)
#print(interventions_total)

interventions_total['CAD9'] = 'N'
interventions_total["T3-T0"] = interventions_total["T3"]-interventions_total["T0"]
interventions_total = interventions_total[interventions_total["EventType Firstcall"] == "P003 - Cardiac arrest"]
print(interventions_total["EventLevel Firstcall"])
#print(interventions_total["T3-T0"])
interventions_total = interventions_total["Lattitude intervention", "Longitude intervention", "CAD9", "EventLevel Firstcall", "T0-T3"]

CAD9_expanded = cad
#print(CAD9_expanded.head())
CAD9_expanded['CAD9'] = 'Y'

CAD9_expanded["T0"] = pd.to_datetime(CAD9_expanded["T0"], format='%Y-%m-%d %H:%M:%S.%f') # Convert the column back to datetime with the new format
CAD9_expanded["T3"] = pd.to_datetime(CAD9_expanded["T3"], format='%Y-%m-%d %H:%M:%S.%f') # Convert the column back to datetime with the new format

CAD9_expanded["T3-T0"] = CAD9_expanded["T3"]-CAD9_expanded["T0"]

CAD9_expanded.loc[CAD9_expanded['T3-T0'] < pd.Timedelta(0), 'T3-T0'] = pd.NaT
CAD9_expanded = CAD9_expanded[CAD9_expanded["EventType Trip"] == "P003 - HARTSTILSTAND - DOOD - OVERLEDEN"]
print(len(CAD9_expanded))

#print(CAD9_expanded["T3-T0"])
#interventions_total = interventions_total[""]

#interventions_total2 = pd.concat([interventions_total, CAD9_expanded], axis=0)
#print(interventions_total2)
#interventions_total2 = interventions_total2["Lattitude intervention", "Longitude intervention", "CAD9", "EventLevel Firstcall", "T0-T3"]


interventions4["T0"] = pd.to_datetime(interventions4["T0"], format='%Y-%m-%d %H:%M:%S.%f %z')# Convert the first time format to datetime
interventions4["T3"] = pd.to_datetime(interventions4["T3"], format='%Y-%m-%d %H:%M:%S.%f %z') # Convert the column back to datetime with the new format
#2022-09-06 13:59:12.0373052 +02:00


interventions5["T0"] = pd.to_datetime(interventions5["T0"], format='#%d%b%y:%H:%M:%S')# Convert the first time format to datetime
interventions5["T3"] = pd.to_datetime(interventions5["T3"], format='#%d%b%y:%H:%M:%S') # Convert the column back to datetime with the new format
#01JUN22:00:46:24