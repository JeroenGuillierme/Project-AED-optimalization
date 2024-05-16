import pandas as pd
from datetime import datetime
import numpy as np

# read data
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

# append interventions dataset (1-3)
interventions1["T0"] = pd.to_datetime(interventions1["T0"],
                                      format='%d%b%y:%H:%M:%S')  # Convert the first time format to datetime
interventions1["T3"] = pd.to_datetime(interventions1["T3"],
                                      format='%Y-%m-%d %H:%M:%S.%f')  # Convert the column back to datetime with the new format

interventions2["T0"] = pd.to_datetime(interventions2["T0"],
                                      format='%d%b%y:%H:%M:%S')  # Convert the first time format to datetime
interventions2["T3"] = pd.to_datetime(interventions2["T3"],
                                      format='%Y-%m-%d %H:%M:%S.%f')  # Convert the column back to datetime with the new format

interventions3["T0"] = pd.to_datetime(interventions3["T0"],
                                      format='%d%b%y:%H:%M:%S')  # Convert the first time format to datetime
interventions3["T3"] = pd.to_datetime(interventions3["T3"],
                                      format='%Y-%m-%d %H:%M:%S.%f')  # Convert the column back to datetime with the new format

interventions_total = pd.concat([interventions1, interventions2, interventions3], axis=0)
print(list(interventions_total.columns))

interventions_total['CAD9'] = 'N'
interventions_total["T3-T0"] = interventions_total["T3"] - interventions_total["T0"]
interventions_total = interventions_total[interventions_total["EventType Firstcall"] == "P003 - Cardiac arrest"]
interventions_total['Eventlevel'] = interventions_total["EventLevel Firstcall"]
cross_tab = pd.crosstab(index=pd.Categorical(interventions_total["Eventlevel"]), columns='count')
print(cross_tab)
interventions_total = interventions_total[
    ["Latitude intervention", "Longitude intervention", "CAD9", "Eventlevel", "T3-T0"]]
print(list(interventions_total.columns))

# Append CAD9 dataset

CAD9_expanded = cad
# add extra column CAD9
CAD9_expanded['CAD9'] = 'Y'
CAD9_expanded["T0"] = pd.to_datetime(CAD9_expanded["T0"],
                                     format='%Y-%m-%d %H:%M:%S.%f')  # Convert the column back to datetime with the new format
CAD9_expanded["T3"] = pd.to_datetime(CAD9_expanded["T3"],
                                     format='%Y-%m-%d %H:%M:%S.%f')  # Convert the column back to datetime with the new format
CAD9_expanded["T3-T0"] = CAD9_expanded["T3"] - CAD9_expanded["T0"]
CAD9_expanded.loc[CAD9_expanded['T3-T0'] < pd.Timedelta(0), 'T3-T0'] = pd.NaT
CAD9_expanded = CAD9_expanded[CAD9_expanded["EventType Trip"] == "P003 - HARTSTILSTAND - DOOD - OVERLEDEN"]
CAD9_expanded['Eventlevel'] = CAD9_expanded['EventLevel Trip']

CAD9_expanded = CAD9_expanded[['Latitude intervention', 'Longitude intervention', 'CAD9', 'Eventlevel', 'T3-T0']]

# Append datasets interventions Brussels
interventions4["t0"] = pd.to_datetime(interventions4["t0"],
                                      format='%Y-%m-%d %H:%M:%S.%f %z')  # Convert the first time format to datetime
interventions4["t3"] = pd.to_datetime(interventions4["t3"],
                                      format='%Y-%m-%d %H:%M:%S.%f %z')  # Convert the column back to datetime with the new format

interventions4['CAD9'] = 'N'
interventions4 = interventions4[interventions4['eventtype_firstcall'] == 'P003 - Cardiac arrest']
interventions4['Eventlevel'] = interventions4['eventLevel_firstcall']
interventions4['T3-T0'] = interventions4['t3'] - interventions4['t0']
interventions4['Latitude intervention'] = interventions4['latitude_intervention']
interventions4['Longitude intervention'] = interventions4['longitude_intervention']
interventions4 = interventions4[['Latitude intervention', 'Longitude intervention', 'CAD9', 'Eventlevel', 'T3-T0']]


interventions5["T0"] = pd.to_datetime(interventions5["T0"],
                                      format='%d%b%y:%H:%M:%S')  # Convert the first time format to datetime
interventions5["T3"] = pd.to_datetime(interventions5["T3"],
                                      format='%d%b%y:%H:%M:%S')  # Convert the column back to datetime with the new format
interventions5['CAD9'] = 'N'
interventions5 = interventions5[
    (interventions5['EventType and EventLevel'] == 'P003  N01 - HARTSTILSTAND - DOOD - OVERLEDEN') | (
                interventions5['EventType and EventLevel'] == 'P003  N05 - HARTSTILSTAND - DOOD - OVERLEDEN')]
interventions5["Eventlevel"] = interventions5["EventType and EventLevel"].str.split(" ").str[1].str.replace("0", "")
interventions5['T3-T0'] = interventions5['T3'] - interventions5['T0']
interventions5 = interventions5[['Latitude intervention', 'Longitude intervention', 'CAD9', 'Eventlevel', 'T3-T0']]

interventions_TOTAL = pd.concat([interventions_total, CAD9_expanded, interventions4, interventions5], axis=0)
print(interventions_TOTAL)
print(interventions_TOTAL['Eventlevel'].unique())

print(interventions_TOTAL['CAD9'].unique())
cross_tab = pd.crosstab(index=pd.Categorical(interventions_TOTAL["Eventlevel"]), columns='count')
print(cross_tab)
print(list(interventions_TOTAL.columns))
