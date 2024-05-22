import pandas as pd

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IMPORTING DATASETS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DATA EXPLORATION
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("Number of observations: {}".format(mug.shape[0]))
print("Number of columns:      {}".format(mug.shape[1]))
print(mug.info())
    # 10 variables : hospital_id, mug_id, campus_id, name_hospital, name_campus, adress_campus, postal_code, municipality, region
    # and province

print("Number of observations: {}".format(aed.shape[0]))
print("Number of columns:      {}".format(aed.shape[1]))
print(aed.info())
    # 11 variables : id, type, address, number, postal_code, municipality, province, location, public, available and hours

print("Number of observations: {}".format(ambulance.shape[0]))
print("Number of columns:      {}".format(ambulance.shape[1]))
print(ambulance.info())
    # 9 variables : base, medical_resource, province, region, departure_location, departion_location_number, latitude, longitude
    # and occasional permanence

print("Number of observations: {}".format(pit.shape[0]))
print("Number of columns:      {}".format(pit.shape[1]))
print(pit.info())
    # 8 variables : unit, campus, province, region, unit_id, ambucode, ambusitecode and link

print("Number of observations: {}".format(cad.shape[0]))
print("Number of columns:      {}".format(cad.shape[1]))
print(cad.info())
    # 35 variables : province, Mission ID, Service Name, Latitude permanence, Longitude permanence, Permanence short name,
    # Permanence long name, Vector Type, EventType Trip, EventSubType Trip, EventLevel Trip, CityName intervention, Latitude
    # intervention, Longitude intervention, Province intervention, T0, T1, T1confirmed, T2, T3, T4, T5, T6, T7, Name destination
    # hospital, Intervention time (T1Reported), Intervention time (T1Confirmed), Departure time (T1Reported), Departure time
    # (T1Confirmed), UI, ID, MISSION_NR, AMBUCODE and UNIT_ID

print("Number of observations: {}".format(interventions4.shape[0]))
print("Number of columns:      {}".format(interventions4.shape[1]))
print(interventions4.info())
    # 45 variables : mission_id, service_name, postalcode_permanence, cityname_permanence, streetname_permanence,
    # housenumber_permanence, latitude_permanence, longitude_permanence, permanence_short_name, permanence_long_name, vector_type,
    # eventtype_firstcall, eventLevel_firstcall, eventtype_trip, eventlevel_trip, postalcode_intervention, cityname_intervention,
    # latitude_intervention, longitude_intervention, t0, t1, t1confirmed, t2, t3, t4, t5, t6, t7, t9, intervention_time_t1reported,
    # waiting_time, intervention_duration, departure_time_t1reported, unavailable_time, name_destination_hospital
    # postalcode_destination_hospital, cityname_destination_hospital, streetname_destination_hospital, housenumber_destination_hospital,
    # calculated_traveltime_departure_, calculated_distance_departure_to, calculated_traveltime_destinatio, calculated_distance_destination_,
    # number_of_transported_persons and abandon_reason

print("Number of observations: {}".format(interventions5.shape[0]))
print("Number of columns:      {}".format(interventions5.shape[1]))
print(interventions5.info())
    # 36 variables : Mission ID, T0, Cityname Intervention, Longitude intervention, Latitude intervention, description_nl,
    # ic_description_nl, EventType and EventLevel, creationtime, Number of transported persons, Permanence long name NL, Permanence long name FR,
    # Permanence short name NL, Permanence short name FR, Service Name NL, Service Name FR, Cityname Permanence, Streetname Permanence,
    # Housenumber Permanence, Latitude Permanence, Longitude Permanence, Vector type NL, Vector type FR, Name destination hospital         12706 non-null  object
    # Cityname destination hospital, Streetname destination hospital, Housenumber destination hospital, Abandon reason NL                 4294 non-null   object
    # Abandon reason FR, T1, T2, T3, T4, T5, T6 and T7

print("Number of observations: {}".format(interventions1.shape[0]))
print("Number of columns:      {}".format(interventions1.shape[1]))
print(interventions1.info())
    # 46 variables : Mission ID, Service Name, PostalCode permanence, CityName permanence, StreetName permanence, HouseNumber permanence,
    # Latitude permanence, Longitude permanence, Permanence short name, Permanence long name, Vector type, EventType Firstcall,
    # EventLevel Firstcall, EventType Trip, EventLevel Trip, PostalCode intervention, CityName intervention, Latitude intervention,
    # Longitude intervention, Province intervention, T0, T1, T1confirmed, T2, T3, T4, T5, T6, T7, T9, Intervention time (T1Reported),
    # Intervention time (T1Confirmed), Waiting time, Intervention duration, Departure time (T1Reported), Departure time (T1Confirmed),
    # Unavailable time, Name destination hospital, PostalCode destination hospital, CityName destination hospital
    # StreetName destination hospital, HouseNumber destination hospital, Calculated travelTime destinatio, Calculated Distance destination,
    # Number of transported persons and Abandon reason

print("Number of observations: {}".format(interventions2.shape[0]))
print("Number of columns:      {}".format(interventions2.shape[1]))
print(interventions2.info())
    # 46 variables : Mission ID, Service Name, PostalCode permanence, CityName permanence, StreetName permanence, HouseNumber permanence,
    # Latitude permanence, Longitude permanence, Permanence short name, Permanence long name, Vector type, EventType Firstcall,
    # EventLevel Firstcall, EventType Trip, EventLevel Trip, PostalCode intervention, CityName intervention, Latitude intervention,
    # Longitude intervention, Province intervention, T0, T1, T1confirmed, T2, T3, T4, T5, T6, T7, T9, Intervention time (T1Reported),
    # Intervention time (T1Confirmed), Waiting time, Intervention duration, Departure time (T1Reported), Departure time (T1Confirmed),
    # Unavailable time, Name destination hospital, PostalCode destination hospital, CityName destination hospital
    # StreetName destination hospital, HouseNumber destination hospital, Calculated travelTime destinatio, Calculated Distance destination,
    # Number of transported persons and Abandon reason

print("Number of observations: {}".format(interventions3.shape[0]))
print("Number of columns:      {}".format(interventions3.shape[1]))
print(interventions3.info())
    # 46 variables : Mission ID, Service Name, PostalCode permanence, CityName permanence, StreetName permanence, HouseNumber permanence,
    # Latitude permanence, Longitude permanence, Permanence short name, Permanence long name, Vector type, EventType Firstcall,
    # EventLevel Firstcall, EventType Trip, EventLevel Trip, PostalCode intervention, CityName intervention, Latitude intervention,
    # Longitude intervention, Province intervention, T0, T1, T1confirmed, T2, T3, T4, T5, T6, T7, T9, Intervention time (T1Reported),
    # Intervention time (T1Confirmed), Waiting time, Intervention duration, Departure time (T1Reported), Departure time (T1Confirmed),
    # Unavailable time, Name destination hospital, PostalCode destination hospital, CityName destination hospital
    # StreetName destination hospital, HouseNumber destination hospital, Calculated travelTime destinatio, Calculated Distance destination,
    # Number of transported persons and Abandon reason