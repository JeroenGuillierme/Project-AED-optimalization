import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from operator import itemgetter
import joblib

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

aedLoc = pd.read_csv('data/aed_samengevoegd.csv') # dit moet nog weg en vervangen worden door de juiste dataset (zie deel 2)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DESIGN APP
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

provinces_gdf = gpd.read_file('data/georef-belgium-province-millesime.geojson')
provinces_gdf['prov_name_nl'] = provinces_gdf['prov_name_nl'].apply(lambda x: x[0] if isinstance(x, list) else x)
provinces_gdf['prov_name_nl'] = provinces_gdf['prov_name_nl'].str.replace('Provincie ', '')

app = dash.Dash(__name__)

# Initial map figure
initial_map_figure = px.scatter_mapbox(aedLoc, lat='latitude', lon='longitude', zoom=10)
initial_map_figure.update_traces(marker=dict(size=10, color='red'))
initial_map_figure.update_layout(mapbox_style='open-street-map')

# Import ResponseTimeModel
model = joblib.load('ResponseTimeModel.joblib')

# Define Dash layout
app.layout = html.Div([
    html.H1("AED localization app"),
    dcc.Input(id='lat-input', type='number', placeholder='Enter the latitude'),
    dcc.Input(id='lon-input', type='number', placeholder='Enter the longitude'),
    html.Button(id='submit-button', n_clicks=0, children='Search'),
    html.Div([
        dcc.Dropdown(
            id='event-level-dropdown',
            options=[
                {'label': 'N0', 'value': 'N0'},
                {'label': 'N1', 'value': 'N1'},
                {'label': 'N2', 'value': 'N2'},
                {'label': 'N3', 'value': 'N3'},
                {'label': 'N4', 'value': 'N4'},
                {'label': 'N5', 'value': 'N5'},
                {'label': 'N6', 'value': 'N6'},
                {'label': 'N7', 'value': 'N7'},
                {'label': 'N8', 'value': 'N8'},
                {'label': 'Other', 'value': 'Other'}
            ],
            placeholder='Select Event Level'
        )
    ]),
    html.Div([
        dcc.Dropdown(
            id='vector-dropdown',
            options=[
                {'label': 'Ambulance', 'value': 'Ambulance'},
                {'label': 'Brandziekenwagen', 'value': 'Brandziekenwagen'},
                {'label': 'Decontaminatiewagen', 'value': 'Decontaminatiewagen'},
                {'label': 'Mug', 'value': 'Mug'},
                {'label': 'Pit', 'value': 'Pit'},
            ],
            placeholder='Select Vector'
        )
    ]),
    html.Div(id='popup-message'),
    dcc.Graph(id='map', figure=initial_map_figure),
    html.Div(id='error-message', style={'color': 'red'})
])

def give_predicted_response_time(vectorMetLengteZes):
    predicted_total_seconds = model.predict([vectorMetLengteZes])[0]
    minutes = int(predicted_total_seconds // 60)
    seconds = int(predicted_total_seconds % 60)
    return [minutes, seconds]


def event_level_to_matrix(event_level):
    # Define a list of possible event levels in the order they should appear in the matrix
    event_level_list = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'Other']

    # Initialize a matrix with zeros
    matrix = [0] * len(event_level_list)

    # Find the index of the selected event level and set the corresponding position to 1
    if event_level in event_level_list:
        index = event_level_list.index(event_level)
        matrix[index] = 1

    return matrix


def vector_to_matrix(vector):
    # Define a list of possible vectors in the order they should appear in the matrix
    vector_list = ['Ambulance', 'Brandziekenwagen', 'Decontaminatiewagen', 'Mug', 'Pit']

    # Initialize a matrix with zeros
    matrix = [0] * len(vector_list)

    # Find the index of the selected vector and set the corresponding position to 1
    if vector in vector_list:
        index = vector_list.index(vector)
        matrix[index] = 1

    return matrix


def province_to_matrix(province):
    # Define a list of provinces
    province_list = ['Antwerpen', 'Brussel', 'Henegouwen', 'Limburg', 'Luik', 'Luxemburg', 'Namen',
                     'Oost-Vlaanderen', 'Vlaams-Brabant', 'Waals-Brabant', 'West-Vlaanderen']

    # Initialize a matrix with zeros
    matrix = [0] * len(province_list)

    # Find the index of the province and set the corresponding position to 1
    if province in province_list:
        index = province_list.index(province)
        matrix[index] = 1

    return matrix

def get_province_from_coordinates(lat, lon):
    point = Point(lon, lat)
    for _, province in provinces_gdf.iterrows():
        if province['geometry'].contains(point):
            return province['prov_name_nl']
    return 'Brussel'

# Callback to update entered coordinates in the map
@app.callback(
    [Output('map', 'figure'), Output('popup-message', 'children'), Output('error-message', 'children')],
    [Input('submit-button', 'n_clicks'), Input('lat-input', 'value'), Input('lon-input', 'value'),
     Input('event-level-dropdown', 'value'), Input('vector-dropdown', 'value')]
)
def update_map(n_clicks, lat, lon, event_level, vector):
    map_figure = initial_map_figure
    popup_message = ''
    error_message = ''

    if n_clicks > 0:
        if lat is None or lon is None:
            error_message = 'Please provide latitude and longitude.'
        else:
            try:
                user_location = pd.DataFrame({
                    'latitude': [lat],
                    'longitude': [lon]
                })

                map_figure = px.scatter_mapbox(aedLoc, lat='latitude', lon='longitude', zoom=10)
                map_figure.update_traces(marker=dict(size=10, color='red'))

                user_trace = px.scatter_mapbox(user_location, lat='latitude', lon='longitude')
                user_trace.update_traces(marker=dict(size=12, color='blue'))

                for trace in user_trace.data:
                    map_figure.add_trace(trace)

                map_figure.update_layout(
                    mapbox_style='open-street-map',
                    mapbox=dict(center=dict(lat=lat, lon=lon), zoom=14)
                )

                province = get_province_from_coordinates(lat, lon)
                # Create the event level and vector matrices
                event_level_matrix = event_level_to_matrix(event_level)
                vector_matrix = vector_to_matrix(vector)
                province_matrix = province_to_matrix(province)

                # Combine the matrices into a single matrix
                combined_matrix = province_matrix + vector_matrix + event_level_matrix

                indices = [0, 8, 11, 14, 20, 21]
                respons_matrix = [combined_matrix[i] if i < len(combined_matrix) else 0 for i in indices]

                # Pop-up message with event level, vector, and combined matrix
                popup_message = (f'Je hebt gezocht op locatie: Breedtegraad {lat}, '
                                 f'Lengtegraad {lon}, Event Level: {event_level}, '
                                 f'Vector: {vector}, Province: {province}, Combined Matrix: {respons_matrix}',
                                  "Voorspelde responstijd: ",give_predicted_response_time(respons_matrix)[0], " minuten en ",give_predicted_response_time(respons_matrix)[1], " seconden")
            except Exception as e:
                error_message = f'Error while processing coordinates: {e}'

    return map_figure, popup_message, error_message


if __name__ == '__main__':
    app.run_server(debug=True)


# Nog iets dan met het type AED en available van de AED? Je kan hover_name bij updaten pas invoeren of al bij het opstellen van je kaart (moet bij map_figure).
# Of is het beter om type en available weer te geven in een pop up? (Volgens mij makkelijker bij hover name dan een pop-up).