import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import joblib
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IMPORTING DATASETS
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

OldAEDs = pd.read_csv('data/aed_placement_df.csv')
NewAEDs = pd.read_csv('data/new_aed_locations.csv')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PRE-PROCESSING
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import ResponseTimeModel (see part I)
model = joblib.load('Response Time Analysis/ResponseTimeModel.joblib')

# Making dataset with new AEDs (see part II)
Old = OldAEDs[OldAEDs['AED'] == 1][['Longitude', 'Latitude']]
AllAEDs = pd.concat([Old, NewAEDs[['Longitude','Latitude']]], ignore_index=True)
AllAEDs['hover'] = "AED"

# Needed for determination provinces
provinces_gdf = gpd.read_file('data/georef-belgium-province-millesime.geojson')
provinces_gdf['prov_name_nl'].head()
provinces_gdf['prov_name_nl'] = provinces_gdf['prov_name_nl'].apply(lambda x: x[0] if isinstance(x, list) else x)
provinces_gdf['prov_name_nl'].unique()
provinces_gdf['prov_name_nl'] = provinces_gdf['prov_name_nl'].str.replace('Provincie ', '')

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DESIGN APP
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

app = dash.Dash(__name__, title='AED localization app', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initial map figure
initial_map_figure = px.scatter_mapbox(AllAEDs, lat='Latitude', lon='Longitude', zoom=10, hover_name="hover")
initial_map_figure.update_traces(marker=dict(size=10, color='red'))
initial_map_figure.update_layout(mapbox_style='open-street-map')

# Define Dash layout
app.layout = html.Div(style={'backgroundColor': 'green'}, children=[
    html.H1("AED localization app", style={'color': 'white'}),
    dcc.Input(id='lat-input', type='number', placeholder='Enter the latitude'),
    dcc.Input(id='lon-input', type='number', placeholder='Enter the longitude'),
    html.Button(id='submit-button', n_clicks=0, children='Search'),
    html.Button(id='reset-button', n_clicks=0, children='Reset'),
    html.Div([dcc.Dropdown(id='event-level-dropdown', options=[
                {'label': 'N0', 'value': 'N0'},
                {'label': 'N1', 'value': 'N1'},
                {'label': 'N2', 'value': 'N2'},
                {'label': 'N3', 'value': 'N3'},
                {'label': 'N4', 'value': 'N4'},
                {'label': 'N5', 'value': 'N5'},
                {'label': 'N6', 'value': 'N6'},
                {'label': 'N7', 'value': 'N7'},
                {'label': 'N8', 'value': 'N8'},
                {'label': 'Other', 'value': 'Other'}], placeholder='Select Event Level')]),
    html.Div([dcc.Dropdown(id='vector-dropdown', options=[
                {'label': 'Ambulance', 'value': 'Ambulance'},
                {'label': 'Fire ambulance', 'value': 'Fire ambulance'},
                {'label': 'Decontamination vehicle', 'value': 'Decontamination vehicle'},
                {'label': 'Mug', 'value': 'Mug'},
                {'label': 'Pit', 'value': 'Pit'}], placeholder='Select Vector')]),
    html.Div(id='popup-message', style={'color': 'white'}),
    html.Button(id='instructions-button', n_clicks=0, children='Instructions'),
    html.Div(id='instructions-message', style={'color': 'white'}),
    dcc.Graph(id='map', figure=initial_map_figure),
    html.Div(id='error-message', style={'color': 'white'})
])

# Functions needed in app to determine the response time
def give_predicted_response_time(vectorMetLengteZes):
    predicted_total_seconds = model.predict([vectorMetLengteZes])[0]
    minutes = int(predicted_total_seconds // 60)
    seconds = int(predicted_total_seconds % 60)
    return [minutes, seconds]

def event_level_to_matrix(event_level):
    event_level_list = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'Other']
    matrix = [0] * len(event_level_list)
    # Find the index of the selected event level and set the corresponding position to 1
    if event_level in event_level_list:
        index = event_level_list.index(event_level)
        matrix[index] = 1
    return matrix

def vector_to_matrix(vector):
    vector_list = ['Ambulance', 'Fire ambulance', 'Decontamination vehicle', 'Mug', 'Pit']
    matrix = [0] * len(vector_list)
    if vector in vector_list:
        index = vector_list.index(vector)
        matrix[index] = 1
    return matrix

def province_to_matrix(province):
    province_list = ['Antwerpen', 'Brussel', 'Henegouwen', 'Limburg', 'Luik', 'Luxemburg', 'Namen',
                     'Oost-Vlaanderen', 'Vlaams-Brabant', 'Waals-Brabant', 'West-Vlaanderen']
    matrix = [0] * len(province_list)
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

# Depending on the location entered, adjust the map and display the response time
@app.callback(
    [Output('lat-input', 'value'), Output('lon-input', 'value'), Output('map', 'figure'), Output('popup-message', 'children'), Output('error-message', 'children')],
    [Input('submit-button', 'n_clicks'), Input('reset-button', 'n_clicks')],
    [State('lat-input', 'value'), State('lon-input', 'value'),
     State('event-level-dropdown', 'value'), State('vector-dropdown', 'value')])

def update_or_reset_map(submit_clicks, reset_clicks, lat, lon, event_level, vector):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'reset-button':
        return None, None, initial_map_figure, '', ''

    map_figure = initial_map_figure
    popup_message = ''
    error_message = ''

    if submit_clicks > 0:
        if lat is None or lon is None:
            error_message = 'Please provide latitude and longitude.'
        else:
            try:
                user_location = pd.DataFrame({
                    'latitude': [lat],
                    'longitude': [lon],
                    'hover': 'Your location'
                })

                map_figure = px.scatter_mapbox(AllAEDs, lat='Latitude', lon='Longitude', zoom=10, hover_name='hover')
                map_figure.update_traces(marker=dict(size=10, color='red'))

                user_trace = px.scatter_mapbox(user_location, lat='latitude', lon='longitude', hover_name='hover')
                user_trace.update_traces(marker=dict(size=12, color='blue'))

                for trace in user_trace.data:
                    map_figure.add_trace(trace)

                map_figure.update_layout(mapbox_style='open-street-map',mapbox=dict(center=dict(lat=lat, lon=lon), zoom=14))

                province = get_province_from_coordinates(lat, lon)
                event_level_matrix = event_level_to_matrix(event_level)
                vector_matrix = vector_to_matrix(vector)
                province_matrix = province_to_matrix(province)
                combined_matrix = province_matrix + vector_matrix + event_level_matrix

                indices = [0, 8, 11, 14, 20, 21]
                respons_matrix = [combined_matrix[i] if i < len(combined_matrix) else 0 for i in indices]

                popup_message = f'Predicted response time in {province} for event level {event_level} and vector {vector} is {give_predicted_response_time(respons_matrix)[0]} minutes and {give_predicted_response_time(respons_matrix)[1]} seconds.'
            except Exception as e:
                error_message = f'Error while processing coordinates: {e}'

    return lat, lon, map_figure, popup_message, error_message

# Giving instructions
@app.callback(
    Output('instructions-message', 'children'),
    [Input('instructions-button', 'n_clicks')],
    [State('instructions-message', 'children')])

def toggle_instructions(n_clicks, current_message):
    if n_clicks > 0:
        if current_message:
            return ''
        return html.P([
            " 1. Check for consciousness and breathing.", html.Br(),
            " 2. Call 112 emergency services immediately.", html.Br(),
            " 3. Start CPR:", html.Br(),
            "   - Lay the person on a firm, flat surface.", html.Br(),
            "   - Place your hands in the center of the chest, one hand on top of the other.", html.Br(),
            "   - Press hard and fast (at least 100-120 compressions per minute) with a depth of about 2-2.5 inches (5-6 cm).", html.Br(),
            "   - Allow the chest to fully recoil after each compression.", html.Br(),
            "   - If trained, give 30 chest compressions followed by 2 rescue breaths. If untrained, continue with chest compressions only.", html.Br(),
            " 4. Use an AED if available:", html.Br(),
            "   - Turn on the AED and follow the voice prompts.", html.Br(),
            "   - Attach the pads as indicated on the AED.", html.Br(),
            "   - Ensure no one is touching the person while the AED analyzes the heart rhythm.", html.Br(),
            "   - Follow the AED’s instructions to deliver a shock if advised.", html.Br(),
            " 5. Continue CPR:", html.Br(),
            "   - Keep performing chest compressions and rescue breaths (if applicable) until the person shows signs of life, such as normal breathing, movement, or regaining consciousness.", html.Br(),
            "   - Continue until professional help arrives and takes over.", html.Br(),
            " 6. Provide information to emergency responders:", html.Br(),
            "   - Inform them of everything you have done and how the person has responded."
        ])
    return current_message

if __name__ == '__main__':
    app.run_server(debug=True)

'''
Due to a less-than-ideal experience with AWS, we opted to work with coordinates. However, it is evident 
that using addresses would be more convenient for the app. Below are the modifications to the code that 
would be required if we were to use addresses instead of coordinates.

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Aws region in which the Amazon locator service is configured, it probably would have been better to replace it with eu-central-1
region_name = 'us-east-1'

# Configuring AWS credentials after creating an IAM user
session = boto3.Session(
    aws_access_key_id='AKIA6GBMAUQCXW4TPTUE',
    aws_secret_access_key='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',
    region_name=region_name)

client = boto3.client('location')

def get_coordinates(address):
    try:
        response = client.search_place_index_for_text(
            IndexName='explore.place.Esri',
            Text=address)
            MaxResults=1
        )
        if response['Results']:
            location = response['Results'][0]['Place']['Geometry']['Point']
            return location[1], location[0]  # Return lat, lon
        else:
            return None, None
    except (BotoCoreError, ClientError) as e:
        print(f"Error occurred: {e}")
        return None, None

# Define Dash layout
app.layout = html.Div(style={'backgroundColor': 'green'}, children=[
    html.H1("AED localization app", style={'color': 'white'}),
    dcc.Input(id='address-input', type='text', placeholder='Enter the address'),
    html.Button(id='submit-button', n_clicks=0, children='Search'),
    html.Button(id='reset-button', n_clicks=0, children='Reset'),
    html.Div([dcc.Dropdown(id='event-level-dropdown', options=[
                {'label': 'N0', 'value': 'N0'},
                {'label': 'N1', 'value': 'N1'},
                {'label': 'N2', 'value': 'N2'},
                {'label': 'N3', 'value': 'N3'},
                {'label': 'N4', 'value': 'N4'},
                {'label': 'N5', 'value': 'N5'},
                {'label': 'N6', 'value': 'N6'},
                {'label': 'N7', 'value': 'N7'},
                {'label': 'N8', 'value': 'N8'},
                {'label': 'Other', 'value': 'Other'}], placeholder='Select Event Level')]),
    html.Div([dcc.Dropdown(id='vector-dropdown', options=[
                {'label': 'Ambulance', 'value': 'Ambulance'},
                {'label': 'Fire ambulance', 'value': 'Fire ambulance'},
                {'label': 'Decontamination vehicle', 'value': 'Decontamination vehicle'},
                {'label': 'Mug', 'value': 'Mug'},
                {'label': 'Pit', 'value': 'Pit'}], placeholder='Select Vector')]),
    html.Div(id='popup-message'),
    html.Button(id='instructions-button', n_clicks=0, children='Instructions'),
    html.Div(id='instructions-message'),
    dcc.Graph(id='map', figure=initial_map_figure),
    html.Div(id='error-message', style={'color': 'white'})
])

# Functions needed in app to determine the response time
def give_predicted_response_time(vectorMetLengteZes):
    predicted_total_seconds = model.predict([vectorMetLengteZes])[0]
    minutes = int(predicted_total_seconds // 60)
    seconds = int(predicted_total_seconds % 60)
    return [minutes, seconds]

def event_level_to_matrix(event_level):
    event_level_list = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'Other']
    matrix = [0] * len(event_level_list)
    # Find the index of the selected event level and set the corresponding position to 1
    if event_level in event_level_list:
        index = event_level_list.index(event_level)
        matrix[index] = 1
    return matrix

def vector_to_matrix(vector):
    vector_list = ['Ambulance', 'Fire ambulance', 'Decontamination vehicle', 'Mug', 'Pit']
    matrix = [0] * len(vector_list)
    if vector in vector_list:
        index = vector_list.index(vector)
        matrix[index] = 1
    return matrix

def province_to_matrix(province):
    province_list = ['Antwerpen', 'Brussel', 'Henegouwen', 'Limburg', 'Luik', 'Luxemburg', 'Namen',
                     'Oost-Vlaanderen', 'Vlaams-Brabant', 'Waals-Brabant', 'West-Vlaanderen']
    matrix = [0] * len(province_list)
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

# Depending on the location entered, adjust the map and display the response time
@app.callback(
    [Output('address-input', 'value'), Output('map', 'figure'), Output('popup-message', 'children'), Output('error-message', 'children')],
    [Input('submit-button', 'n_clicks'), Input('reset-button', 'n_clicks')],
    [State('address-input', 'value'),
     State('event-level-dropdown', 'value'), State('vector-dropdown', 'value')])

def update_or_reset_map(submit_clicks, reset_clicks, address, event_level, vector):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'reset-button':
        return None, initial_map_figure, '', ''

    map_figure = initial_map_figure
    popup_message = ''
    error_message = ''

    if submit_clicks > 0:
        if not address:
            error_message = 'Please provide an address.'
        else:
            try:
                lat, lon = get_coordinates_from_address(address)
                if lat is None or lon is None:
                    error_message = 'Could not find the address.'
                else:
                    user_location = pd.DataFrame({
                        'latitude': [lat],
                        'longitude': [lon],
                        'hover': "Your location"
                    })

                    map_figure = px.scatter_mapbox(AllAEDs, lat='Latitude', lon='Longitude', zoom=10, hover_name="hover")
                    map_figure.update_traces(marker=dict(size=10, color='red'))

                    user_trace = px.scatter_mapbox(user_location, lat='latitude', lon='longitude', hover_name='hover')
                    user_trace.update_traces(marker=dict(size=12, color='blue'))

                    for trace in user_trace.data:
                        map_figure.add_trace(trace)

                    map_figure.update_layout(mapbox_style='open-street-map',mapbox=dict(center=dict(lat=lat, lon=lon), zoom=14))

                    province = get_province_from_coordinates(lat, lon)
                    event_level_matrix = event_level_to_matrix(event_level)
                    vector_matrix = vector_to_matrix(vector)
                    province_matrix = province_to_matrix(province)
                    combined_matrix = province_matrix + vector_matrix + event_level_matrix

                    indices = [0, 8, 11, 14, 20, 21]
                    respons_matrix = [combined_matrix[i] if i < len(combined_matrix) else 0 for i in indices]

                    popup_message = f'Predicted response time for in {province} for event level {event_level} and vector {vector} is {give_predicted_response_time(respons_matrix)[0]} minutes and {give_predicted_response_time(respons_matrix)[1]} seconds.'
            except Exception as e:
                error_message = f'Error while processing address: {e}'

    return address, map_figure, popup_message, error_message
'''

'''
For the same reason (avoiding even more costs), the deployment of the app was not done because it was fee-based.
'''