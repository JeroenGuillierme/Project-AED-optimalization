import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
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

aedLoc = pd.read_csv('data/aed_samengevoegd.csv') # dit moet nog weg en vervangen worden door de juiste dataset (zie deel 2)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DESIGN APP
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

app = dash.Dash(__name__)

# Map making with plotly
initial_map_figure = px.scatter_mapbox(aedLoc, lat='latitude', lon='longitude', zoom=10)
initial_map_figure.update_traces(marker=dict(size=10, color='red'))
initial_map_figure.update_layout(mapbox_style='open-street-map')

# Defining Dash-layout
app.layout = html.Div([
    html.H1("AED localization app"),
    dcc.Input(id='lat-input', type='number', placeholder='Enter the latitude'),
    dcc.Input(id='lon-input', type='number', placeholder='Enter the longitude'),
    html.Button(id='submit-button', n_clicks=0, children='Search'),
    html.Div(id='popup-message'),
    dcc.Graph(id='map', figure=initial_map_figure),
    html.Div(id='error-message', style={'color': 'red'})
])


# Callback to update entered coordinates in the map
@app.callback(
    [Output('map', 'figure'), Output('popup-message', 'children'), Output('error-message', 'children')],
    [Input('submit-button', 'n_clicks'), Input('lat-input', 'value'), Input('lon-input', 'value')]
)
def update_map(n_clicks, lat, lon):
    map_figure = initial_map_figure
    popup_message = ''
    error_message = ''

    if n_clicks > 0 and lat is not None and lon is not None:
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

            map_figure.update_layout(mapbox_style='open-street-map', mapbox=dict(center=dict(lat=lat, lon=lon),
                                                 zoom=14))

            # Pop-up message
            popup_message = f'Je hebt gezocht op locatie: Breedtegraad {lat}, Lengtegraad {lon}'
        except Exception as e:
            error_message = f'Error while processing coordinates: {e}'

    return map_figure, popup_message, error_message


if __name__ == '__main__':
    app.run_server(debug=True)

# Nog iets dan met het type AED en available van de AED? Je kan hover_name bij updaten pas invoeren of al bij het opstellen van je kaart (moet bij map_figure).
# Of is het beter om type en available weer te geven in een pop up? (Volgens mij makkelijker bij hover name dan een pop-up).