import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

# Stap 1: Data laden vanuit een CSV-bestand
interventions1 = pd.read_parquet('DATA/interventions1.parquet.gzip')

# Stap 2: Dash-applicatie initialiseren
app = dash.Dash(__name__)

# Stap 3: Kaart maken met Plotly
map_figure = px.scatter_mapbox(interventions1, lat='Latitude intervention', lon='Longitude intervention',
                               hover_name='Mission ID', zoom=10)
map_figure.update_layout(mapbox_style='open-street-map')

# Stap 4: Dash-layout definiëren
app.layout = html.Div([
    html.H1("Kaart van AED-locaties"),
    dcc.Graph(figure=map_figure)
])

if __name__ == '__main__':
    app.run_server(debug=True)

