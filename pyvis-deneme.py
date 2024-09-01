#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import dash
# from dash import html
# from dash.dependencies import Input, Output
# import pyvis
# from pyvis.network import Network

# # Create Pyvis Network graph
# net = Network(notebook=True)
# net.add_node('Pelin')
# net.add_node('Volkan')
# net.add_edge('Pelin', 'Volkan')
# net_html = net.show("graph.html")

# # Create Dash app
# app = dash.Dash(__name__)
# server = app.server

# app.layout = html.Div([
#     html.Iframe(srcDoc=open('graph.html', 'r').read(), style={'width': '100%', 'height': '600px'})
# ])

# if __name__ == '__main__':
#     app.run_server(debug=True)

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
from pyvis.network import Network
import time

# Create Dash app
app = dash.Dash(__name__)
server = app.server  # Required for deployment with Gunicorn

# Global variable to store loaded data
loaded_data = None

# Initial layout with the "Load Data" button and placeholders for progress and graph
app.layout = html.Div([
    html.Button('Load Data', id='load-button'),
    dcc.Loading(id="loading", type="default", children=[
        html.Div(id='loading-output'),
    ]),
    html.Div(id='graph-container')
])

# Callback to handle loading data and creating graph
@app.callback(
    Output('loading-output', 'children'),
    Output('graph-container', 'children'),
    Input('load-button', 'n_clicks'),
    prevent_initial_call=True  # This prevents the callback from firing when the app first loads
)
def load_data_and_create_graph(n_clicks):
    global loaded_data

    if n_clicks is None:
        return "", ""

    # Simulate loading process with a progress message
    loading_status = "Loading data... 0%"
    time.sleep(1)
    loading_status = "Loading data... 50%"

    # Load data (using Iris dataset as an example)
    try:
        loaded_data = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
            header=None,
            names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        )
        time.sleep(1)
        loading_status = "Loading data... 100%"
    except Exception as e:
        return f"Failed to load data: {e}", ""

    loading_status = "Loading complete!"

    # Create Pyvis Network graph using the loaded data
    net = Network(notebook=True)
    species_set = loaded_data['species'].unique()
    for species in species_set:
        net.add_node(species, label=species)

    # Example: Add edges between species (just for demonstration)
    if len(species_set) > 1:
        net.add_edge(species_set[0], species_set[1])
    if len(species_set) > 2:
        net.add_edge(species_set[1], species_set[2])

    net.show("graph.html")  # Save graph to an HTML file

    # Display the graph in an iframe
    graph_layout = html.Div([
        dcc.Dropdown(
            id='combobox',
            options=[{'label': species, 'value': species} for species in species_set],
            value=species_set[0]  # Default to the first species
        ),
        html.Iframe(srcDoc=open('graph.html', 'r').read(), style={'width': '100%', 'height': '600px'})
    ])

    return loading_status, graph_layout

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)  # Use port 8051 instead of 8050


# In[ ]:




