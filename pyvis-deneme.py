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
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
from pyvis.network import Network
import base64
import textwrap

# Sample data for demonstration purposes
data = {
    'Main': pd.DataFrame(np.random.randint(0, 10, size=(10, 10)), columns=[f'Code{i}' for i in range(10)], index=[f'Code{i}' for i in range(10)]),
    'Condition': pd.DataFrame(np.random.randint(0, 10, size=(10, 10)), columns=[f'Code{i}' for i in range(10)], index=[f'Code{i}' for i in range(10)]),
    'Observation': pd.DataFrame(np.random.randint(0, 10, size=(10, 10)), columns=[f'Code{i}' for i in range(10)], index=[f'Code{i}' for i in range(10)]),
}

SUBGROUP_COLORS = {
    'Condition': "#00bfff",
    'Observation': "#ffc0cb",
}

app = dash.Dash(__name__)
server = app.server  # Required for deployment with Gunicorn

# Initialize layout with default components
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(
            id='code-dropdown',
            options=[{'label': f'Code{i}', 'value': f'Code{i}'} for i in range(10)],
            value='Code0',  # Default value
        ),
        dcc.Slider(
            id='node-slider',
            min=1,
            max=10,
            value=5,  # Default value
            marks={i: str(i) for i in range(1, 11)},
        ),
        dcc.Checklist(
            id='show-labels',
            options=[{'label': 'Show Labels', 'value': 'show_labels'}],
            value=['show_labels'],  # Default checked
        ),
        html.Button('Generate Graph', id='generate-button'),
    ], style={'marginBottom': '20px'}),
    html.Div(id='graph-container'),
    dcc.Loading(id="loading", type="default", children=[
        html.Div(id='loading-output'),
    ]),
])

@app.callback(
    [Output('graph-container', 'children'),
     Output('loading-output', 'children')],
    [Input('generate-button', 'n_clicks')],
    [State('code-dropdown', 'value'),
     State('node-slider', 'value'),
     State('show-labels', 'value')],
)
def update_graph(n_clicks, selected_code, num_nodes_to_visualize, show_labels):
    if not n_clicks:
        return "", ""

    # Initialize a PyVis network
    net = Network(notebook=False, cdn_resources='remote')

    # Example: Generating a graph
    try:
        # Get the main co-occurrence matrix
        main_df = data['Main']

        # Sort neighbors by descending order
        neighbors_sorted = main_df.loc[selected_code].sort_values(ascending=False)
        top_neighbors = list(neighbors_sorted.index[:15])

        # Function to add nodes and edges to the PyVis network
        def add_nodes_edges(child_df, prefix, group_name):
            top_neighbor = None
            for neighbor_code in neighbors_sorted.index:
                if neighbor_code.startswith(prefix):
                    top_neighbor = neighbor_code
                    break

            if top_neighbor:
                selected_code_label = wrap_text(selected_code)
                top_neighbor_label = wrap_text(top_neighbor)

                net.add_node(selected_code, title=selected_code, label=selected_code_label if 'show_labels' in show_labels else selected_code[2:], color=SUBGROUP_COLORS.get(group_name, 'gray'))
                net.add_node(top_neighbor, title=top_neighbor, label=top_neighbor_label if 'show_labels' in show_labels else top_neighbor[2:], color=SUBGROUP_COLORS.get(group_name, 'gray'))
                net.add_edge(selected_code, top_neighbor, value=int(main_df.loc[selected_code, top_neighbor]))

                top_neighbor_row = child_df.loc[top_neighbor].sort_values(ascending=False)
                top_neighbors = list(top_neighbor_row.index[:num_nodes_to_visualize])

                for neighbor in top_neighbors:
                    if neighbor != top_neighbor and child_df.loc[top_neighbor, neighbor] > 0:
                        neighbor_label = wrap_text(neighbor)
                        net.add_node(neighbor, title=neighbor, label=neighbor_label if 'show_labels' in show_labels else neighbor[2:], color=SUBGROUP_COLORS.get(group_name, 'gray'))
                        net.add_edge(top_neighbor, neighbor, value=int(child_df.loc[top_neighbor, neighbor]))

                        for other_code in top_neighbor_row.index[:num_nodes_to_visualize]:
                            if other_code != neighbor and child_df.loc[neighbor, other_code] > 0:
                                other_code_label = wrap_text(other_code)
                                net.add_node(other_code, title=other_code, label=other_code_label if 'show_labels' in show_labels else other_code[2:], color=SUBGROUP_COLORS.get(group_name, 'gray'))
                                net.add_edge(neighbor, other_code, value=int(child_df.loc[neighbor, other_code]))

        if 'Condition' in data:
            add_nodes_edges(data['Condition'], 'Co', 'Condition')
        if 'Observation' in data:
            add_nodes_edges(data['Observation'], 'Ob', 'Observation')

        # Generate HTML content directly
        graph_html = net.generate_html(notebook=False)

        # Embed the HTML content directly into an iframe using a data URL
        graph_base64 = base64.b64encode(graph_html.encode()).decode()
        graph_layout = html.Div([
            html.Iframe(
                src=f"data:text/html;base64,{graph_base64}",
                style={'width': '100%', 'height': '600px'}
            )
        ])

        return graph_layout, "Graph generated successfully!"

    except Exception as e:
        return "", f"Error generating graph: {e}"

def wrap_text(text, max_width=15):
    """Wrap text to fit within a maximum width, defaulting to 15 characters."""
    return "\n".join(textwrap.wrap(text, width=max_width))

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)


# In[ ]:




