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
################################################################## DELL-SYNTHETIC ##############################################################

# import os
# import re
# import pandas as pd
# import numpy as np
# import dash
# from dash import html, dcc
# from dash.dependencies import Input, Output
# from pyvis.network import Network
# import tempfile
# import networkx as nx
# import requests
# import zipfile

# # URL of the ZIP file you want to download (replace with your own URL)
# dropbox_url = 'https://www.dropbox.com/scl/fo/y0i3bwhd6rnijujmm6uyx/AHJnzBz20d9hsi8XYFBjd64?rlkey=dxm0mky9bdq43xbov9x7ir39u&st=wrsjk7bw&dl=0'
# # Change 'dl=0' to 'dl=1' to directly download the file
# download_url = dropbox_url.replace('dl=0', 'dl=1')

# # Local directory and file path
# directory = 'C:/dataset'
# if not os.path.exists(directory):
#     os.makedirs(directory)

# zip_file_path = os.path.join(directory, 'downloaded_file.zip')

# # Download the ZIP file
# response = requests.get(download_url)
# if response.status_code == 200:
#     with open(zip_file_path, 'wb') as file:
#         file.write(response.content)
#     print(f"File downloaded successfully to {zip_file_path}")
# else:
#     print("Failed to download file:", response.status_code)
#     response.raise_for_status()

# # Unzip the file
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(directory)
#     print(f"Files extracted to {directory}")


# # Data processing (same as before)
# directory = 'C:/dataset' 

# pattern = re.compile(r'(\"resourceType\": \"(Patient|Encounter|Condition|Observation|Procedure|Medication)\")')

# #classes = ['Patient', 'Encounter', 'Condition', 'Observation', 'Procedure', 'Medication']
# classes = ['Patient', 'Encounter', 'Condition', 'Observation', 'Procedure']#, 'Medication']

# current_resource_type = None
# new_resource_type = None
# resource_type = None
# current_id = None
# new_id = None
# current_references = []
# current_codes = []
# display = []
# current_patient_id = None
# previous_display = None 
# previous_code = None 
# code_starts_with_LA = False  # Flag to track if the current code starts with "LA"

# # Process the files and populate the data dictionary (same as before)
# data = {'ResourceType': [], 'ID': [], 'References': [], 'Codes': [], 'PatientID': []}

# counts = {resource_type: 1 for resource_type in classes}

# INIT = False

# for filename in os.listdir(directory):
#     if filename.endswith('.json'):
#         with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
#             for line in f:
#                 match = pattern.search(line)
#                 if '"resourceType": "' in line and not match:
#                     resource_type = line.split('"')[3]
#                     INIT = False
#                 if match:
#                     if match.group(2) in classes:
#                         Let = match.group(2)[:2]
#                         #print("resource_type", Let)
#                         new_resource_type = match.group(2)
#                         INIT = True
#                         next_line = next(f, None)
#                         if next_line and '"id": "' in next_line:
#                             new_id = next_line.split('"id": "')[1].split('"')[0]
#                 if current_resource_type == new_resource_type and '"reference": "urn:uuid:' in line and INIT:
#                     reference = line.split(':')[3].split('"')[0]
#                     current_references.append(reference)    
                    
#                 if current_resource_type == new_resource_type and '"code": "' in line and INIT:
#                     code = line.split('"code": "')[1].split('"')[0]
#                     if any(char.isdigit() for char in code):
#                         #print(current_patient_id, code)
#                         next_line = next(f, None)
#                         if next_line and '"display": "' in next_line:
#                             if code.startswith("LA") and previous_display is not None:                    
#                                 display = previous_display + ': ' + next_line.split('"display": "')[1].split('"')[0]
#                                 code = previous_code
#                                 #current_codes[-1] = (previous_code, display)
#                                 current_codes[-1] = (Let + previous_code, display)
#                                 code_starts_with_LA = True
#                             else:
#                                 display = next_line.split('"display": "')[1].split('"')[0]
#                                 #current_codes.append((code, display))
#                                 current_codes.append((Let + code, display))
#                                 code_starts_with_LA = False
#                                 #print('(code, display)', (current_patient_id, code, display))
#                         # Update previous display only if the code doesn't start withp "LA"
#                         if not code_starts_with_LA:
#                             previous_display = display
#                             previous_code = code

                 
#                 if current_resource_type != new_resource_type  and INIT:
#                     if current_resource_type and current_id:
#                         data['ResourceType'].append(current_resource_type)
#                         data['ID'].append(current_id)
#                         if current_resource_type == 'Patient':
#                             current_patient_id = current_id
#                         data['References'].append(tuple(current_references))
#                         data['Codes'].append(tuple(current_codes))
#                         data['PatientID'].append(current_patient_id)
#                         current_references = []
#                         current_codes = []
#                     current_resource_type = new_resource_type    
#                     current_id = new_id

# df = pd.DataFrame(data)
# flat_df = df.explode('Codes').reset_index().iloc[:, [1, 4, 5]]
# Codes = [code[0] for code in flat_df['Codes']]
# Displays = [display[1] for display in flat_df['Codes']]
# flat_df['Codes'] = Codes
# flat_df['Displays'] = Displays
# flat_df = flat_df[['PatientID', 'Codes', 'Displays', 'ResourceType']]

# # Generate co-occurrence matrices (same as before)
# #en_df = flat_df[flat_df['ResourceType'] == 'Encounter']
# co_df = flat_df[flat_df['ResourceType'] == 'Condition']
# ob_df = flat_df[flat_df['ResourceType'] == 'Observation']
# pr_df = flat_df[flat_df['ResourceType'] == 'Procedure']
# #me_df = flat_df[flat_df['ResourceType'] == 'Medication']

# resources = [flat_df, co_df, ob_df, pr_df]
# co_occurrence_matrices = {}

# for r in resources:
#     icd_patient = r.pivot_table(index='PatientID', columns='Codes', aggfunc='size', fill_value=0)
#     icd_patient = icd_patient.loc[:, (icd_patient != 0).any(axis=0)]
#     co_occurrence_matrix = icd_patient.T.dot(icd_patient)
#     np.fill_diagonal(co_occurrence_matrix.values, 0)
#     if r.equals(flat_df):
#         co_occurrence_matrices['Main'] = co_occurrence_matrix
#     elif r.equals(co_df):
#         co_occurrence_matrices['Condition'] = co_occurrence_matrix
#     elif r.equals(ob_df):
#         co_occurrence_matrices['Observation'] = co_occurrence_matrix
#     elif r.equals(pr_df):
#         co_occurrence_matrices['Procedure'] = co_occurrence_matrix

# graph = nx.from_pandas_adjacency(co_occurrence_matrices['Main'])

# # Define colors for each subgroup (same as before)
# SUBGROUP_COLORS = {
#     'Condition': "#00bfff",
#     'Observation': "#ffc0cb",
#     'Procedure': "#9a31a8"
# }

# # Dash application setup
# app = dash.Dash(__name__)
# server = app.server

# app.layout = html.Div([
#     html.H1("Co-Occurrences in FHIR Codes"),
#     html.Div([
#         html.Label("Select the number of nodes to visualize:"),
#         dcc.Slider(
#             id='num-nodes-slider',
#             min=1,
#             max=10,
#             step=1,
#             value=1,
#             marks={i: str(i) for i in range(1, 11)},
#             tooltip={"placement": "bottom", "always_visible": True}
#         )
#     ]),
#     html.Div([
#         html.Label("Select a code:"),
#         dcc.Dropdown(
#             id='code-dropdown',
#             options=[{'label': code[2:], 'value': code} for code in co_occurrence_matrices['Main'].columns],
#             placeholder="Select a code",
#             clearable=False
#         )
#     ]),
#     dcc.Checklist(
#         id='show-labels',
#         options=[{'label': 'Show Labels', 'value': 'show'}],
#         value=[] #'show'
#     ),
#     html.Iframe(id='graph-iframe', style={'width': '100%', 'height': '600px'})
# ])

# @app.callback(
#     Output('graph-iframe', 'srcDoc'),
#     Input('code-dropdown', 'value'),
#     Input('num-nodes-slider', 'value'),
#     Input('show-labels', 'value')
# )
# def generate_graph(selected_code, num_nodes_to_visualize, show_labels):
#     if not selected_code:
#         return ""

#     net = Network(notebook=True)
#     main_df = co_occurrence_matrices['Main']

#     neighbors_sorted = main_df.loc[selected_code].sort_values(ascending=False)
#     top_neighbors = list(neighbors_sorted.index[:15])

#     def add_nodes_edges(graph, child_df, prefix, group_name):
#         top_neighbor = None
#         for neighbor_code in neighbors_sorted.index:
#             if neighbor_code.startswith(prefix):
#                 top_neighbor = neighbor_code
#                 break

#         if top_neighbor:
#             net.add_node(selected_code, title=selected_code, label=selected_code if 'show' in show_labels else selected_code[2:], color=SUBGROUP_COLORS.get(group_name, 'gray'))
#             net.add_node(top_neighbor, title=top_neighbor, label=top_neighbor if 'show' in show_labels else top_neighbor[2:], color=SUBGROUP_COLORS.get(group_name, 'gray'))

#             net.add_edge(selected_code, top_neighbor, value=int(main_df.loc[selected_code, top_neighbor]))

#             top_neighbor_row = child_df.loc[top_neighbor].sort_values(ascending=False)
#             top_neighbors = list(top_neighbor_row.index[:num_nodes_to_visualize])

#             for neighbor in top_neighbors:
#                 if neighbor != top_neighbor and child_df.loc[top_neighbor, neighbor] > 0:
#                     net.add_node(neighbor, title=neighbor, label=neighbor if 'show' in show_labels else neighbor[2:], color=SUBGROUP_COLORS.get(group_name, 'gray'))
#                     net.add_edge(top_neighbor, neighbor, value=int(child_df.loc[top_neighbor, neighbor]))

#     if 'Condition' in co_occurrence_matrices:
#         add_nodes_edges(net, co_occurrence_matrices['Condition'], 'Co', 'Condition')
#     if 'Observation' in co_occurrence_matrices:
#         add_nodes_edges(net, co_occurrence_matrices['Observation'], 'Ob', 'Observation')
#     if 'Procedure' in co_occurrence_matrices:
#         add_nodes_edges(net, co_occurrence_matrices['Procedure'], 'Pr', 'Procedure')

#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
#     temp_file_name = temp_file.name
#     temp_file.close()

#     net.show(temp_file_name)
#     return open(temp_file_name, 'r').read()

# if __name__ == '__main__':
#     app.run_server(debug=True, port=8053)

################################################################### DELL-READ-FLAT_DF ##############################################################

import os
import pandas as pd
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from pyvis.network import Network
import tempfile
import base64
import io
import numpy as np
import plotly.figure_factory as ff
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


SUBGROUP_COLORS = {
    'Condition': "#00bfff",
    'Observation': "#ffc0cb",
    'Procedure': "#9a31a8"
}

# Dash application setup
app = dash.Dash(__name__)
server = app.server

# app.layout = html.Div([
#     html.H1("Co-Occurrences in FHIR Codes"),
#     dcc.Upload(
#         id='upload-data',
#         children=html.Button('Upload Data'),
#         multiple=False
#     ),
#     html.Div(id='upload-feedback', children='', style={'color': 'red'}),
#     html.Div([
#         html.Label("Select the number of nodes to visualize:"),
#         dcc.Slider(
#             id='num-nodes-slider',
#             min=0,
#             max=10,
#             step=1,
#             value=1,
#             marks={i: str(i) for i in range(1, 11)},
#             tooltip={"placement": "bottom", "always_visible": True}
#         )
#     ]),
#     html.Div([
#         html.Label("Select a code:"),
#         dcc.Dropdown(
#             id='code-dropdown',
#             options=[],  # Options will be populated after loading data
#             placeholder="Select a code",
#             clearable=False
#         )
#     ]),
#     dcc.Checklist(
#         id='show-labels',
#         options=[{'label': 'Show Labels', 'value': 'show'}],
#         value=[]  # Start with an empty list so labels are not shown by default
#     ),
#     dcc.Loading(
#         id="loading",
#         type="circle",
#         children=[
#             html.Div(id='data-container', style={'display': 'none'}),
#             html.Div(id='data-loading-message', children='Data is still loading...')
#         ]
#     ),
#     html.Iframe(id='graph-iframe', style={'width': '100%', 'height': '600px'}),
#     dcc.Graph(id='bar-chart', style={'height': '600px'}),  # Set height for the bar chart
#     dcc.Graph(id='dendrogram', style={'height': '1000px'}),  # Set height for the dendrogram
#     dcc.Store(id='data-store')  # Hidden store to keep data
# ])

app.layout = html.Div([
    html.H1("Co-Occurrences in FHIR Codes"),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload Data'),
        multiple=False
    ),
    html.Div(id='upload-feedback', children='', style={'color': 'red'}),
    html.Div([
        html.Label("Select the number of nodes to visualize:"),
        dcc.Slider(
            id='num-nodes-slider',
            min=0,
            max=10,
            step=1,
            value=1,
            marks={i: str(i) for i in range(1, 11)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ]),
    html.Div([
        html.Label("Select a code:"),
        dcc.Dropdown(
            id='code-dropdown',
            options=[],  # Options will be populated after loading data
            placeholder="Select a code",
            clearable=False
        )
    ]),
    dcc.Checklist(
        id='show-labels',
        options=[{'label': 'Show Labels', 'value': 'show'}],
        value=[]  # Start with an empty list so labels are not shown by default
    ),
    
    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            html.Div(id='data-container', style={'display': 'none'}),  # Hidden div for callbacks
            html.Div(id='data-loading-message', children='Data is still loading...')
        ]
    ),

    # Graphs positioned side-by-side using CSS flexbox
    html.Div([
        # Left column - PyVis graph
        html.Div([
            html.Iframe(id='graph-iframe', style={'width': '100%', 'height': '600px'}),
        ], style={'flex': '1', 'padding': '10px'}),  # PyVis on the left side, takes 50% of space

        # Right column - Bar chart and dendrogram stacked
        html.Div([
            dcc.Graph(id='dendrogram', style={'height': '300px'}),  # Dendrogram below bar chart
            dcc.Graph(id='bar-chart', style={'height': '300px'})  # Bar chart on top , 'margin-bottom': '20px'
        ], style={'flex': '1', 'padding': '10px'}),  # Bar chart and dendrogram on the right side, takes 50% of space

    ], style={'display': 'flex', 'flex-direction': 'row'}),  # Use flexbox to position the graphs side by side
    
    dcc.Store(id='data-store')  # Hidden store to keep data
])



def fetch_and_process_data(file_content):
    try:
        # Read CSV data from uploaded content
        flat_df = pd.read_parquet(io.BytesIO(file_content))
        print('flat_df', flat_df)
        
        # Check for required columns
        required_columns = ['PatientID', 'Codes', 'Displays', 'ResourceType']
        missing_columns = [col for col in required_columns if col not in flat_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")
        
        condition_df = flat_df[flat_df['ResourceType'] == 'Condition']
        observation_df = flat_df[flat_df['ResourceType'] == 'Observation']
        procedure_df = flat_df[flat_df['ResourceType'] == 'Procedure']
        
        # Create co-occurrence matrices
        def create_co_occurrence_matrix(df):
            if df.empty:
                return pd.DataFrame()
            patient_matrix = df.pivot_table(index='PatientID', columns='Codes', aggfunc='size', fill_value=0)
            patient_matrix = patient_matrix.loc[:, (patient_matrix != 0).any(axis=0)]
            co_occurrence_matrix = patient_matrix.T.dot(patient_matrix)
            np.fill_diagonal(co_occurrence_matrix.values, 0)
            return co_occurrence_matrix
        
        co_occurrence_matrices = {
            'Main': create_co_occurrence_matrix(flat_df),
            'Condition': create_co_occurrence_matrix(condition_df),
            'Observation': create_co_occurrence_matrix(observation_df),
            'Procedure': create_co_occurrence_matrix(procedure_df)
        }

        return {'success': True, 'message': 'Data is loaded.', 'data': co_occurrence_matrices}
    
    except Exception as e:
        return {'success': False, 'message': f"Error: {e}", 'data': {'Main': pd.DataFrame(), 'Condition': pd.DataFrame(), 'Observation': pd.DataFrame(), 'Procedure': pd.DataFrame()}}

@app.callback(
    Output('upload-feedback', 'children'),
    Output('data-container', 'style'),
    Output('code-dropdown', 'options'),
    Output('data-store', 'data'),
    Input('upload-data', 'contents')
)
def upload_file(file_content):
    feedback_message = ""
    data_style = {'display': 'none'}
    options = []
    data = {}

    if file_content:
        # Decode and process uploaded file
        content_type, content_string = file_content.split(',')
        decoded = base64.b64decode(content_string)
        result = fetch_and_process_data(decoded)
        if result['success']:
            co_occurrence_matrices = result['data']
            
            # Get the columns for the dropdown options
            options = [{'label': code[2:], 'value': code} for code in co_occurrence_matrices.get('Main', pd.DataFrame()).columns]
            
            # Convert DataFrames to JSON-serializable dictionaries
            data = {
                'co_occurrence_matrices': {
                    key: matrix.to_dict() for key, matrix in co_occurrence_matrices.items()
                }
            }
            
            feedback_message = result['message']
            data_style = {'display': 'block'}
        else:
            feedback_message = result['message']

    return feedback_message, data_style, options, data

@app.callback(
    Output('graph-iframe', 'srcDoc'),
    Input('code-dropdown', 'value'),
    Input('num-nodes-slider', 'value'),
    Input('show-labels', 'value'),
    State('data-store', 'data')
)

# Define colors for each subgroup
# SUBGROUP_COLORS = {
#     'ICD': "#00bfff",
#     'LOINC': "#ffc0cb",
#     'OPS': "#9a31a8"
# }


def update_graph(selected_code, num_nodes_to_visualize, show_labels, data):
    if not selected_code:
        return ""

    net = Network(notebook=True, cdn_resources='remote')
    co_occurrence_matrices = data.get('co_occurrence_matrices', {})
    main_df = pd.DataFrame(co_occurrence_matrices.get('Main', {}))
    print('main_df', main_df)

    # Step 1: Get neighbors of the selected code and sort them
    neighbors_sorted = main_df.loc[selected_code].sort_values(ascending=False)
    top_neighbors = list(neighbors_sorted.index[:15])

    # Print selected code and its top neighbors
    print(f"\nSelected code: {selected_code}")
    print(f"Top neighbors of {selected_code}: {top_neighbors}")

    code_patient_group = flat_df.groupby('Codes')['PatientID'].nunique()
    selected_code_occurrence = code_patient_group.get(selected_code, 0)  # Get occurrence count or 0 if code not found
    print(f"Occurrence count for {selected_code}: {selected_code_occurrence}")

    def add_nodes_edges(graph, child_df, prefix, group_name):
        top_neighbor = None
        for neighbor_code in neighbors_sorted.index:
            if neighbor_code.startswith(prefix):
                top_neighbor = neighbor_code
                break

        if top_neighbor:
            if 'show' in show_labels:
                selected_code_label = flat_df.loc[flat_df['Codes'] == selected_code, 'Displays'].iloc[0]
                top_neighbor_label = flat_df.loc[flat_df['Codes'] == top_neighbor, 'Displays'].iloc[0]
            else:
                selected_code_label = selected_code[2:]
                top_neighbor_label = top_neighbor[2:]

            group_name1 = 'Condition' if selected_code in co_occurrence_matrices.get('Condition', {}) else \
                          'Observation' if selected_code in co_occurrence_matrices.get('Observation', {}) else \
                          'Procedure' if selected_code in co_occurrence_matrices.get('Procedure', {}) else 'Unknown'

            # Add selected code node if not already present
            if selected_code not in net.get_nodes():
                net.add_node(selected_code, title=selected_code, label=selected_code_label, color=SUBGROUP_COLORS.get(group_name1, 'gray'))

            # Add top neighbor node if not already present
            if top_neighbor not in net.get_nodes():
                net.add_node(top_neighbor, title=top_neighbor, label=top_neighbor_label, color=SUBGROUP_COLORS.get(group_name, 'gray'))

            # Add edge if both nodes exist
            if selected_code in net.get_nodes() and top_neighbor in net.get_nodes():
                edge_value = int(main_df.loc[selected_code, top_neighbor])
                net.add_edge(selected_code, top_neighbor, value=edge_value, color=SUBGROUP_COLORS.get(group_name, 'gray'))

            top_neighbor_row = child_df.loc[top_neighbor].sort_values(ascending=False)
            top_neighbors_list = list(top_neighbor_row.index[:num_nodes_to_visualize])

            top_neighbor_occurrence = code_patient_group.get(top_neighbor, 0)  # Get occurrence count or 0 if code not found
            print(f"\nTop neighbor: {top_neighbor} (Occurrence: {top_neighbor_occurrence})")
            print(f"Edge between {selected_code} and {top_neighbor}: {int(main_df.loc[selected_code, top_neighbor])} occurrences")

            # Add nodes and edges for top neighbor's neighbors
            for neighbor in top_neighbors_list:
                if neighbor != top_neighbor and child_df.loc[top_neighbor, neighbor] > 0:
                    neighbor_label = flat_df.loc[flat_df['Codes'] == neighbor, 'Displays'].iloc[0] if 'show' in show_labels else neighbor[2:]
                    
                    # Add neighbor node if not already present
                    if neighbor not in net.get_nodes():
                        net.add_node(neighbor, title=neighbor, label=neighbor_label, color=SUBGROUP_COLORS.get(group_name, 'gray'))

                    # Add edge if both nodes exist
                    if top_neighbor in net.get_nodes() and neighbor in net.get_nodes():
                        edge_value = int(child_df.loc[top_neighbor, neighbor])
                        net.add_edge(top_neighbor, neighbor, value=edge_value)

            # Add edges between the neighbors
            for i in range(len(top_neighbors_list)):
                for j in range(i + 1, len(top_neighbors_list)):
                    neighbor1 = top_neighbors_list[i]
                    neighbor2 = top_neighbors_list[j]
                    
                    # Check if there's a co-occurrence between the two neighbors
                    if neighbor1 in child_df.index and neighbor2 in child_df.columns:
                        count = child_df.loc[neighbor1, neighbor2]
                        if count > 0:
                            # Add edge if both nodes exist
                            if neighbor1 in net.get_nodes() and neighbor2 in net.get_nodes():
                                net.add_edge(neighbor1, neighbor2, value=int(count), color=SUBGROUP_COLORS.get(group_name, 'gray'))

                                # Print occurrence and edge counts
                                neighbor1_occurrence = code_patient_group.get(neighbor1, 0)
                                neighbor2_occurrence = code_patient_group.get(neighbor2, 0)
                                print(f"Neighbor 1: {neighbor1} (Occurrence: {neighbor1_occurrence})")
                                print(f"Neighbor 2: {neighbor2} (Occurrence: {neighbor2_occurrence})")
                                print(f"Edge between {neighbor1} and {neighbor2}: {count} occurrences")
                            
    if 'Condition' in co_occurrence_matrices:
        add_nodes_edges(net, pd.DataFrame(co_occurrence_matrices['Condition']), 'Co', 'Condition')
    if 'Observation' in co_occurrence_matrices:
        add_nodes_edges(net, pd.DataFrame(co_occurrence_matrices['Observation']), 'Ob', 'Observation')
    if 'Procedure' in co_occurrence_matrices:
        add_nodes_edges(net, pd.DataFrame(co_occurrence_matrices['Procedure']), 'Pr', 'Procedure')

    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix='.html')
    temp_file_name = temp_file.name
    temp_file.close()

    net.show(temp_file_name)
    return open(temp_file_name, 'r').read()



def create_dendrogram_plot(cooccurrence_array, labels):
    fig, ax = plt.subplots(figsize=(20,5))
    linked = linkage(cooccurrence_array, 'ward')
    sch.dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True, labels=labels, ax=ax)
    plt.title('Dendrogram for Clustering')
    plt.xlabel('Code')
    plt.ylabel('Distance')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close(fig)
    
    return base64.b64encode(img_bytes.getvalue()).decode('utf-8')



@app.callback(
    [Output('bar-chart', 'figure'),
     Output('dendrogram', 'figure')],
    [Input('code-dropdown', 'value'),
     Input('show-labels', 'value'),
     Input('num-nodes-slider', 'value')],
    State('data-store', 'data')
)
def update_charts(selected_code, show_labels, slider_value, data):
    if not selected_code:
        return (
            {
                'data': [],
                'layout': {'title': 'Select a code to see the bar chart'}
            },
            {
                'data': [],
                'layout': {'title': 'Dendrogram not available'}
            }
        )

    # Retrieve the co-occurrence matrices and calculate occurrence counts
    co_occurrence_matrices = data.get('co_occurrence_matrices', {})
    main_df = pd.DataFrame(co_occurrence_matrices.get('Main', {}))
    
    frequency_distribution = main_df.sum(axis=1)
    total_sum = frequency_distribution.sum()
    code_patient_group = frequency_distribution / total_sum


    # Get occurrences for selected code
    #code_patient_group = flat_df.groupby('Codes')['PatientID'].nunique()
    selected_code_occurrence = code_patient_group.get(selected_code, 0)

    # Get top neighbors
    neighbors_sorted = main_df.loc[selected_code].sort_values(ascending=False)
    
    # Define categories and initialize grouped neighbors
    categories = ['Condition', 'Observation', 'Procedure']
    grouped_neighbors = {cat: [] for cat in categories}
    
    # Sort neighbors into categories
    for neighbor in neighbors_sorted.index:
        for cat in categories:
            if neighbor in co_occurrence_matrices.get(cat, {}):
                grouped_neighbors[cat].append(neighbor)
                break

    # Create the sorted list of top neighbors by alternating categories
    sorted_neighbors = []
    for _ in range(len(neighbors_sorted)):
        for cat in categories:
            if grouped_neighbors[cat]:
                sorted_neighbors.append(grouped_neighbors[cat].pop(0))
                if len(sorted_neighbors) >= 10:  # Limit to top 10 neighbors
                    break
        if len(sorted_neighbors) >= 10:
            break

    # Ensure slider_value is defined and within a reasonable range
    if slider_value is None:
        slider_value = 0  # Default to 0 if slider_value is not provided

    # Determine the number of neighbors to display based on slider value
    num_neighbors_to_display = min(3 + slider_value * 3, len(sorted_neighbors))

    # Prepare data for bar chart
    bar_data = []
    x_labels = []
    y_values = []
    line_widths = []
    bar_colors = []

    for neighbor in sorted_neighbors[:num_neighbors_to_display]:
        occurrence_count = code_patient_group.get(neighbor, 0)
        
        # Determine the x-axis label based on the show-labels option
        if 'show' in show_labels:
            neighbor_label = flat_df.loc[flat_df['Codes'] == neighbor, 'Displays'].iloc[0]
            selected_code_label = flat_df.loc[flat_df['Codes'] == selected_code, 'Displays'].iloc[0]
        else:
            neighbor_label = neighbor[2:]
            selected_code_label = selected_code[2:]

        bar_data.append({'x': neighbor_label, 'y': occurrence_count, 'code': neighbor})
        x_labels.append(neighbor_label)
        y_values.append(occurrence_count)
        
        # Set line width based on whether it is the selected code
        if neighbor == selected_code:
            line_widths.append(5)  # Thicker line for selected code
        else:
            line_widths.append(1)  # Thinner line for other bars

        # Determine color for the bar based on subgroup
        color = 'gray'  # Default color
        for subgroup, color_code in SUBGROUP_COLORS.items():
            if neighbor in co_occurrence_matrices.get(subgroup, {}):
                color = color_code
                break
        bar_colors.append(color)

    # Sort the bar data based on the 'code' value
    bar_data_sorted = sorted(bar_data, key=lambda x: x['code'])
    
    # Extract sorted x and y values for the bar chart
    sorted_x = [item['x'] for item in bar_data_sorted]
    sorted_y = [item['y'] for item in bar_data_sorted]
    sorted_line_widths = [line_widths[x_labels.index(item['x'])] for item in bar_data_sorted]
    sorted_colors = [bar_colors[x_labels.index(item['x'])] for item in bar_data_sorted]

    # Create the bar chart
    bar_chart_figure = {
        'data': [{
            'x': sorted_x,
            'y': sorted_y,
            'type': 'bar',
            'name': 'Occurrences',
            'marker': {'color': sorted_colors},  # Assign colors to the barss
            'line': {'width': sorted_line_widths},  # Set line width
            'text': sorted_x,
            'textposition': 'outside'
        }],
        'layout': {
            'title': f'Frequency Distribution of {selected_code_label} and Top Neighbors',
            'xaxis': {'title': 'Codes'},
            'yaxis': {'title': 'Frequency'}
        }
    }


    try:
        # List of codes to include in the sub matrix
        codes_of_interest = [selected_code] + sorted_neighbors[:num_neighbors_to_display]

        def create_sub_cooccurrence_matrix(cooccurrence_dict, codes):
            # Filter codes to ensure they are in the dictionary
            valid_codes = [code for code in codes if code in cooccurrence_dict]

            # Create a DataFrame for the sub matrix
            sub_matrix = pd.DataFrame(
                {code: {sub_code: cooccurrence_dict.get(code, {}).get(sub_code, 0) for sub_code in valid_codes} for code in valid_codes}
            ).fillna(0)

            return sub_matrix

        # Extract the main co-occurrence dictionary
        co_dict = co_occurrence_matrices.get('Main', {})

        # Create the sub co-occurrence matrix
        cooccurrence_dict = create_sub_cooccurrence_matrix(co_dict, codes_of_interest)
        
        # Symmetrize the matrix (since co-occurrence is undirected)
        cooccurrence_matrix = cooccurrence_dict.dot(cooccurrence_dict.T)

        # Replace NaN values with 0 (if any)
        cooccurrence_matrix = cooccurrence_matrix.fillna(0)

        # Convert matrix to array for clustering
        cooccurrence_array = cooccurrence_matrix.values

        # Perform Agglomerative Clustering
        clustering = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
        cluster_labels = clustering.fit_predict(cooccurrence_array)

        # Add cluster labels to the matrix
        cooccurrence_matrix['Cluster'] = cluster_labels

        # Generate the dendrogram plot
        dendrogram_base64 = create_dendrogram_plot(cooccurrence_array, cooccurrence_matrix.index.tolist())

        # Return the base64 string for the dendrogram
        dendrogram_figure = {
            'data': [{
                'type': 'image',
                'source': f'data:image/png;base64,{dendrogram_base64}',
                'sizing': 'contain',
                'name': 'Dendrogram'
            }],
            'layout': {
                'title': 'Dendrogram',
                'xaxis': {'visible': False},
                'yaxis': {'visible': False}
            }
        }

        return bar_chart_figure, dendrogram_figure

    except Exception as e:
        print(f"Error in generating dendrogram: {e}")
        return bar_chart_figure, {'data': [], 'layout': {'title': 'Error generating dendrogram'}}


if __name__ == '__main__':
    app.run_server(debug=True, port=8052)


# In[ ]:




