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

import os
import re
import pandas as pd
import numpy as np
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from pyvis.network import Network
import tempfile
import networkx as nx

# Data processing (same as before)
directory = 'C:/dataset' 

pattern = re.compile(r'(\"resourceType\": \"(Patient|Encounter|Condition|Observation|Procedure|Medication)\")')

#classes = ['Patient', 'Encounter', 'Condition', 'Observation', 'Procedure', 'Medication']
classes = ['Patient', 'Encounter', 'Condition', 'Observation', 'Procedure']#, 'Medication']

current_resource_type = None
new_resource_type = None
resource_type = None
current_id = None
new_id = None
current_references = []
current_codes = []
display = []
current_patient_id = None
previous_display = None 
previous_code = None 
code_starts_with_LA = False  # Flag to track if the current code starts with "LA"

# Process the files and populate the data dictionary (same as before)
data = {'ResourceType': [], 'ID': [], 'References': [], 'Codes': [], 'PatientID': []}

counts = {resource_type: 1 for resource_type in classes}

INIT = False

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if '"resourceType": "' in line and not match:
                    resource_type = line.split('"')[3]
                    INIT = False
                if match:
                    if match.group(2) in classes:
                        Let = match.group(2)[:2]
                        #print("resource_type", Let)
                        new_resource_type = match.group(2)
                        INIT = True
                        next_line = next(f, None)
                        if next_line and '"id": "' in next_line:
                            new_id = next_line.split('"id": "')[1].split('"')[0]
                if current_resource_type == new_resource_type and '"reference": "urn:uuid:' in line and INIT:
                    reference = line.split(':')[3].split('"')[0]
                    current_references.append(reference)    
                    
                if current_resource_type == new_resource_type and '"code": "' in line and INIT:
                    code = line.split('"code": "')[1].split('"')[0]
                    if any(char.isdigit() for char in code):
                        #print(current_patient_id, code)
                        next_line = next(f, None)
                        if next_line and '"display": "' in next_line:
                            if code.startswith("LA") and previous_display is not None:                    
                                display = previous_display + ': ' + next_line.split('"display": "')[1].split('"')[0]
                                code = previous_code
                                #current_codes[-1] = (previous_code, display)
                                current_codes[-1] = (Let + previous_code, display)
                                code_starts_with_LA = True
                            else:
                                display = next_line.split('"display": "')[1].split('"')[0]
                                #current_codes.append((code, display))
                                current_codes.append((Let + code, display))
                                code_starts_with_LA = False
                                #print('(code, display)', (current_patient_id, code, display))
                        # Update previous display only if the code doesn't start withp "LA"
                        if not code_starts_with_LA:
                            previous_display = display
                            previous_code = code

                 
                if current_resource_type != new_resource_type  and INIT:
                    if current_resource_type and current_id:
                        data['ResourceType'].append(current_resource_type)
                        data['ID'].append(current_id)
                        if current_resource_type == 'Patient':
                            current_patient_id = current_id
                        data['References'].append(tuple(current_references))
                        data['Codes'].append(tuple(current_codes))
                        data['PatientID'].append(current_patient_id)
                        current_references = []
                        current_codes = []
                    current_resource_type = new_resource_type    
                    current_id = new_id

df = pd.DataFrame(data)
flat_df = df.explode('Codes').reset_index().iloc[:, [1, 4, 5]]
Codes = [code[0] for code in flat_df['Codes']]
Displays = [display[1] for display in flat_df['Codes']]
flat_df['Codes'] = Codes
flat_df['Displays'] = Displays
flat_df = flat_df[['PatientID', 'Codes', 'Displays', 'ResourceType']]

# Generate co-occurrence matrices (same as before)
#en_df = flat_df[flat_df['ResourceType'] == 'Encounter']
co_df = flat_df[flat_df['ResourceType'] == 'Condition']
ob_df = flat_df[flat_df['ResourceType'] == 'Observation']
pr_df = flat_df[flat_df['ResourceType'] == 'Procedure']
#me_df = flat_df[flat_df['ResourceType'] == 'Medication']

resources = [flat_df, co_df, ob_df, pr_df]
co_occurrence_matrices = {}

for r in resources:
    icd_patient = r.pivot_table(index='PatientID', columns='Codes', aggfunc='size', fill_value=0)
    icd_patient = icd_patient.loc[:, (icd_patient != 0).any(axis=0)]
    co_occurrence_matrix = icd_patient.T.dot(icd_patient)
    np.fill_diagonal(co_occurrence_matrix.values, 0)
    if r.equals(flat_df):
        co_occurrence_matrices['Main'] = co_occurrence_matrix
    elif r.equals(co_df):
        co_occurrence_matrices['Condition'] = co_occurrence_matrix
    elif r.equals(ob_df):
        co_occurrence_matrices['Observation'] = co_occurrence_matrix
    elif r.equals(pr_df):
        co_occurrence_matrices['Procedure'] = co_occurrence_matrix

graph = nx.from_pandas_adjacency(co_occurrence_matrices['Main'])

# Define colors for each subgroup (same as before)
SUBGROUP_COLORS = {
    'Condition': "#00bfff",
    'Observation': "#ffc0cb",
    'Procedure': "#9a31a8"
}

# Dash application setup
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Co-Occurrences in FHIR Codes"),
    html.Div([
        html.Label("Select the number of nodes to visualize:"),
        dcc.Slider(
            id='num-nodes-slider',
            min=1,
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
            options=[{'label': code[2:], 'value': code} for code in co_occurrence_matrices['Main'].columns],
            placeholder="Select a code",
            clearable=False
        )
    ]),
    dcc.Checklist(
        id='show-labels',
        options=[{'label': 'Show Labels', 'value': 'show'}],
        value=[] #'show'
    ),
    html.Iframe(id='graph-iframe', style={'width': '100%', 'height': '600px'})
])

@app.callback(
    Output('graph-iframe', 'srcDoc'),
    Input('code-dropdown', 'value'),
    Input('num-nodes-slider', 'value'),
    Input('show-labels', 'value')
)
def generate_graph(selected_code, num_nodes_to_visualize, show_labels):
    if not selected_code:
        return ""

    net = Network(notebook=True)
    main_df = co_occurrence_matrices['Main']

    neighbors_sorted = main_df.loc[selected_code].sort_values(ascending=False)
    top_neighbors = list(neighbors_sorted.index[:15])

    def add_nodes_edges(graph, child_df, prefix, group_name):
        top_neighbor = None
        for neighbor_code in neighbors_sorted.index:
            if neighbor_code.startswith(prefix):
                top_neighbor = neighbor_code
                break

        if top_neighbor:
            net.add_node(selected_code, title=selected_code, label=selected_code if 'show' in show_labels else selected_code[2:], color=SUBGROUP_COLORS.get(group_name, 'gray'))
            net.add_node(top_neighbor, title=top_neighbor, label=top_neighbor if 'show' in show_labels else top_neighbor[2:], color=SUBGROUP_COLORS.get(group_name, 'gray'))

            net.add_edge(selected_code, top_neighbor, value=int(main_df.loc[selected_code, top_neighbor]))

            top_neighbor_row = child_df.loc[top_neighbor].sort_values(ascending=False)
            top_neighbors = list(top_neighbor_row.index[:num_nodes_to_visualize])

            for neighbor in top_neighbors:
                if neighbor != top_neighbor and child_df.loc[top_neighbor, neighbor] > 0:
                    net.add_node(neighbor, title=neighbor, label=neighbor if 'show' in show_labels else neighbor[2:], color=SUBGROUP_COLORS.get(group_name, 'gray'))
                    net.add_edge(top_neighbor, neighbor, value=int(child_df.loc[top_neighbor, neighbor]))

    if 'Condition' in co_occurrence_matrices:
        add_nodes_edges(net, co_occurrence_matrices['Condition'], 'Co', 'Condition')
    if 'Observation' in co_occurrence_matrices:
        add_nodes_edges(net, co_occurrence_matrices['Observation'], 'Ob', 'Observation')
    if 'Procedure' in co_occurrence_matrices:
        add_nodes_edges(net, co_occurrence_matrices['Procedure'], 'Pr', 'Procedure')

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    temp_file_name = temp_file.name
    temp_file.close()

    net.show(temp_file_name)
    return open(temp_file_name, 'r').read()

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)




# In[ ]:




