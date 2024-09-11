# In[ ]:

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
            dcc.Store(id='codes-of-interest-store'),
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

        return {'success': True, 'message': 'Data is loaded.', 'data': {'flat_df': flat_df.to_dict(), 'co_occurrence_matrices': co_occurrence_matrices}}
    
    except Exception as e:
        return {'success': False, 'message': f"Error: {e}", 'data': {'flat_df': pd.DataFrame().to_dict(), 'co_occurrence_matrices': {}}}

@app.callback(
    Output('upload-feedback', 'children'),
    Output('data-container', 'style'),
    Output('code-dropdown', 'options'),
    Output('data-store', 'data'),  # Store `flat_df` and matrices here
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
            co_occurrence_matrices = result['data']['co_occurrence_matrices']
            flat_df = result['data']['flat_df']
            
            # Get the columns for the dropdown options
            options = [{'label': code[2:], 'value': code} for code in co_occurrence_matrices.get('Main', pd.DataFrame()).columns]
            
            # Store data in the dcc.Store component, including `flat_df`
            data = {
                'flat_df': flat_df,  # Store the `flat_df` here
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
    [Output('graph-iframe', 'srcDoc'),
     Output('codes-of-interest-store', 'data')],
    [Input('code-dropdown', 'value'),
     Input('num-nodes-slider', 'value'),
     Input('show-labels', 'value')],
    State('data-store', 'data')
)


def update_graph(selected_code, num_nodes_to_visualize, show_labels, data):
    if not selected_code:
        # Return empty values if no code is selected
        return "", {'codes_of_interest': [], 'top_neighbor_info': {}}

    net = Network(notebook=True, cdn_resources='remote')
    flat_df = pd.DataFrame(data.get('flat_df', {}))
    print('flat_df',flat_df.columns)
    co_occurrence_matrices = data.get('co_occurrence_matrices', {})
    main_df = pd.DataFrame(co_occurrence_matrices.get('Main', {}))
    print('main_df', main_df)

    # Step 1: Get neighbors of the selected code and sort them
    neighbors_sorted = main_df.loc[selected_code].sort_values(ascending=False)
    top_neighbors = list(neighbors_sorted.index[:])
    
    codes_of_interest = [selected_code]
    top_neighbor_info = {}

    # Print selected code and its top neighbors
    print(f"\nSelected code: {selected_code}")
    print(f"Top neighbors of {selected_code}: {top_neighbors}")

#     code_patient_group = flat_df.groupby('Codes')['PatientID'].nunique()
#     selected_code_occurrence = code_patient_group.get(selected_code, 0)  # Get occurrence count or 0 if code not found
#     print(f"Occurrence count for {selected_code}: {selected_code_occurrence}")

    def add_nodes_edges(graph, child_df, prefix, group_name):
        top_neighbor = None
        for neighbor_code in neighbors_sorted.index:
            if neighbor_code.startswith(prefix):
                top_neighbor = neighbor_code
                break

        if top_neighbor:
            top_neighbor_info['top_neighbor'] = top_neighbor
            top_neighbor_row = child_df.loc[top_neighbor].sort_values(ascending=False)
            top_neighbors_list = list(top_neighbor_row.index[:num_nodes_to_visualize])
            top_neighbor_info['top_neighbors_list'] = top_neighbors_list
            
            codes_of_interest.extend([top_neighbor] + top_neighbors_list)
            print('codes_of_interest in update_graph', codes_of_interest)
            
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
            print('top_neighbor', top_neighbor)
            print('top_neighbors_list', top_neighbors_list)

#             top_neighbor_occurrence = code_patient_group.get(top_neighbor, 0)  # Get occurrence count or 0 if code not found
#             print(f"\nTop neighbor: {top_neighbor} (Occurrence: {top_neighbor_occurrence})")
#             print(f"Edge between {selected_code} and {top_neighbor}: {int(main_df.loc[selected_code, top_neighbor])} occurrences")

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

#                                 # Print occurrence and edge counts
#                                 neighbor1_occurrence = code_patient_group.get(neighbor1, 0)
#                                 neighbor2_occurrence = code_patient_group.get(neighbor2, 0)
#                                 print(f"Neighbor 1: {neighbor1} (Occurrence: {neighbor1_occurrence})")
#                                 print(f"Neighbor 2: {neighbor2} (Occurrence: {neighbor2_occurrence})")
#                                 print(f"Edge between {neighbor1} and {neighbor2}: {count} occurrences")
                            
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
    return open(temp_file_name, 'r').read(), {'codes_of_interest': codes_of_interest, 'top_neighbor_info': top_neighbor_info}




import plotly.figure_factory as ff


def create_dendrogram_plot(cooccurrence_array, labels, flat_df, show_labels):
    # Adjust labels based on the 'show_labels' input
    if 'show' in show_labels:
        # Use 'Displays' from flat_df for labels
        labels = [
            flat_df.loc[flat_df['Codes'] == label, 'Displays'].iloc[0] 
            if not flat_df.loc[flat_df['Codes'] == label, 'Displays'].empty 
            else label  # Fallback to code if display is missing
            for label in labels
        ]
    else:
        # Use truncated codes (remove the first two characters)
        labels = [label[2:] for label in labels]

    # Create the dendrogram plot with Plotly
    fig = ff.create_dendrogram(cooccurrence_array, orientation='bottom', labels=labels)
    
    # Update layout to improve appearance
    fig.update_layout(
        title='Dendrogram for Clustering',
        xaxis_title='Code',
        yaxis_title='Distance',
        xaxis={'tickangle': -45},  # Rotate labels for better readability
    )
    
    return fig


@app.callback(
    [Output('bar-chart', 'figure'),
     Output('dendrogram', 'figure')],
    [Input('code-dropdown', 'value'),
     Input('show-labels', 'value'),
     Input('num-nodes-slider', 'value'),
     Input('codes-of-interest-store', 'data')],  # Corrected input to fetch 'codes_of_interest'
    State('data-store', 'data')
)
def update_charts(selected_code, show_labels, slider_value, codes_of_interest, data):
    if not selected_code:
        return (
            {
                'data': [],
                'layout': {'title': 'Bar chart not available'}
            },
            {
                'data': [],
                'layout': {'title': 'Dendrogram not available'}
            }
        )
    
    # Retrieve co-occurrence matrices and flat_df
    co_occurrence_matrices = data.get('co_occurrence_matrices', {})
    flat_df = pd.DataFrame(data.get('flat_df', {}))
    num_neighbors_to_display = slider_value or 0
    
    # Compute frequency distribution
    main_df = pd.DataFrame(co_occurrence_matrices.get('Main', {}))
    frequency_distribution = main_df.sum(axis=1)
    total_sum = frequency_distribution.sum()
    total_freq_dist = frequency_distribution / total_sum
    
    # Get the selected code's label
    selected_code_label = flat_df.loc[flat_df['Codes'] == selected_code, 'Displays'].iloc[0] if 'show' in show_labels else selected_code[2:]

    # Ensure codes_of_interest is a list
    codes_of_interest = codes_of_interest.get('codes_of_interest', [])
    print("Codes of interest:", codes_of_interest)

    # Prepare bar chart data
    bar_data = []
    x_labels = []
    y_values = []
    line_widths = []
    bar_colors = []

    for neighbor in codes_of_interest:
        occurrence_count = total_freq_dist.get(neighbor, 0)
        neighbor_label = flat_df.loc[flat_df['Codes'] == neighbor, 'Displays'].iloc[0] if 'show' in show_labels else neighbor[2:]
        bar_data.append({'x': neighbor_label, 'y': occurrence_count, 'code': neighbor})
        x_labels.append(neighbor_label)
        y_values.append(occurrence_count)
        line_widths.append(5 if neighbor == selected_code else 1)
        color = 'gray'
        for subgroup, color_code in SUBGROUP_COLORS.items():
            if neighbor in co_occurrence_matrices.get(subgroup, {}):
                color = color_code
                break
        bar_colors.append(color)

    # Sort the bar data based on the 'code' value
    bar_data_sorted = sorted(bar_data, key=lambda x: x['code'])
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
            'marker': {'color': sorted_colors},
            'line': {'width': sorted_line_widths},
            'text': sorted_x,
            'textposition': 'outside'
        }],
        'layout': {
            'title': f'Frequency Distribution of {selected_code_label} and its Top Neighbors',
            'xaxis': {'title': 'Codes'},
            'yaxis': {'title': 'Frequency'}
        }
    }

    # Create dendrogram figure
    try:
        if len(codes_of_interest) < 2:
            raise ValueError("Not enough codes for clustering")

        def create_sub_cooccurrence_matrix(cooccurrence_dict, codes):
            valid_codes = [code for code in codes if code in cooccurrence_dict]
            if not valid_codes:
                raise ValueError("No valid codes found for sub-co-occurrence matrix")
            sub_matrix = pd.DataFrame(
                {code: {sub_code: cooccurrence_dict.get(code, {}).get(sub_code, 0) for sub_code in valid_codes} for code in valid_codes}
            ).fillna(0)
            return sub_matrix

        co_dict = co_occurrence_matrices.get('Main', {})
        cooccurrence_dict = create_sub_cooccurrence_matrix(co_dict, codes_of_interest)
        
        if cooccurrence_dict.shape[0] < 2:
            raise ValueError("Sub-co-occurrence matrix does not have enough samples for clustering")

        cooccurrence_matrix = cooccurrence_dict.dot(cooccurrence_dict.T).fillna(0)
        cooccurrence_array = cooccurrence_matrix.values

        print("Co-occurrence matrix:\n", cooccurrence_matrix)
        print("Co-occurrence array shape:", cooccurrence_array.shape)

        clustering = AgglomerativeClustering(n_clusters=1, metric='euclidean', linkage='ward')
        cluster_labels = clustering.fit_predict(cooccurrence_array)
        cooccurrence_matrix['Cluster'] = cluster_labels

        # Generate dendrogram plot
        dendrogram_figure = create_dendrogram_plot(cooccurrence_array, cooccurrence_matrix.index.tolist(), flat_df, show_labels)

        return bar_chart_figure, dendrogram_figure

    except Exception as e:
        print(f"Error in generating dendrogram: {e}")
        return bar_chart_figure, {'data': [], 'layout': {'title': 'Error generating dendrogram'}}

    
if __name__ == '__main__':
    app.run_server(debug=True, port=8052)



# In[ ]:




