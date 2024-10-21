#!/usr/bin/env python
# coding: utf-8

# # CoCo: Analysis of Co-Occurrences in FHIR Codes
# **Pelin Genc**  
# **October 22, 2024**
# <!-- This line will not be displayed. -->
# 
# ## 1 Introduction, Motivation and Problem Statement
# The healthcare industry produces vast amounts of complex data, especially in the form of clinical codes such as ICD (International Classification of Diseases), LOINC (Logical Observation Identifiers Names and Codes), and OPS (German Procedure Classification). Analyzing the co-occurrences of these codes can reveal valuable insights into clinical patterns, as well as relationships between diagnoses, tests, and procedures. This application is designed to provide a flexible, interactive platform for visualizing these co-occurrences between Fast Healthcare Interoperability Resources (FHIR) codes using Dash, enabling users to explore the data effectively.
# 
# As datasets grow in size and complexity, interpreting relationships between healthcare codes becomes increasingly challenging. Without specialized visualization tools, healthcare professionals may find it difficult to extract meaningful patterns or insights from the data. The key challenge is to develop a user-friendly interface that allows users to upload datasets and visualize both inter- and intra-relationships across different resource types—such as ICD, LOINC, and OPS. Additionally, such tools should support the exploration of healthcare code co-occurrences to enhance the understanding of patient conditions.
# 
# In clinical decision-making, doctors often need to examine the broader context of a patient’s diagnosis by analyzing associated medical events and conditions. For example, a doctor treating a patient with diabetes may want to investigate which other medical conditions or events frequently co-occur with diabetes. Such an exploration could provide insights into potential comorbidities, complications, or treatment patterns commonly seen in diabetic patients. This type of analysis can inform the doctor’s treatment approach, drawing attention to other conditions that may require consideration or intervention.
# 
# ## 2 FHIR Data and SQL Query
# 
# ### FHIR Data Hierarchy
# The FHIR standard defines a set of resource types.
# 
# **Administrative/Financial Resources:**
# - Claim
# - ExplanationOfBenefit
# - ServiceRequest
# - Coverage
# 
# **Clinical Resources:**
# - Condition
# - Procedure
# - DiagnosticReport
# - Observation
# - Immunization
# - CareTeam
# - CarePlan
# - MedicationRequest
# - Encounter
# 
# **Document/Reference Resources:**
# - DocumentReference
# - ImagingStudy
# - Provenance
# 
# These resource types follow an inter-referencing logic, where a particular resource type may reference codes from other resource types to highlight the dependency and flow of events between them. This relationship can be represented as a directed cycle, as shown in the figure below. The key resource types involved in below cyclic graph are Patient, E: Encounter, C: Condition, D: DiagnosticReport, O: Observation, P: Procedure.
# The key resource types involved in the below cyclic graph are **Patient**, **E: Encounter**, **C: Condition**, **D: DiagnosticReport**, **O: Observation**, **P: Procedure**.
# 
# 
# <!-- ![Directed Cycle of Resource Types in FHIR](Overleaf-images/Cycle-Patient-Encounter-Observation-Condition---.png) -->
# 
# <img src="Overleaf-images/Cycle-Patient-Encounter-Observation-Condition---.png" alt="Directed Cycle of Resource Types in FHIR" width="600"/>
# 
# 
# 
# **Figure 1:** Directed Cycle of Resource Types in FHIR
# 
# As shown in Figure 1, this interconnectedness creates a directional cycle among these resource types, where each type can reference others in a coherent manner. For example, the **Patient** resource serves as a fundamental building block in the healthcare domain. Each Patient can be involved in multiple **Encounters**, such as hospital visits, outpatient appointments, or emergency room admissions. During an Encounter, healthcare providers may gather various types of clinical data, which can include **Observations** (e.g., vital signs, lab results), **Conditions** (diagnoses), and **Procedures** (medical interventions). For instance, an Encounter references the Patient while also linking to various Observations and Conditions that arise during that Encounter. This cyclical nature enables comprehensive data modeling that reflects the complexity of patient care in real-world settings.
# 
# In the implementation of CoCo code, it is important to note that only the codes from the **Observation (LOINC)**, **Procedure (OPS)**, and **Condition (ICD)** resource types are utilized.
# 
# ### Executing SQL Queries in Python
# Below is the pseudocode for executing SQL queries (for ICD as an example) using the trino connector in Python. The queries retrieve not only ICD, OPS, and LOINC data from a FHIR dataset that occurred during a specified time interval but also the full history of those codes for the same patients. 
# 
# ```sql
# WITH TimeIntervalICD AS (
#     SELECT
#         encounter.subject.reference AS PatientID,
#         condition_coding.code AS Codes,
#         FROM_ISO8601_TIMESTAMP(condition.onsetdatetime) AS EncounterDate
#     FROM
#         fhir.qs.Encounter encounter
#         LEFT JOIN UNNEST(encounter.diagnosis) AS encounter_diagnosis ON TRUE
#         LEFT JOIN fhir.qs.Condition condition ON encounter_diagnosis.condition.reference = CONCAT('Condition/', condition.id)
#         LEFT JOIN UNNEST(condition.code.coding) AS condition_coding ON TRUE
#     WHERE
#         condition_coding.code IS NOT NULL
#         AND condition_coding.system = 'http://fhir.de/CodeSystem/bfarm/icd-10-gm'
#         AND FROM_ISO8601_TIMESTAMP(condition.onsetdatetime) BETWEEN TIMESTAMP '2023-02-25 00:00:00' AND TIMESTAMP '2023-02-25 01:00:00'
# ),
# AllPatientICDCodes AS (
#     SELECT
#         encounter.subject.reference AS PatientID,
#         condition_coding.code AS Codes,
#         FROM_ISO8601_TIMESTAMP(condition.onsetdatetime) AS EncounterDate
#     FROM
#         fhir.qs.Encounter encounter
#         LEFT JOIN UNNEST(encounter.diagnosis) AS encounter_diagnosis ON TRUE
#         LEFT JOIN fhir.qs.Condition condition ON encounter_diagnosis.condition.reference = CONCAT('Condition/', condition.id)
#         LEFT JOIN UNNEST(condition.code.coding) AS condition_coding ON TRUE
#     WHERE
#         condition_coding.code IS NOT NULL
#         AND condition_coding.system = 'http://fhir.de/CodeSystem/bfarm/icd-10-gm'
# )
# SELECT
#     p.PatientID,
#     ARRAY_AGG(DISTINCT a.Codes) AS ICDCodesAllTime
# FROM
#     TimeIntervalICD p
# JOIN
#     AllPatientICDCodes a ON p.PatientID = a.PatientID
# GROUP BY
#     p.PatientID
# 

# ## 3 Methodology
# The Dash application is structured around several key components, each with specific input and output arguments. Below is a breakdown of the code:
# 
# ### 3.1 User Interface Components
# 
# #### 3.1.1 Upload Component
# In this section, users can upload the FHIR data previously downloaded via a SQL query. The dataset contains three columns: Patient (with anonymized numbering), Codes, and Resource Types, and is formatted as a Parquet file.
# 
# 
# ![Data Upload Process in the Dash Application.](Overleaf-images/Upload_Data.png)
# 
# 
# #### 3.1.2 Dropdown Menu
# This dropdown menu contains all available codes from the FHIR dataset, along with an option labeled **ALL CODES** that allows users to visualize the broader relationships among all codes.
# 
# ![Dropdown Menu for Code Selection](Overleaf-images/Select_a_code.png)
# 
# 
# #### 3.1.3 Sliders
# The application includes two sliders designed to enhance user interaction and data visualization.
# 
# - **Level Slider for All Codes:** This slider is displayed when the user selects **selected code = 'ALL CODES'**. It allows users to choose from four different levels of hierarchy to visualize the PyVis graph in a hierarchical manner. Each level corresponds to a different depth in the code structure, enabling users both to explore various degrees of relationships among the codes and to avoid the complexity.
# 
# ![Hierarchy Level Slider](Overleaf-images/Hierarchy_Slider.png)
# 
# 
# - **Node Count Slider for Individual Codes:** This slider is utilized when the user selects an individual code. It ranges from 1 to 10, allowing users to specify the number of top neighbors (most co-occurring nodes) to be displayed for the selected code.
# 
# ![Maximum Number of Nodes to Visualize](Overleaf-images/Top_Neighbors_Slider.png)
# 
# #### 3.1.4 Graphs
# The application presents different types of graphs depending on whether an individual code or all codes are selected.
# 
# - **Graphs for Individual Codes:** For individual codes, three types of graphs are provided:
# 
#   - **PyVis Graph:** This graph visualizes the co-occurrences of the top neighbors of an individual code, categorized into three resource types: ICD, OPS, and LOINC. Each resource type is represented by a distinct color, facilitating quick identification of the types of relationships present.
#   
#   - **Dendrogram:** The dendrogram illustrates groups of codes that are likely to occur together, highlighting the likelihood of co-occurrence independent of resource type. Unlike the PyVis graph, which focuses on pairs, the dendrogram can reveal relationships among larger groups of codes, providing a comprehensive view of interconnections.
#   
#   - **Bar Chart:** This chart displays the frequency distribution of the selected code and its top neighbors, offering insights into the prevalence of these codes within the dataset.
#   
#   The maximum number of nodes to be visualized can be adjusted using the top neighbors slider. This functionality is essential for limiting the amount of data presented in the network visualization. The graph generated by this configuration visualizes Individual Codes. It displays the Top Neighbors using PyVis, highlighting clusters in a dendrogram and presenting frequencies in a bar chart. To ensure clarity and prevent overlapping of text labels, the labels are truncated. However, users can hover over a node to view the full label, which provides additional context for the data represented.
# 
#     ![Graphs for Individual Codes](Overleaf-images/Individual_codes-Top_neighbors_3Graphs_Show_Label-Full_label.png)
# 
# - **Graphs for All Codes:**
# 
#     When `ALL_CODES` is selected, a PyVis graph is displayed to illustrate the big picture of all FHIR codes.
# 
#     In the "All Codes" version of the visualization, a slider is provided with four levels to simplify the analysis and reduce complexity. Additionally, the "Enter Code" feature allows users to highlight a specific code among the entire dataset by coloring its node and the edges connected to it in a vibrant lime color. To prevent overlapping labels, the text is truncated, similar to the individual code view. However, users can hover over any node to reveal the full display of the code, ensuring that essential information is still accessible.
# 
#     ![All Codes Visualization with Hierarchy Level and Code Highlighting](Overleaf-images/All_codes-Hierarchy_Level-Enter_code-Show_labels-Full_label2.png)
#     
# ### 3.2 Co-Occurrence Matrix
# The application generates a co-occurrence matrix based on the data processed earlier. This matrix quantifies relationships by counting how many patients share multiple codes. The following code demonstrates how to create a co-occurrence matrix.
# 
# ```python
# # Pivot the DataFrame to create a patient matrix
# patient_matrix = flat_df.pivot_table(index='PatientID', columns='Codes', aggfunc='size', fill_value=0)
# 
# # Generate the co-occurrence matrix
# co_occurrence_matrix = patient_matrix.T @ patient_matrix
# 
# 

# ### 3.3 Overview of the Datasets
# 
# This section presents an overview of a Dash web application designed to visualize co-occurrences in FHIR (Fast Healthcare Interoperability Resources) codes. The following parts detail the data flow and the interrelationships among various components and functions within the application.
# 
# #### 3.3.1 Data Upload and Processing
# 
# The application begins by allowing users to upload a dataset through the `dcc.Upload` component. The uploaded data is then processed within the `upload_file` callback, which reads the file's content and sends it to the `fetch_and_process_data` function.
# 
# - **`fetch_and_process_data`**: This function processes the uploaded data and generates a structured output. It adds two key columns, `Displays` and `Full_Displays`, to the `flat_df`. The `Displays` column provides a truncated version of the information in the `Full_Displays` column, which contains complete and detailed descriptions. This distinction enhances data presentation while still granting access to comprehensive information when required. The `fetch_and_process_data` function outputs three primary data structures:
# 
#   - **`flat_df`**: This DataFrame contains three essential columns: `PatientID`, `Codes`, and `ResourceType`. The `PatientID` column includes anonymized integer identifiers starting from 1. The `Codes` column comprises various healthcare-related codes (e.g., ICD, LOINC, or OPS), while the `ResourceType` column categorizes each code's specific type (e.g., ICD for diagnoses, LOINC for lab tests, OPS for procedures).
# 
#     | PatientID | Codes    | ResourceType |
#     |-----------|----------|--------------|
#     | 1         | M87.24   | ICD          |
#     | 1         | I41.1    | ICD          |
#     | 1         | 9-694.t  | OPS          |
#     | 1         | 8-826.0h | OPS          |
#     | 1         | 6-008.gs | OPS          |
#     | 1         | 5-812.n0 | OPS          |
#     | 1         | 2951-2   | LOINC        |
#     | 1         | 26450-7  | LOINC        |
#     | 1         | 5894-1   | LOINC        |
#     | 1         | 1988-5   | LOINC        |
#     | 2         | U35.1    | ICD          |
#     | 2         | T53.4    | ICD          |
# 
#     The labels added to the uploaded `flat_df` are sourced from FHIR catalogs that encompass various coding systems, including ICD, OPS, and LOINC. The following code snippets illustrate how these datasets are loaded:
# 
#     ```python
#     icd_df = pd.read_csv('ICD_Katalog_2023_DWH_export_202406071440.csv') 
#     ops_df = pd.read_csv('OPS_Katalog_2023_DWH_export_202409200944.csv')
#     loinc_df = pd.read_csv('LOINC_DWH_export_202409230739.csv')  
#     ```
# 
#     Additionally, the function `get_display_label` is employed to retrieve the appropriate display labels based on the provided code, level, and resource type.
# 
#     | PatientID | Codes    | ResourceType | Displays                 | Full_Displays                                                    |
#     |-----------|----------|--------------|--------------------------|-----------------------------------------------------------------|
#     | 1         | M87.24   | ICD          | Knochennekrose           | Knochennekrose durch vorangegangenes Trauma: ...                |
#     | 1         | I41.1    | ICD          | Myokarditis              | Myokarditis bei anderenorts klassifizierten Viruserkrankungen ...|
#     | 1         | 9-694.t  | OPS          | Spezifische Behandlung    | Spezifische Behandlung im besonderen Setting bei ...            |
#     | 1         | 8-826.0h | OPS          | Doppelfiltrationsplasmapherese | Doppelfiltrationsplasmapherese (DFPP): Ohne K ...             |
#     | 1         | 6-008.gs | OPS          | Applikation von Medikamenten | Applikation von Medikamenten, Liste 8: Isavuconazol ...        |
#     | 1         | 5-812.n0 | OPS          | Arthroskopie             | Arthroskopische Operation am Gelenkknorpel ...                 |
#     | 1         | 2951-2   | LOINC        | Sodium [Moles/volume]    | Sodium [Moles/volume] in Serum or Plasma                        |
#     | 1         | 26450-7  | LOINC        | Eosinophils              | Eosinophils/100 leukocytes in Blood                             |
#     | 1         | 5894-1   | LOINC        | Prothrombin time         | Prothrombin time (PT) actual/normal in Platelets ...            |
#     | 1         | 1988-5   | LOINC        | C reactive protein       | C reactive protein [Mass/volume] in Serum or Plasma            |
#     | 2         | U35.1    | ICD          | Nicht belegte Schlüsselnummer | Nicht belegte Schlüsselnummer U35.1                           |
#     | 2         | T53.4    | ICD          | Toxische Wirkung         | Toxische Wirkung: Dichlormethan                                 |
# 
#   - **`co_occurrences_df`**: This DataFrame illustrates co-occurrences of codes among patients and provides insights into relationships between various codes.
# 
# - **`co_occurrence_matrix_df`**: This DataFrame illustrates co-occurrences of codes among patients, indicating the number of patients sharing pairs of codes.
# 
# |         | M87.24 | I41.1 | U35.1 | T53.4 | 9-694.t | 8-826.0h | 6-008.gs | 5-812.n0 | 2951-2 | 26450-7 | 5894-1 | 1988-5 |
# |---------|--------|-------|-------|-------|---------|----------|----------|----------|--------|---------|--------|--------|
# | **M87.24** | 1      | 0     | 0     | 0     | 0       | 0        | 0        | 0        | 0      | 0       | 0      | 0      |
# | **I41.1**  | 0      | 1     | 0     | 0     | 0       | 0        | 0        | 0        | 0      | 0       | 0      | 0      |
# | **U35.1**  | 0      | 0     | 1     | 0     | 0       | 0        | 0        | 0        | 0      | 0       | 0      | 0      |
# | **T53.4**  | 0      | 0     | 0     | 1     | 0       | 0        | 0        | 0        | 0      | 0       | 0      | 0      |
# | **9-694.t**| 0      | 0     | 0     | 0     | 1       | 1        | 1        | 1        | 0      | 0       | 0      | 0      |
# | **8-826.0h**| 0      | 0     | 0     | 0     | 1       | 1        | 1        | 1        | 0      | 0       | 0      | 0      |
# | **6-008.gs**| 0      | 0     | 0     | 0     | 1       | 1        | 1        | 1        | 0      | 0       | 0      | 0      |
# | **5-812.n0**| 0      | 0     | 0     | 0     | 1       | 1        | 1        | 1        | 0      | 0       | 0      | 0      |
# | **2951-2** | 0      | 0     | 0     | 0     | 0       | 0        | 0        | 0        | 1      | 1       | 1      | 1      |
# | **26450-7**| 0      | 0     | 0     | 0     | 0       | 0        | 0        | 0        | 1      | 1       | 1      | 1      |
# | **5894-1** | 0      | 0     | 0     | 0     | 0       | 0        | 0        | 0        | 1      | 1       | 1      | 1      |
# | **1988-5** | 0      | 0     | 0     | 0     | 0       | 0        | 0        | 0        | 1      | 1       | 1      | 1      |
# 
# 
# ## 4 Limitations and Future Improvements
# 
# Despite its strengths, the CoCo application has limitations:
# 
# 1. **Performance Issues**: With larger dataset sizes, the user experience may be impacted. Significant time is required for data uploads and visualization rendering.
# 2. **Data Imbalance**: A noted imbalance in the prevalence of LOINC codes compared to ICD and OPS codes could skew insights derived from the visualizations.
# 
#      
#    <!-- ![Node Degree Distribution](Overleaf-images/Node_degree_distribution.png) -->
#    
#    <img src="Overleaf-images/Node_degree_distribution.png" alt="Node Degree Distribution" width="700" />
# 
#    *Figure: Node Degree Distribution*
# 
# 3. **Missing Codes**: Some codes present in real FHIR datasets may not exist in catalogs, affecting the visualization and label display.
# 
# Future improvements could focus on enhancing the performance of the application, expanding the database of available codes, and incorporating additional data sources to enrich the analysis.
# 
# ## 5 Conclusion
# 
# The CoCo software represents a significant advancement in analyzing co-occurrences among clinical codes within FHIR datasets. By providing flexible and interactive visualizations, it enhances the understanding of complex clinical relationships, thus aiding healthcare professionals in decision-making and patient care.
# 

# # LIBRARY

# In[1]:


import os
import io
import base64
import tempfile
import re
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from itertools import combinations

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

from pyvis.network import Network
import plotly.figure_factory as ff
import plotly.graph_objs as go

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering



# # MAIN CODE

# ### Colors of Resource Types
# The `SUBGROUP_COLORS` dictionary assigns specific colors to different resource types: **ICD** is represented in **light blue**, **LOINC** in **pink**, and **OPS** in **purple**.
# 

# In[2]:


SUBGROUP_COLORS = {
    'ICD': "#00bfff", #"#00bfff",
    'LOINC': "#ffc0cb", #"#ffc0cb",
    'OPS': "#9a31a8" ##9a31a8"
}


def get_color_for_resource_type(resource_type):
    """Map resource types to colors using SUBGROUP_COLORS."""
    return SUBGROUP_COLORS.get(resource_type, 'gray')  # Default to gray if not found


# ### Assigning Resource Types depending on their encoding properties
# This code segment provides functions to validate and classify medical codes for ICD, LOINC, and OPS, ensuring that only properly formatted codes are processed within the application.
# 

# In[3]:


def is_icd_code(code):
    """Check if the given code is a valid ICD code."""
    return bool(re.match(r"^[A-Z]", code))

def is_loinc_code(code):
    """Check if the given code is a valid LOINC code with a hyphen at [-2]."""
    return len(code) > 1 and code[-2] == '-'

def is_ops_code(code):
    """Check if the given code is a valid OPS code."""
    return len(code) > 1 and code[1] == '-'

# Function to classify ResourceType based on the code
def get_resource_type(code):
    if re.match(r"^[A-Z]", code):
        return "ICD"
    elif len(code) > 1 and code[-2] == '-':
        return "LOINC"
    elif len(code) > 1 and code[1] == '-':
        return "OPS"
    else:
        return "Unknown"


# ### Dash Application Setup Components
# 
# 
# - **dcc.Upload(...)**: This component allows users to upload FHIR data. 
# 
# - **dcc.Slider(...)**: Implements a slider for user input for adjusting parameters like the number of nodes to visualize or the hierarchy level.
# 
# - **dcc.Dropdown(...)**: Creates a dropdown menu that allows users to select a code from a list. 
# 
# - **dcc.Checklist(...)**: If checked, labels are shown.
# 
# - **dcc.Loading(...)**: Displays a loading spinner or indicator while data is being processed or loaded.
# 
# - **dcc.Graph(...)**: Pyvis, bar chartand dendogram are plotted.
# 

# In[4]:


# Dash application setup
app = dash.Dash(__name__)
server = app.server


app.layout = html.Div([
    html.H1("CoCo: Co-Occurrences in FHIR Codes"),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload Data'),
        multiple=False
    ),
    html.Div(id='upload-feedback', children='', style={'color': 'red'}),

    
    # Slider for the number of top neighbor nodes
    html.Div(id='slider-container', children=[
        html.Label("Select the number of nodes to visualize:"),
        dcc.Slider(
            id='num-nodes-slider',
            min=1,
            max=10,
            step=1,
            value=1,
            marks={i: str(i) for i in range(1, 11)},
            tooltip={"placement": "bottom", "always_visible": False}
        )
    ], style={'display': 'none'}),  # Initially hidden
    
    # Slider for the hierarchy levels
    html.Div(id='level-slider-container', children=[
        html.Label("Select the hierarchy level:"),
        dcc.Slider(
            id='level-slider',
            min=1,
            max=4,
            step=1,
            value=1,
            marks={i: str(i) for i in range(5)},  # 0 to 4
            tooltip={"placement": "bottom", "always_visible": False}
        )
    ], style={'display': 'none'}),  # Initially hidden 
    
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
            
        ]
    ),

    # Graphs positioned side-by-side using CSS flexbox
    html.Div([
        # Left column - PyVis graph
        html.Div([
            html.Label("Enter code to search:"),
            dcc.Input(id='code-input', type='text', placeholder='Enter code', debounce=False),  # Debounce=False to update as you type
            html.Iframe(id='graph-iframe', style={'width': '100%', 'height': '100%'}),
        ], style={'flex': '1', 'padding': '10px'}),  # PyVis on the left side, takes 50% of space

        # Right column - Bar chart and dendrogram stacked
        html.Div([
            dcc.Store(id='codes-of-interest-store'),
            dcc.Graph(id='dendrogram', style={ 'width': '100%', 'height': '50%'}),  # Dendrogram below bar chart
            dcc.Graph(id='bar-chart', style={'width': '100%', 'height': '50%'})  # Bar chart on top , 'margin-bottom': '20px'
        ], style={'flex': '1', 'padding': '0px'}),  # Bar chart and dendrogram on the right side, takes 50% of space

    ], style={'display': 'flex', 'flex-direction': 'row'}),  # Use flexbox to position the graphs side by side
    
    dcc.Store(id='data-store')  # Hidden store to keep data
])


# ### Co-Occurrence Matrix Creation
# 
# The `create_co_occurrence_matrix` function generates a co-occurrence matrix from a dataset of patient codes, summarizing how different codes appear together among patients.
# 
# ### Example Dataset
# | PatientID | Codes  |
# |-----------|--------|
# | P1        | A      |
# | P1        | B      |
# | P2        | A      |
# | P2        | C      |
# | P3        | B      |
# | P3        | C      |
# | P3        | D      |
# 
# **Create Patient-Codes Matrix**: Constructs a pivot table that counts the occurrences of codes for each patient.
# #### Patient-Codes Matrix
# | PatientID | A | B | C | D |
# |-----------|---|---|---|---|
# | P1        | 1 | 1 | 0 | 0 |
# | P2        | 1 | 0 | 1 | 0 |
# | P3        | 0 | 1 | 1 | 1 |
# 
# **Calculate Co-Occurrence Matrix**: Uses a dot product of the transposed patient matrix to create a co-occurrence matrix representing how often each pair of codes appears together.
# **Return the Matrix**: Provides the resulting co-occurrence matrix for further analysis.
# 
# #### Co-Occurrence Matrix Example
# |     | A | B | C | D |
# |-----|---|---|---|---|
# | A   | 0 | 2 | 1 | 0 |
# | B   | 2 | 0 | 2 | 1 |
# | C   | 1 | 2 | 0 | 1 |
# | D   | 0 | 1 | 1 | 0 |
# 
# ### Summary
# The co-occurrence matrix reveals relationships between different codes based on their simultaneous occurrences across patients, facilitating insights into associations among conditions or procedures.
# 

# In[5]:


# Create co-occurrence matrices
def create_co_occurrence_matrix(df):
    if df.empty:
        return pd.DataFrame()
    patient_matrix = df.pivot_table(index='PatientID', columns='Codes', aggfunc='size', fill_value=0)
    patient_matrix = patient_matrix.loc[:, (patient_matrix != 0).any(axis=0)]
    co_occurrence_matrix = patient_matrix.T.dot(patient_matrix)
    np.fill_diagonal(co_occurrence_matrix.values, 0)
    return co_occurrence_matrix


# ### Rescale Node Size and Edge Thickness
# The `normalize_weights` function rescales a node size and edge thickness depending on the maximum and minimum weights in an uploaded dataset.
#     

# In[6]:


def normalize_weights(value, gain=1, offset=0, min_limit=None, max_limit=None):
    # Normalize the value
    normalized_value = (value * gain) + offset
    
    if min_limit is not None:
        normalized_value = max(min_limit, normalized_value)
    if max_limit is not None:
        normalized_value = min(max_limit, normalized_value)

    return normalized_value


# ### Pyvis Graph if ALL_Codes is selected
# The `generate_network_viz` function creates a network visualization using the PyVis library, transforming a pandas DataFrame into a NetworkX graph, applying various visual attributes such as node colors, edge colors, and thickness, while also allowing for specific layouts like circular arrangements for nodes at a selected level, and incorporating physics parameters to enhance the visual representation of relationships in the data.
# 
# #### net.toggle_physics(True)
# A physics control button will appear in the visualization, allowing users to dynamically adjust the following parameters:
# 
# - **central_gravity**: Influences the strength of the attractive force toward the center of the network.
# - **node_distance**: Sets the minimum distance between nodes in the graph.
# - **spring_length**: Determines the resting length of the springs connecting the nodes.
# - **spring_constant**: Affects the stiffness of the springs; a higher value results in a stronger pull between nodes.
# - **spring_strength**: Governs how strongly nodes are attracted to each other.
# - **damping**: Controls the damping effect, which reduces oscillations in node movement over time.
# - **min_velocity**: Sets the minimum speed for the nodes in the simulation, ensuring they do not come to a complete stop. 
# 
# #### net.toggle_physics(False)
# The `net.toggle_physics(False)` line, when uncommented, disables the physics simulation in the PyVis network visualization, allowing for a static layout of the nodes and edges, which can help maintain the arrangement of the graph when presenting the network without dynamic movement.
# 
# 

# In[7]:


def generate_network_viz(df, code1_col, code2_col, weight_col, 
                         layout='barnes_hut', selected_level=None,  
                         node_color=None, edge_color=None,
                         edge_thickness_min=1, edge_thickness_max=10,
                         central_gravity=0,
                         node_distance=200,
                         spring_length=0,
                         spring_constant=0.3,
                         spring_strength=0.005,
                         damping=0.5,
                        min_velocity=0.75):
    # Generate a NetworkX graph
    G = nx.from_pandas_edgelist(df, source=code1_col, target=code2_col, edge_attr=weight_col)

    bgcolor, font_color = 'white', 'gray'  # Default colors

    # Initiate PyVis network object
    net = Network(
        height='700px', 
        width='100%',
        bgcolor=bgcolor, 
        font_color=font_color, 
        notebook=True
    )

    # Take NetworkX graph and translate it to a PyVis graph format
    net.from_nx(G)

    # Set colors for nodes and sizes
    if node_color is not None:
        for node in G.nodes():
            net.get_node(node)['color'] = node_color.get(node, 'gray')  # Default to gray if no color is provided

    # Set colors and thickness for edges
    if edge_color is not None:
        for u, v in G.edges():
            net.get_edge(u, v)['color'] = edge_color.get((u, v), 'rgba(255, 255, 255, 0.3)')  # Default to white with transparency
            thickness = G.edges[u, v].get(weight_col, 1)  # Default thickness if not set
            thickness = normalize_weights(thickness, gain=5, offset=1, 
                                          edge_thickness_min=edge_thickness_min, 
                                          edge_thickness_max=edge_thickness_max)
            net.get_edge(u, v)['width'] = thickness

    # Apply circular layout only for nodes at the selected level
    if selected_level is not None:
        level_nodes = df[df['level'] == selected_level][code1_col].unique()
        
        # Compute circular layout for all nodes in G
        pos = nx.circular_layout(G)  # Removed the `nodes` argument

        # Setting positions for the network for the selected level nodes
        for node in level_nodes:
            if node in pos:  # Check if node is in the computed positions
                net.get_node(node)['x'] = pos[node][0] * 300  # Scale position for visualization
                net.get_node(node)['y'] = pos[node][1] * 300  # Scale position for visualization
                
    #net.toggle_physics(False)
    return net


# ### Dendogram  if an individual code is selected
# The advantage of the dendrogram plot is that it visually represents multiple nodes (more than two) that frequently co-occur together, allowing for easy identification of complex relationships within the dataset.
# 

# In[8]:


def create_dendrogram_plot(cooccurrence_array, labels, flat_df, show_labels):

    if 'show' in show_labels:
        # Use 'Displays' from flat_df for labels
        labels = [
            flat_df.loc[flat_df['Codes'] == label, 'Displays'].iloc[0] 
            if not flat_df.loc[flat_df['Codes'] == label, 'Displays'].empty 
            else label  # Fallback to code if display is missing
            for label in labels
        ]

    # Create the dendrogram plot with Plotly
    fig = ff.create_dendrogram(cooccurrence_array, orientation='bottom', labels=labels)

        # Update line color for all links in the dendrogram
    for line in fig.data:
        line.update(line=dict(color='gray'))  # Set your desired color here
    
    # Update layout to improve appearance
    fig.update_layout(
        title='Dendrogram',
        title_x=0.5,
        xaxis_title='',
        yaxis_title='Distance',
        xaxis={'tickangle': -45},  # Rotate labels for better readability
    )
    
    return fig


# ### CoCo_Input folder will be created on the user desktop for the catalogues
# The `create_dataset_directory` function creates a folder on the user's desktop called CoCo_Input, and within that folder, there should be the three catalogues of ICD, OPS, and LOINC.
# 

# In[9]:


def create_dataset_directory(directory_name):
    # Get the user's home directory
    home_directory = os.path.expanduser("~")
    
    # Construct the path to the desktop directory
    desktop_directory = os.path.join(home_directory, "Desktop")
    
    # Create the full path for the new directory
    datasets_dir = os.path.join(desktop_directory, directory_name)

    try:
        # Create the directory (if it doesn't already exist)
        os.makedirs(datasets_dir, exist_ok=True)
        print(f"Directory created at: {datasets_dir}")
        return datasets_dir
    except Exception as e:
        print(f"Error creating directory: {str(e)}")
        return None


# ###
# ### Dash callback: `update_slider_visibility`
# The `update_slider_visibility` function dynamically adjusts the visibility of sliders based on the user's selection in the code dropdown; if "ALL_CODES" is selected, it displays a slider for hierarchical levels (1 to 4), while for individual codes, it shows a slider for the top neighbors with which they most frequently co-occur.
# 

# In[10]:


@app.callback(
    Output('slider-container', 'style'),
    Output('level-slider-container', 'style'),
    Input('code-dropdown', 'value')
)
def update_slider_visibility(selected_code):
    if selected_code == 'ALL_CODES':
        return {'display': 'none'}, {'display': 'block'}  # Show level slider, hide num-nodes
    else:
        return {'display': 'block'}, {'display': 'none'}  # Show num-nodes slider, hide level slider


# ### Preparing the data
# It performs data loading, validation, analysis, and structuring, ultimately preparing the data for further processing
# 
# 1. **Read Data**: Load CSV data from the uploaded Parquet file into a DataFrame (`flat_df`).
# 
# 2. **Check Required Columns**: Verify that the necessary columns (`PatientID`, `Codes`, `ResourceType`) exist in `flat_df`.
# 
# 3. **Create Dataset Directory**: Define a directory name ("CoCo_Input") and create the corresponding directory on the user's desktop.
# 
# 4. **Load External Datasets**: Load three catalogues (ICD, OPS, LOINC) from the created directory.
# 
# 5. **Process DataFrames**: Extract and prepare DataFrames for each resource type (ICD, OPS, LOINC).
# 
# 6. **Generate Co-occurrence Matrix**: Create a co-occurrence matrix from the `flat_df` using a helper function `create_co_occurrence_matrix`.
# 
# 7. **Create Code Pairs**: Iterate through the co-occurrence matrix to create pairs of codes with their weights and add these to a DataFrame (`pairs_df`).
# 
# 8. **Build Hierarchies**: Construct hierarchical structures for each code set (ICD, OPS, LOINC) using a helper function (`build_hierarchy_and_get_pairs`).
# 
# 9. **Add parents from the catalogues**: Append new rows for the hierarchy levels (0, 1, 2, 3) coming from the cataglogues(`new_rows`).
# 
# 10. **Combine DataFrames**: Convert `new_rows` into a DataFrame (`new_entries_df`) and concatenate it with `pairs_df` to form `new_pairs_df`.
# 
# 11. **Add Displays (labels)**: Fill in the `Displays` column in `flat_df` based on the resource type and other conditions.
# 
# 12. **Segment DataFrames**: Split `flat_df` into separate DataFrames for each resource type (ICD, LOINC, OPS).
# 
# 13. **Generate Co-occurrence Matrices for Resource Types**: Create co-occurrence matrices for each specific resource type.
# 
# 14. **Return Results**: Return a dictionary containing `flat_df`, co-occurrence matrices, and `new_pairs_df`.
# 

# In[11]:


def fetch_and_process_data(file_content):
    
    
# 1. Read CSV data from uploaded content
    flat_df = pd.read_parquet(io.BytesIO(file_content))
    print('flat_df:', flat_df)

##################################################################################################    
# 2. Check for required columns
    required_columns = ['PatientID', 'Codes', 'ResourceType']
    missing_columns = [col for col in required_columns if col not in flat_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

################################################################################################## 
# 3. Create Dataset Directory
    new_directory_name = "CoCo_Input"
    datasets_dir = create_dataset_directory(new_directory_name)

    # Initialize a dictionary to hold the dataframes
    dataframes = {}
    file_names = {
        'ICD': 'ICD_Katalog_2023_DWH_export_202406071440.csv',
        'OPS': 'OPS_Katalog_2023_DWH_export_202409200944.csv',
        'LOINC': 'LOINC_DWH_export_202409230739.csv'
    }

##################################################################################################
# 4. Load External Datasets
    for key, filename in file_names.items():
        file_path = os.path.join(datasets_dir, filename)
        try:
            dataframes[key] = pd.read_csv(file_path)
        except FileNotFoundError:
            return {
                'success': False,
                'message': f"Put the catalogue files into the directory: {datasets_dir}"
            }

    # Continue processing with icd_df, ops_df, and loinc_df if needed
    icd_df = dataframes.get('ICD')
    ops_df = dataframes.get('OPS')
    loinc_df = dataframes.get('LOINC')

##################################################################################################
# 5. Process DataFrames:

    def get_display_label(code, level,  resource_type):
        """Retrieve the display label for codes and their associated group or chapter labels based on resource type."""
        code = str(code).strip()
        # get the main display label based on the specific code
        if resource_type == 'ICD':
           #print('ICD code', code)
            if level == 4:
                result = icd_df.loc[icd_df['ICD_CODE'] == code, 'ICD_NAME']
                if not result.empty:
                    return result.iloc[0]  # Return the first result if found
            if level == 3:
                # get group or chapter label
                gruppe_result = icd_df.loc[icd_df['GRUPPE_CODE'] == code, 'GRUPPE_NURNAME']
               #print('GRUPPE', code, gruppe_result.iloc[0])
                if not gruppe_result.empty:
                    return gruppe_result.iloc[0]  # Return the first result
                    
            if level == 2:
                icd_df['KAPITEL_CODE'] = icd_df['KAPITEL_CODE'].astype(str)  # Convert KAPITEL_CODE to string
                code = str(code).strip()

                # get group or chapter label for level 2
                kapitel_result = icd_df.loc[icd_df['KAPITEL_CODE'] == code, 'KAPITEL_NURNAME']

                if not kapitel_result.empty:
                   #print(f"Level 2 display found: {kapitel_result.iloc[0]}")
                    return kapitel_result.iloc[0]  # Return the first result


        elif resource_type == 'OPS':
           #print('OPS code', code)
            if level == 4:
                result = ops_df.loc[ops_df['OPS_CODE'] == code, 'OPS_NAME']
                if not result.empty:
                    return result.iloc[0]  # Return the first result if found
                
            if level == 3:
                # get group or chapter label
                gruppe_result = ops_df.loc[ops_df['GRUPPE_CODE'] == code, 'GRUPPE_NURNAME']
               #print('GRUPPE', code, gruppe_result.iloc[0])
                if not gruppe_result.empty:
                    return gruppe_result.iloc[0]  # Return the first result
                
            if level == 2:
                icd_df['KAPITEL_CODE'] = icd_df['KAPITEL_CODE'].astype(str)  # Convert KAPITEL_CODE to string
                code = str(code).strip()
                kapitel_result = ops_df.loc[ops_df['KAPITEL_CODE'] == code, 'KAPITEL_NURNAME']
                if not kapitel_result.empty:
                    return kapitel_result.iloc[0]  # Return the first result

        elif resource_type == 'LOINC':
           #print('LOINC code', code)
            if level == 4:            
                result = loinc_df.loc[loinc_df['LOINC_CODE'] == code, 'LOINC_NAME']
               #print('LOINC result',result)
                if not result.empty:
                    return result.iloc[0]  # Return the first result if found
                
            if level == 3:
                # get group or chapter label
                gruppe_result = loinc_df.loc[loinc_df['LOINC_PROPERTY'] == code, 'LOINC_PROPERTY']
                if not gruppe_result.empty:
                    return gruppe_result.iloc[0]  # Return the first result
                
            if level == 2:
                kapitel_result = loinc_df.loc[loinc_df['LOINC_SYSTEM'] == code, 'LOINC_SYSTEM']
                if not kapitel_result.empty:
                    return kapitel_result.iloc[0]  # Return the first result

        return None  

    
##################################################################################################  
# 6. Generate Co-occurrence Matrix:
    
    main_df = create_co_occurrence_matrix(flat_df)
    print('main_df', main_df)
    
##################################################################################################    
# 7. Create Code Pairs:
    code_pairs = []

    # Iterate through main_df to create initial pairs
    for i in range(len(main_df)):
        for j in range(i + 1, len(main_df)):
            code1 = main_df.index[i]
            code2 = main_df.columns[j]
            weight = main_df.iloc[i, j]

            if weight > 0:
                code_pairs.append((code1, code2, weight))

    pairs_df = pd.DataFrame(code_pairs, columns=['Code1', 'Code2', 'Weight'])
    pairs_df['level'] = 4    
    print('pairs_df', pairs_df)
    
##################################################################################################    
# 8. Build Hierarchies:    
    def build_hierarchy_and_get_pairs(df, code_column, kapitel_column, gruppe_column):
        if df is None:
            return []

        df = df[df[code_column].isin(flat_df['Codes'])]
        df_subset = df[[kapitel_column, gruppe_column, code_column]]  # Select by column names
        level_0 = []

        for index, row in df_subset.iterrows():
            level_2 = str(row[kapitel_column])
            #print('level_2', level_2)
            level_3 = f"{level_2},{str(row[gruppe_column])}"  # Make level unique
            #print('level_3', level_3)
            level_4 = f"{level_3},{str(row[code_column])}"
            #print('level_4', level_4)      

            resource_type1 = get_resource_type(row[code_column])  # Custom function to get resource type

            if resource_type1 == 'ICD':
                level_1 = f"{'ICD'}, {level_4}"
                level_0.append((f"{'FHIR'}, {level_1}"))
                #print('level_0', level_0)

            if resource_type1 == 'OPS':
                level_1 = f"{'OPS'}, {level_4}"
                level_0.append((f"{'FHIR'}, {level_1}"))
                #print('level_0', level_0)

            if resource_type1 == 'LOINC':
                level_1 = f"{'LOINC'}, {level_4}"
                level_0.append((f"{'FHIR'}, {level_1}"))
                #print('level_0', level_0)

        return level_0
    
##################################################################################################    
# 9. Add parents from the catalogues: 

    # Get node structure for each DataFrame
    icd_level_0 = build_hierarchy_and_get_pairs(icd_df, 'ICD_CODE', 'KAPITEL_CODE', 'GRUPPE_CODE')
    ops_level_0 = build_hierarchy_and_get_pairs(ops_df, 'OPS_CODE', 'KAPITEL_CODE', 'GRUPPE_CODE')  # Adjust column names if necessary
    loinc_level_0 = build_hierarchy_and_get_pairs(loinc_df, 'LOINC_CODE', 'LOINC_SYSTEM', 'LOINC_PROPERTY')  # Adjust column names if necessary

    new_rows = []

    # level 0
    new_rows.append({'Code1':'FHIR' , 'Code2':'ICD' , 'Weight': len(icd_level_0), 'level': 0, 'ResourceType':'ICD'})
    new_rows.append({'Code1':'FHIR' , 'Code2':'OPS' , 'Weight': len(ops_level_0), 'level': 0, 'ResourceType':'OPS'})
    new_rows.append({'Code1':'FHIR' , 'Code2':'LOINC' , 'Weight': len(loinc_level_0), 'level': 0, 'ResourceType':'LOINC'})

    # Level 1 - Split the 3rd item (index 2) in icd_level_0
    icd_items = [item.split(',')[2] for item in icd_level_0]
    icd_item_counts = Counter(icd_items)

    # Iterate over each unique ICD level 1 item and its count
    for item, count in icd_item_counts.items():
        # Add a row for each level 1 ICD item
        new_rows.append({'Code1': 'ICD', 'Code2': 'icd'+item, 'Weight': count, 'level': 1, 'ResourceType':'ICD',
                        'Displays': 'ICD'})

        # Level 2 - Split the 4th item (index 3) for level 1 connections
        icd_items1 = [lvl_0_item.split(',')[3] for lvl_0_item in icd_level_0 if lvl_0_item.split(',')[2] == item]
        icd_item_counts1 = Counter(icd_items1)

        for item1, count1 in icd_item_counts1.items():
            new_rows.append({
                            'Code1': 'icd' + item, 
                            'Code2': item1,          
                            'Weight': count1,        
                            'level': 2,              
                            'ResourceType': 'ICD',   
                            'Displays': get_display_label(item, 2, 'ICD') 
                        })
           #print(item, 2, 'ICD')

            # Level 3 - Split the 5th item (index 4) for level 2 connections
            icd_items2 = [lvl_0_item.split(',')[4] for lvl_0_item in icd_level_0 if lvl_0_item.split(',')[3] == item1]
            icd_item_counts2 = Counter(icd_items2)

            for item2, count2 in icd_item_counts2.items():
                new_rows.append({
                            'Code1': item1, 
                            'Code2': item2,          
                            'Weight': count2,        
                            'level': 3,              
                            'ResourceType': 'ICD',   
                            'Displays': get_display_label(item1, 3, 'ICD')  
                        })
           #print(item1, 3, 'ICD')

    # OPS Level 1 - Split the 3rd item (index 2) in ops_level_0
    ops_items = [item.split(',')[2] for item in ops_level_0]
    ops_item_counts = Counter(ops_items)

    # Iterate over each unique OPS level 1 item and its count
    for item, count in ops_item_counts.items():
        # Add a row for each level 1 OPS item
        new_rows.append({'Code1': 'OPS', 'Code2': 'ops'+item, 'Weight': count, 'level': 1, 'ResourceType':'OPS',
                        'Displays': 'OPS'})

        # OPS Level 2 - Split the 4th item (index 3) for level 1 connections
        ops_items1 = [lvl_0_item.split(',')[3] for lvl_0_item in ops_level_0 if lvl_0_item.split(',')[2] == item]
        ops_item_counts1 = Counter(ops_items1)

        for item1, count1 in ops_item_counts1.items():
            new_rows.append({
                            'Code1': 'ops' + item,  # Ensure the code is prefixed with 'icd'
                            'Code2': item1,          # Level 2 ICD code
                            'Weight': count1,        # Count for this item
                            'level': 2,              # Specify level
                            'ResourceType': 'OPS',   # Set resource type
                            'Displays': get_display_label(item, 2, 'OPS')  # Fetch display label or group name
                        })

            # OPS Level 3 - Split the 5th item (index 4) for level 2 connections
            ops_items2 = [lvl_0_item.split(',')[4] for lvl_0_item in ops_level_0 if lvl_0_item.split(',')[3] == item1]
            ops_item_counts2 = Counter(ops_items2)

            for item2, count2 in ops_item_counts2.items():
                new_rows.append({
                            'Code1': item1,  # Ensure the code is prefixed with 'icd'
                            'Code2': item2,          # Level 2 ICD code
                            'Weight': count2,        # Count for this item
                            'level': 3,              # Specify level
                            'ResourceType': 'OPS',   # Set resource type
                            'Displays': get_display_label(item1, 3, 'OPS')  # Fetch display label or group name
                        })

    # LOINC Level 1 - Split the 3rd item (index 2) in loinc_level_0
    loinc_items = [item.split(',')[2] for item in loinc_level_0]
   #print('loinc_items', loinc_items)
    loinc_item_counts = Counter(loinc_items)

    # Iterate over each unique LOINC level 1 item and its count
    for item, count in loinc_item_counts.items():
        # Add a row for each level 1 LOINC item
        new_rows.append({'Code1': 'LOINC', 'Code2': item, 'Weight': count, 'level': 1, 'ResourceType':'LOINC',
                        'Displays': 'LOINC'})

        # LOINC Level 2 - Split the 4th item (index 3) for level 1 connections
        loinc_items1 = [lvl_0_item.split(',')[3] for lvl_0_item in loinc_level_0 if lvl_0_item.split(',')[2] == item]
        loinc_item_counts1 = Counter(loinc_items1)

        for item1, count1 in loinc_item_counts1.items():
            # Add a row for each level 2 LOINC item
            new_rows.append({'Code1': item, 'Code2': item1, 'Weight': count1, 'level': 2, 'ResourceType':'LOINC',
                            'Displays':get_display_label(item, 2, 'LOINC') })

            # LOINC Level 3 - Split the 5th item (index 4) for level 2 connections
            loinc_items2 = [lvl_0_item.split(',')[4] for lvl_0_item in loinc_level_0 if lvl_0_item.split(',')[3] == item1]

            loinc_item_counts2 = Counter(loinc_items2)

            for item2, count2 in loinc_item_counts2.items():
                # Add a row for each level 3 LOINC item
                new_rows.append({'Code1': item1, 'Code2': item2, 'Weight': count2, 'level': 3, 'ResourceType':'LOINC',
                                'Displays':get_display_label(item1, 3, 'LOINC')})


##################################################################################################
# 10. Combine DataFrames: 

    new_entries_df = pd.DataFrame(new_rows)    
    new_pairs_df = pd.concat([pairs_df, new_entries_df], ignore_index=True)
    new_pairs_df = new_pairs_df.drop_duplicates(subset=['Code1', 'Code2', 'Weight','level'])
    print('new_pairs_df', new_pairs_df)
#     new_pairs_df.to_csv('new_pairs_df.csv', index=False)

##################################################################################################
# 10. Combine DataFrames: 

    # Fill the Displays column
    flat_df['Displays'] = flat_df.apply(
        lambda row: get_display_label(row['Codes'], 4, row['ResourceType']),
        axis=1
    )


##################################################################################################    
# 11.Add Displays (labels):

    # Create 'Full_Displays' column based on 'Codes' and 'Displays' for LOINC resource type
    flat_df['Full_Displays'] = flat_df.apply(
        lambda row: f"{row['Codes']}: {row['Displays']}" if row['ResourceType'] == 'LOINC' else row['Displays'],
        axis=1
    )

    flat_df.loc[flat_df['ResourceType'].isin(['ICD', 'OPS']), 'Displays'] = \
        flat_df.loc[flat_df['ResourceType'].isin(['ICD', 'OPS']), 'Displays'].apply(
            lambda x: ': '.join(x.split(':')[1:]).strip() if isinstance(x, str) else x
        )

    flat_df.loc[flat_df['ResourceType'] == 'LOINC', 'Displays'] = flat_df.loc[flat_df['ResourceType'] == 'LOINC', 'Displays']
   
    
    flat_df['Displays'] = flat_df['Displays'].astype(str)
    
    flat_df['Displays'] = flat_df['Displays'].str.slice(0, 11) + '...'
   #print('flat_df', flat_df)

##################################################################################################
# 12. Segment DataFrames

    ICD_df = flat_df[flat_df['ResourceType'] == 'ICD']
    LOINC_df = flat_df[flat_df['ResourceType'] == 'LOINC']
    OPS_df = flat_df[flat_df['ResourceType'] == 'OPS']
    
##################################################################################################    
# 13. Generate Co-occurrence Matrices for Resource Types:
    co_occurrence_matrices = {
        'Main': create_co_occurrence_matrix(flat_df),
        'ICD': create_co_occurrence_matrix(ICD_df),
        'LOINC': create_co_occurrence_matrix(LOINC_df),
        'OPS': create_co_occurrence_matrix(OPS_df)
    }

##################################################################################################
# 14. Return Results:  
    return {
        'success': True,
        'message': 'Data is loaded.',
        'data': {
            'flat_df': flat_df.to_dict(),
            'co_occurrence_matrices': co_occurrence_matrices,
            'new_pairs_df': new_pairs_df.to_dict()  # Ensure this is returned
        }
    }


# ###
# ### Dash callback: `upload_file`
# 
# The `upload_file` function is a Dash callback that:
# 
# - **Handles File Uploads**: Processes user-uploaded files.
# - **Provides User Feedback**: Displays messages based on upload status.
# - **Updates UI Components**: Modifies dropdown options and visibility of data containers based on the uploaded data.
# - **Stores Processed Data**: Saves structured data (like `flat_df`, co-occurrence matrices) for later use in the application.
# 
# ### Key Outputs
# - **Feedback Message**: Informs the user about the upload status.
# - **Data Container Style**: Controls visibility based on file upload success.
# - **Dropdown Options**: Updates available choices based on processed data.
# - **Stored Data**: Keeps processed data for further interaction.
# 

# In[12]:


@app.callback(
    Output('upload-feedback', 'children'),
    Output('data-container', 'style'),
    Output('code-dropdown', 'options'),
    Output('data-store', 'data'),  # Store `flat_df` and matrices here
    Input('upload-data', 'contents')
)


def upload_file(file_content):
    
    if file_content is None:
        return "Please upload the FHIR dataset.", {'display': 'none'}, [], None
    feedback_message = ""
    data_style = {'display': 'none'}
    options = []
    data = {}
    
    print('file_content',file_content)

    if file_content:
        # Decode and process uploaded file
        content_type, content_string = file_content.split(',')
        decoded = base64.b64decode(content_string)
        result = fetch_and_process_data(decoded)


        if result['success']:
            co_occurrence_matrices = result['data']['co_occurrence_matrices']
            flat_df = result['data']['flat_df']
            new_pairs_df = result['data']['new_pairs_df']  # Retrieve new_pairs_df
            
            # Prepend "All codes" to the dropdown options
            options = [{'label': 'All codes', 'value': 'ALL_CODES'}] + [{'label': code, 'value': code} for code in co_occurrence_matrices.get('Main', pd.DataFrame()).columns]

            # Update this part in upload_file callback
            data = {
                'flat_df': flat_df,  # Store the `flat_df` here
                'co_occurrence_matrices': {
                    key: matrix.to_dict() for key, matrix in co_occurrence_matrices.items()
                },
                'new_pairs_df': new_pairs_df  # Store new_pairs_df here directly
            }

            feedback_message = result['message']
            data_style = {'display': 'block'}
        else:
            feedback_message = result['message']

    return feedback_message, data_style, options, data


# ### 
# ### Dash callback: `update_graph`
# ### PYVIS VISUALIZATION
# 
# 1. **Callback Definition**:  
#    The `@app.callback` listens to inputs (dropdowns, sliders, user inputs) and updates the graph, codes of interest, and visualization styles accordingly.
# 
# 2. **Inputs**:  
#    Load necessary data frames (`flat_df`, `co_occurrence_matrices`, `new_pairs_df`, `main_df`) from the `data-store`.
# 
# #### Plotting Pyvis Graphs for All_codes
# 
# 3. **If All Codes Selected**:  
#    - Calculate node sizes and shapes based on code levels.
#    - Find top nodes per resource type.
#    - Identify ancestor codes across levels and merge them into a result dataframe (`result_df`).
# 
# 4. **Update Weights**:  
#    Calculate and propagate weights for nodes across different levels (3, 2, 1).
# 
# 5. **Network Graph Update**:  
#    - Filter `result_df` based on the selected level and update node/edge data.
#    - Adjust edge colors, node styles, and sizes dynamically based on the selected level.
# 
# 6. **User Entry Code Highlighting**:  
#    If a user enters a code, highlight it in the graph for better visibility.
# 
# 
# #### Plotting Pyvis Graphs for individual codes
# 
# 7. **Neighbors of Selected Code**:  
#    Compute the degree (number of neighbors) for each node in the network using the `main_df` data frame. Sort the neighbors of the `selected_code` by their co-occurrence values and store them in `neighbors_sorted`. Then, gather all top neighbors.
# 
# 8. **Helper Function (`add_nodes_edges`)**:  
#    Define a function to add nodes and edges to the network graph.
# 
#    - Iterate through the neighbor codes.
#    - Based on the resource group (ICD, LOINC, OPS), identify the top neighbor.
#    - Store top neighbor info and its connected neighbors.
#    - Compute node size using `normalize_weights` and add the `selected_code` and `top_neighbor` to the graph.
#    - Add edges between nodes and normalize edge thickness using co-occurrence values.
# 
# 9. **Top Neighbor Relations**:  
#    For each top neighbor, check its other neighbors and add them to the graph.
# 
# 10. **Co-occurrence Matrices**:  
#    For each resource type (ICD, LOINC, OPS), check if a co-occurrence matrix is present, and if so, use it to add nodes and edges by calling `add_nodes_edges()`.
# 

# In[13]:


@app.callback(
    [Output('graph-iframe', 'srcDoc'),
     Output('codes-of-interest-store', 'data'),
     Output('graph-iframe', 'style'),
     Output('bar-chart', 'style'),
     Output('dendrogram', 'style')],
    [Input('code-dropdown', 'value'),
     Input('num-nodes-slider', 'value'),
     Input('level-slider', 'value'),
     Input('show-labels', 'value'),
     Input('code-input', 'value'),  # Add input for user code
     State('data-store', 'data')]
)


def update_graph(selected_code, num_nodes_to_visualize, selected_level, show_labels, user_code, data):
  

    EDGE_THICKNESS_MIN = 1
    EDGE_THICKNESS_MAX = 32
    
    #user_code = user_code.replace(" ", "") if user_code else None
    
    if not data:
        return "No data loaded.", None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    
    # Initialize styles
    graph_style = {'display': 'none'}
    bar_chart_style = {'display': 'none'}
    dendrogram_style = {'display': 'none'}
    
    # Check if selected_code is None or empty
    if not selected_code:
        return "", {'codes_of_interest': [], 'top_neighbor_info': {}}, graph_style, bar_chart_style, dendrogram_style

    # Initialize PyVis network
    net = Network(notebook=True, cdn_resources='remote')

    # Handle data
    try:
        flat_df = pd.DataFrame(data.get('flat_df', {}))
        co_occurrence_matrices = data.get('co_occurrence_matrices', {})
        new_pairs_df = pd.DataFrame(data.get('new_pairs_df', {}))  # Access new_pairs_df
        main_df = pd.DataFrame(co_occurrence_matrices.get('Main', {}))
    except Exception as e:
       #print(f"Error processing data: {e}")
        return "", {'codes_of_interest': [], 'top_neighbor_info': {}}, graph_style, bar_chart_style, dendrogram_style

    codes_of_interest = []
    top_neighbor_info = {}
    flat_df['Color'] = flat_df['ResourceType'].map(SUBGROUP_COLORS)
    color_mapping = flat_df.set_index('ResourceType')['Color'].to_dict()


    if selected_code == 'ALL_CODES':
        NODE_SIZE_MIN = 4
        NODE_SIZE_MAX = 32

        # Ensure all node IDs are strings
#         new_pairs_df['Code1'] = new_pairs_df['Code1'].astype(str)
#         new_pairs_df['Code2'] = new_pairs_df['Code2'].astype(str)

        delta_limit = NODE_SIZE_MAX-NODE_SIZE_MIN  # Adjust this value as needed

        # Define node sizes based on levels correlated to delta_limit
        size_mapping = {
            0: NODE_SIZE_MAX,  # Size for level 0
            1: (delta_limit / 4) * 3,  # Size for level 1
            2: (delta_limit / 4) * 2,  # Size for level 2
            3: (delta_limit / 4) * 2,  # Size for level 3
            4: delta_limit / 4   # Size for level 4
        }
        
        shape_mapping = {
            0: 'dot',          # For level 0 (default)
            1: 'triangleDown',     # For level 1
            2: 'square',      # For level 2
            3: 'star',         # For level 3
            4: 'dot',       # For level 4
        }

        def calculate_node_degrees(co_occurrence_matrix):
            # Degree is the sum of co-occurrences for each node across all neighbors
            node_degrees = co_occurrence_matrix.sum(axis=1).reset_index()
            node_degrees.columns = ['Node', 'Degree']
            return node_degrees
        
        # Calculate node degrees
        node_degrees = calculate_node_degrees(main_df)

        # Applying get_resource_type to each node
        node_degrees['ResourceType'] = node_degrees['Node'].apply(get_resource_type)

# MAX NUMBER OF NODES TO BE SHOWN ON THE LEAFS

        n = 10  # You can change this value to get more or fewer top nodes

        # Group by ResourceType, sort by Degree in descending order, and take the top 'n' nodes for each group
        top_n_per_resource = node_degrees.groupby('ResourceType').apply(lambda x: x.nlargest(n, 'Degree')).reset_index(drop=True)

        # Display the result
        print("\nTop Nodes per Resource Type:")
        print(top_n_per_resource)

        # Initialize an empty DataFrame to store results
        result_df = pd.DataFrame()

        top_nodes = top_n_per_resource['Node'].tolist()

        # Create a set for level 4 codes using top_nodes
        level4_codes = set(top_nodes)
        print('')

        # Track Code2 values for ancestors
        ancestor_codes = set()

        # Loop through levels from 4 to 0
        for level in range(4, -1, -1):
            if level == 4:
                # Level 4: Get rows where either Code1 or Code2 is in top_nodes
                level_df = new_pairs_df[(new_pairs_df['level'] == level) & 
                                         (new_pairs_df['Code1'].isin(top_nodes) | 
                                          new_pairs_df['Code2'].isin(top_nodes))]

                # Collect Code2 values from level 4 to track ancestors
                ancestor_codes.update(level_df['Code2'].unique())
                print('Level 4 DataFrame:', level_df)

                # Update prev_level_code1 to include both Code1 and Code2
                prev_level_code1 = set(level_df['Code1']).union(set(level_df['Code2']))
                print('prev_level_code1:', prev_level_code1)

            else:
                # Get previous level Code1 values to filter the current level

                print('prev_level_code1')
                print(level,level_code1, locals())
                level_df = new_pairs_df[(new_pairs_df['level'] == level) & 
                                         (new_pairs_df['Code2'].isin(prev_level_code1)) |
                                         (new_pairs_df['Code1'].isin(ancestor_codes))]  # Include ancestors for all lower levels
                prev_level_code1 = set(level_df['Code1'])
                
            # Collect Code1 values from the current level
            level_code1 = level_df['Code1'].unique()

            # Add the current level results to the result DataFrame
            result_df = pd.concat([result_df, level_df], ignore_index=True)
        result_df.to_csv('result_df-ancestors.csv')
        
        # 2. Create a dictionary to store total weights for level 3 Code2 values
        level3_weights_dict = {}

        # 3. Iterate over each row in result_df where level is 3
        for index, row in result_df[result_df['level'] == 3].iterrows():
            code2_value = row['Code2']

            # 4. Check if the Code2 value exists in the level 4 codes
            if code2_value in level4_codes:
                # 5. Assign the degree from node_degrees to the level 3 weights dictionary
                degree_value = node_degrees[node_degrees['Node'] == code2_value]['Degree']

                # Check if the degree value exists and assign it
                if not degree_value.empty:
                    level3_weights_dict[code2_value] = degree_value.values[0]
                else:
                    level3_weights_dict[code2_value] = 0  # Default to 0 if not found

        # 6. Update the Weight in level 3 rows based on the calculated total weights
        result_df.loc[result_df['level'] == 3, 'Weight'] = result_df.loc[result_df['level'] == 3, 'Code2'].map(level3_weights_dict)

             
        # Function to calculate and assign weights based on levels
        def calculate_weights(levels):
            for current_level in levels:
                # Step 1: Calculate total weights for each Code1 in the current level
                level_weights = result_df[result_df['level'] == current_level].groupby('Code1')['Weight'].sum().reset_index()
                level_weights.rename(columns={'Weight': 'TotalWeight'}, inplace=True)

                # Step 2: Assign weights from the current level to the next lower level
                next_level = current_level - 1
                for index, row in result_df[result_df['level'] == next_level].iterrows():
                    code2_value = row['Code2']

                    if code2_value in level_weights['Code1'].values:
                        total_weight = level_weights.loc[level_weights['Code1'] == code2_value, 'TotalWeight'].values[0]
                        result_df.at[index, 'Weight'] = total_weight

        # Call the function with levels in descending order
        calculate_weights(levels=[3, 2, 1])

 #############   
        print('result_df', result_df)
        result_df.to_csv('result_df.csv')

        def update_fhir_net(df_int, level):

            df = df_int.copy()  

            levels_to_process = list(range(3, level-1, -1))  # Create a list [3, 2, ..., level]

            for current_level in levels_to_process:
                # Step 1: Filter rows for the current level
                level_rows = df[df['level'] == current_level].copy()
                print('level_rows', level_rows)

                # Step 2: Create a translation dictionary from Code2 to Code1
                translation_dict = dict(zip(level_rows['Code2'], level_rows['Code1']))
                print('translation_dict', translation_dict)

                next_level = current_level + 1

                # Step 2: Filter rows for the next level
                df_next_level = df[df['level'] == next_level].copy()

                # Step 3: Create the mask for rows where Code1 or Code2 are NOT in level_rows['Code2']
                mask = ~df_next_level['Code1'].isin(level_rows['Code2']) | ~df_next_level['Code2'].isin(level_rows['Code2'])

                # Step 4: Drop rows where mask is True
                df = df.drop(df_next_level[mask].index, errors='ignore')

                print('filtered_pairs_df', df)

                # Step 5: Apply the translation to the next level rows (level + 1)                
                df.loc[df['level'] == next_level, 'Code1'] = df.loc[df['level'] == next_level, 'Code1'].apply(lambda x: translation_dict.get(x, x))
                df.loc[df['level'] == next_level, 'Code2'] = df.loc[df['level'] == next_level, 'Code2'].apply(lambda x: translation_dict.get(x, x))
                
                # Step 4: Delete rows of the current level
                df_del = df[df['level'] != current_level]
                df_del = df_del[df_del['Code1'] != df_del['Code2']]

                # Display the DataFrame after deleting the level rows
                print(f"\nAfter deleting level {current_level} rows:")
                print(df_del)

                # Step 5: Group by Code1, Code2, level, ResourceType, and Displays, and sum the weights for duplicates
                df_grouped = df_del.groupby(['Code1', 'Code2', 'level'], as_index=False).agg({
                    'Weight': 'sum',
                    'ResourceType': 'first',
                    'Displays': 'first'
                })

                # Step 6: Replace level 4 with level 3 (if applicable)
                df_grouped.loc[df_grouped['level'] == current_level + 1, 'level'] = current_level 

                print(f'df_grouped after replacing level {current_level + 1} with {current_level}:')
                print(set(df_grouped['level']))
                print(len(df_grouped))

                # Update the DataFrame for the next iteration
                df = df_grouped

            return df_grouped
        
        
        if selected_level==4:
            filtered_pairs_df = result_df

        else:
            filtered_pairs_df = update_fhir_net(result_df, selected_level)
    
####################
        print('filtered_pairs_df', filtered_pairs_df)
        fhir_net = generate_network_viz(filtered_pairs_df, 
                                          code1_col='Code1', 
                                          code2_col='Code2', 
                                          weight_col='Weight', 
                                          layout='barnes_hut',
                                          edge_thickness_min=EDGE_THICKNESS_MIN, 
                                          edge_thickness_max=EDGE_THICKNESS_MAX,
                                           selected_level=selected_level)


        min_weight = filtered_pairs_df['Weight'].min()
        max_weight = filtered_pairs_df['Weight'].max()
        
        node_resource_type_map = {}

        for _, row in result_df.iterrows():
            if row['level'] in [1, 2, 3]:  # If level is 1 or 2
                node_resource_type_map[row['Code1'].strip()] = row['ResourceType']
            elif row['level'] == 4:  # If level is 4, assign resource types using the get_resource_type function
                node_resource_type_map[row['Code1'].strip()] = get_resource_type(row['Code1'].strip())
                node_resource_type_map[row['Code2'].strip()] = get_resource_type(row['Code2'].strip())


        # Print the resource type mapping for verification
        print("Node Resource Type Map:")
        for node, resource_type in node_resource_type_map.items():
            print(f"{node}: {resource_type}")

        # Manually map the key nodes to their correct ResourceType
        key_nodes_resource_type = {
            'FHIR': 'FHIR',
            'ICD': 'ICD',
            'LOINC': 'LOINC',
            'OPS': 'OPS'
        }

        # Add key nodes to the resource type map
        node_resource_type_map.update(key_nodes_resource_type)

        # Iterate over each edge in the network to assign normalized thickness and color based on resource type
        for edge in fhir_net.edges:
            from_node = edge['from'].strip()
            to_node = edge['to'].strip()

            # Get the resource types for both 'from' and 'to' nodes from the pre-built map
            from_resource_type = node_resource_type_map.get(from_node, None)
            to_resource_type = node_resource_type_map.get(to_node, None)

            # Debugging prints to check resource types for both nodes
            print(f"Edge from {from_node} to {to_node}")
            print(f"Resource type for {from_node}: {from_resource_type}")
            print(f"Resource type for {to_node}: {to_resource_type}")

            # Default to 'gray' if no specific resource type is found for either node
            edge_color = 'rgba(128, 128, 128, 0.3)'

            # Check if the resource types for both nodes are the same
            if from_resource_type and to_resource_type and from_resource_type == to_resource_type:
                # Assign color based on the resource type found
                edge_color = color_mapping.get(from_resource_type, 'gray')  # Use the color for the shared resource type
                edge_color = color_mapping.get(to_resource_type, 'gray')
            else:
                # Check if from_node or to_node are 'FHIR' and handle accordingly
                if from_node == 'FHIR' and to_node in ['ICD', 'OPS', 'LOINC']:
                    edge_color = color_mapping.get(to_resource_type, 'gray')  # Use color of the connected node
                elif to_node == 'FHIR' and from_node in ['ICD', 'OPS', 'LOINC']:
                    edge_color = color_mapping.get(from_resource_type, 'gray')  # Use color of the connected node

            # Debugging print to verify the assigned edge color
            print(f"Assigned color for edge: {edge_color}")

            # Assign the color to the edge
            edge['color'] = edge_color

            # Normalizing the edge thickness based on weight as before
            weight_value = filtered_pairs_df[
                ((filtered_pairs_df['Code1'].str.strip() == from_node) & 
                 (filtered_pairs_df['Code2'].str.strip() == to_node)) |
                ((filtered_pairs_df['Code1'].str.strip() == to_node) & 
                 (filtered_pairs_df['Code2'].str.strip() == from_node))
            ]['Weight']

            if not weight_value.empty:
                weight_value = weight_value.values[0]

                # Apply the normalization using the correct gain calculation
                normalized_thickness = normalize_weights(
                    weight_value,
                    gain=(EDGE_THICKNESS_MAX - EDGE_THICKNESS_MIN) / (max_weight - min_weight),
                    offset=EDGE_THICKNESS_MIN,
                    min_limit=EDGE_THICKNESS_MIN,
                    max_limit=EDGE_THICKNESS_MAX
                )
                edge['width'] = normalized_thickness
            else:
                edge['width'] = EDGE_THICKNESS_MIN  # Default edge thickness

##########
        level_4_nodes_by_type = {}

        # First, process each node to assign sizes, colors, and labels
        for node in fhir_net.nodes:
            # Identify the node's level and assign the size
            level = result_df.loc[result_df['Code1'] == node['id'], 'level'].values
            level = level[0] if len(level) > 0 else None

            # Set the node size
            node['size'] = size_mapping.get(level, 5)

            # Determine the color based on the level
            if node['id'] == 'FHIR':
                color = 'black'  # FHIR node is black
                node['title'] = 'Fast Healthcare Interoperability Resources'
            if node['id'] == 'ICD':
                node['title'] = 'International Statistical Classification of Diseases and Related Health Problems'
            if node['id'] == 'OPS':
                node['title'] = 'Operationen- und Prozedurenschlüssel'
            if node['id'] == 'LOINC':
                node['title'] = 'Logical Observation Identifiers Names and Codes'
                
            if level in [1, 2, 3]:  # Levels 1, 2, and 3 get their color from ResourceType
                resource_type = result_df.loc[result_df['Code1'] == node['id'], 'ResourceType'].values
                if len(resource_type) > 0:
                    color = color_mapping.get(resource_type[0], 'gray')
                else:
                    color = 'gray'

                # Check if 'show' is in show_labels before assigning display labels for level 2 and level 3 nodes
                if 'show' in show_labels:  # Only assign labels if 'show' is in show_labels
                    display_label = result_df.loc[result_df['Code1'] == node['id'], 'Displays'].values
                    if len(display_label) > 0 and display_label[0] is not None:
                        full_display = display_label[0]  # Full display label
                        truncated_label = full_display[:15]  # Truncate to first 15 characters

                        node['label'] = truncated_label  # Set the truncated display label
                        node['title'] = full_display #if full_display else 'No Description Available'  # Store full display label in 'title'
                        node['text'] = full_display  # Use truncated label for the text field as well
                    else:
                        node['label'] = 'No Label'  # Default or empty label if None
                        node['title'] = 'No Description Available'  # Default title if None

            elif level == 4:  # Level 4 nodes
                
                resource_type = node['id']  # Store the current node's ID
                resource_type_result = get_resource_type(resource_type)
                color = get_color_for_resource_type(resource_type_result)

                # Store level 4 nodes by resource type
                level_4_nodes_by_type.setdefault(resource_type_result, []).append(node['id'])

                # Assign labels and titles for level 4 nodes
                if 'show' in show_labels:
                    display_label = flat_df.loc[flat_df['Codes'] == node['id'], 'Full_Displays'].values
                    if len(display_label) > 0 and display_label[0] is not None:
                        full_display = display_label[0]
                        node['label'] = full_display[:22]  # Limit to first 22 characters
                        node['title'] = full_display
                        node['text'] = full_display
                    else:
                        node['label'] = 'No Label'
                        node['title'] = 'No Description Available'
                node['color'] = color  # Assign color to the node
                
            else:
                node['color'] = 'gray'  # Default color for undefined levels

            node['color'] = color  # Assign color to the node
            node_shape = shape_mapping.get(level, 'dot')  # Default shape is 'dot'
            node['shape'] = node_shape  # Assign the shape

            # Optionally set font size
            node['font'] = {'size': 20}
#########
#########
        print("Final Nodes in fhir_net after level-based filtering:", [node['id'] for node in fhir_net.nodes])
        
            
        # Highlight the selected node and its connections
        if user_code:
            # Highlight the user input node and its direct connections
            for node in fhir_net.nodes:
                if node['id'] == user_code:
                    node['color'] = 'red'  # Highlight color for the selected node
                    node['size'] += 5  # Increase size for visibility

            # Highlight edges connected to the selected node
            for edge in fhir_net.edges:
                if edge['from'] == user_code or edge['to'] == user_code:
                    edge['color'] = 'lime'  # Highlight color for the edge
                    edge['width'] += 5  # Increase size for visibility

        # Show the network visualization
        fhir_net.show('fhir_interactions_highlighted.html')

        # Read the HTML file and return its content for the iframe
        graph_style = {'display': 'block', 'width': '200%', 'height': '750px'}
        try:
            with open('fhir_interactions_highlighted.html', 'r') as file:
                html_content = file.read()
            return html_content, {'codes_of_interest': codes_of_interest, 'top_neighbor_info': top_neighbor_info}, graph_style, bar_chart_style, dendrogram_style
        except Exception as e:
            print(f"Error reading HTML file: {e}")
            return "", {'codes_of_interest': [], 'top_neighbor_info': {}}, graph_style, bar_chart_style, dendrogram_style

        
    else:
        NODE_SIZE_MIN = 4
        NODE_SIZE_MAX = 44
        
        
        graph_style = {'display': 'block', 'width': '100%', 'height': '600px'}
        bar_chart_style = {'display': 'block', 'width': '95%', 'height': '300px'}
        dendrogram_style = {'display': 'block', 'width': '95%', 'height': '300px'}

        # Calculate node degrees (number of neighbors)
        node_degree = main_df.astype(bool).sum(axis=1)
        
        # Sort nodes by their degree in descending order and select the top 10
        top_10_degrees = node_degree.sort_values(ascending=False)[:30]
        print('top_10_degrees', top_10_degrees)

        # Get the minimum and maximum degree from the top 10
        min_weight = top_10_degrees.min()
        max_weight = top_10_degrees.max()
        
        # Get neighbors of the selected code
        if selected_code not in main_df.index:
            return "", {'codes_of_interest': [], 'top_neighbor_info': {}}, graph_style, bar_chart_style, dendrogram_style

        neighbors_sorted = main_df.loc[selected_code].sort_values(ascending=False)
       #print('neighbors_sorted', neighbors_sorted)
        top_neighbors = list(neighbors_sorted.index[:])
       #print('top_neighbors', top_neighbors)

        codes_of_interest = [selected_code]
        top_neighbor_info = {}
      
        
        def add_nodes_edges(graph, child_df, group_name):
            top_neighbor = None

            # Iterate over the neighbor codes in the DataFrame
            for neighbor_code in neighbors_sorted.index:
                if group_name == 'ICD' and is_icd_code(neighbor_code):
                    top_neighbor = neighbor_code
                    break
                elif group_name == 'LOINC' and is_loinc_code(neighbor_code):
                    top_neighbor = neighbor_code
                    break
                elif group_name == 'OPS' and is_ops_code(neighbor_code):
                    top_neighbor = neighbor_code
                    break

            if top_neighbor:
                top_neighbor_info['top_neighbor'] = top_neighbor
                top_neighbor_row = child_df.loc[top_neighbor].sort_values(ascending=False)
                top_neighbors_list = list(top_neighbor_row.index[:num_nodes_to_visualize])
                top_neighbor_info['top_neighbors_list'] = top_neighbors_list

                codes_of_interest.extend([top_neighbor] + top_neighbors_list)

                if 'show' in show_labels:
                    selected_code_label = flat_df.loc[flat_df['Codes'] == selected_code, 'Displays'].iloc[0] if not flat_df.empty else selected_code
                    top_neighbor_label = flat_df.loc[flat_df['Codes'] == top_neighbor, 'Displays'].iloc[0] if not flat_df.empty else top_neighbor
                else:
                    selected_code_label = selected_code
                    top_neighbor_label = top_neighbor

                group_name1 = 'ICD' if selected_code in co_occurrence_matrices.get('ICD', {}) else \
                              'LOINC' if selected_code in co_occurrence_matrices.get('LOINC', {}) else \
                              'OPS' if selected_code in co_occurrence_matrices.get('OPS', {}) else 'Unknown'

                node_size = int(node_degree.get(selected_code, 1))/2
#                 node_size = normalize_weights(node_size, 
#                                        gain=5, offset=1, 
#                                        min_limit=NODE_SIZE_MIN, max_limit=NODE_SIZE_MAX)

                node_size = normalize_weights(
                    node_size,
                    gain=(NODE_SIZE_MAX - NODE_SIZE_MIN) / (max_weight - min_weight),
                    offset=NODE_SIZE_MIN,
                    min_limit=NODE_SIZE_MIN,
                    max_limit=NODE_SIZE_MAX
                )

               #print('selected_code node size', node_size)
                if selected_code not in net.get_nodes():
                    net.add_node(selected_code, size=node_size, title=flat_df.loc[flat_df['Codes'] == selected_code, 'Full_Displays'].iloc[0], label=selected_code_label, color=SUBGROUP_COLORS.get(group_name1, 'gray'))

                node_size = int(node_degree.get(top_neighbor, 1))/2
#                 node_size = normalize_weights(node_size, 
#                        gain=5, offset=1, 
#                        min_limit=NODE_SIZE_MIN, max_limit=NODE_SIZE_MAX)
                node_size = normalize_weights(
                    node_size,
                    gain=(NODE_SIZE_MAX - NODE_SIZE_MIN) / (max_weight - min_weight),
                    offset=NODE_SIZE_MIN,
                    min_limit=NODE_SIZE_MIN,
                    max_limit=NODE_SIZE_MAX
                )
               #print('top_neighbor node size', node_size)
                if top_neighbor not in net.get_nodes():
                    net.add_node(top_neighbor, size=node_size, title=flat_df.loc[flat_df['Codes'] == top_neighbor, 'Full_Displays'].iloc[0], label=top_neighbor_label, color=SUBGROUP_COLORS.get(group_name, 'gray'))

                # Prevent adding edges if the nodes are the same
                if selected_code in net.get_nodes() and top_neighbor in net.get_nodes() and selected_code != top_neighbor:
                    edge_value = int(main_df.loc[selected_code, top_neighbor])
                    # Normalize the edge thickness
                    edge_value = normalize_weights(edge_value, gain=5, offset=1, 
                              min_limit=EDGE_THICKNESS_MIN, max_limit=EDGE_THICKNESS_MAX)

                    net.add_edge(selected_code, top_neighbor, value=edge_value, color=SUBGROUP_COLORS.get(group_name, 'gray'))

                top_neighbor_row = child_df.loc[top_neighbor].sort_values(ascending=False)
                top_neighbors_list = list(top_neighbor_row.index[:num_nodes_to_visualize])

                for neighbor in top_neighbors_list:
                    if neighbor != top_neighbor and child_df.loc[top_neighbor, neighbor] > 0:
                        neighbor_label = flat_df.loc[flat_df['Codes'] == neighbor, 'Displays'].iloc[0] if 'show' in show_labels else neighbor

                        node_size = int(node_degree.get(neighbor, 1))/2
#                         node_size = normalize_weights(node_size, 
#                                gain=5, offset=1, 
#                                min_limit=NODE_SIZE_MIN, max_limit=NODE_SIZE_MAX)  
                        node_size = normalize_weights(
                            node_size,
                            gain=(NODE_SIZE_MAX - NODE_SIZE_MIN) / (max_weight - min_weight),
                            offset=NODE_SIZE_MIN,
                            min_limit=NODE_SIZE_MIN,
                            max_limit=NODE_SIZE_MAX
                        )
                       #print('neighbor node size', node_size)
                        if neighbor not in net.get_nodes():
                            net.add_node(neighbor, size=node_size, title=flat_df.loc[flat_df['Codes'] == neighbor, 'Full_Displays'].iloc[0], label=neighbor_label, color=SUBGROUP_COLORS.get(group_name, 'gray'))

                        # Prevent adding edges if the nodes are the same
                        if top_neighbor in net.get_nodes() and neighbor in net.get_nodes() and top_neighbor != neighbor:
                            edge_value = int(child_df.loc[top_neighbor, neighbor])
                            # Normalize the edge thickness
                            edge_value = normalize_weights(edge_value, gain=5, offset=1, 
                              min_limit=EDGE_THICKNESS_MIN, max_limit=EDGE_THICKNESS_MAX)
                            net.add_edge(top_neighbor, neighbor, value=edge_value)

                for i in range(len(top_neighbors_list)):
                    for j in range(i + 1, len(top_neighbors_list)):
                        neighbor1 = top_neighbors_list[i]
                        neighbor2 = top_neighbors_list[j]

                        if neighbor1 in child_df.index and neighbor2 in child_df.columns:
                            count = child_df.loc[neighbor1, neighbor2]
                            if count > 0:
                                # Prevent adding edges if the nodes are the same
                                if neighbor1 in net.get_nodes() and neighbor2 in net.get_nodes() and neighbor1 != neighbor2:
                                    net.add_edge(neighbor1, neighbor2, value=int(count) / 2, color=SUBGROUP_COLORS.get(group_name, 'gray'))


        # Check for specific keys and#print the corresponding matrix in pandas DataFrame style
        if 'ICD' in co_occurrence_matrices:
           #print('\nICD Co-Occurrence Matrix:')
            icd_matrix = pd.DataFrame(co_occurrence_matrices['ICD'])
           #print(icd_matrix)
            add_nodes_edges(net, icd_matrix, 'ICD')

        if 'LOINC' in co_occurrence_matrices:
           #print('\nLOINC Co-Occurrence Matrix:')
            loinc_matrix = pd.DataFrame(co_occurrence_matrices['LOINC'])
           #print(loinc_matrix)
            add_nodes_edges(net, loinc_matrix, 'LOINC')

        if 'OPS' in co_occurrence_matrices:
           #print('\nOPS Co-Occurrence Matrix:')
            ops_matrix = pd.DataFrame(co_occurrence_matrices['OPS'])
           #print(ops_matrix)
            add_nodes_edges(net, ops_matrix, 'OPS')

        temp_file = tempfile.NamedTemporaryFile(delete=True, suffix='.html')
        temp_file_name = temp_file.name
        temp_file.close()

        net.show(temp_file_name)
        
        #graph_style = {'display': 'block', 'width': '100%', 'height': '600px'}  # Default height for specific codes
        return open(temp_file_name, 'r').read(), {'codes_of_interest': codes_of_interest, 'top_neighbor_info': top_neighbor_info}, graph_style, bar_chart_style, dendrogram_style



# ### 
# ### Dash callback: `update_charts`
# ### DENDOGRAM AND BAR CHART VISUALIZATION
# 1. The callback function updates two charts (a bar chart and a dendrogram) based on user inputs such as the selected code, visibility of labels, the number of nodes, and codes of interest.
# 3. It retrieves co-occurrence matrices and computes a frequency distribution of the selected code and its neighbors to build the bar chart.
# #### Calculate frequency distribution:
# 
# | Code  | A   | B   | C   |
# |-------|-----|-----|-----|
# | A     | 5   | 3   | 1   |
# | B     | 3   | 4   | 2   |
# | C     | 1   | 2   | 3   |
# 
# ###. Steps:
# 
# 1. **Inputs:**
#    - `codes_of_interest`: `['A', 'B', 'C']`.
# 
# 2. **Frequency Distribution:**
#    - Sum co-occurrence values: 
#      - **A = 9**, **B = 9**, **C = 6**.
#    - Total: 24, so relative frequencies are:
#      - **A = 0.375**, **B = 0.375**, **C = 0.25**.
# 
# 3. **Bar Chart:**
#    - Displays frequencies for **A**, **B**, and **C** on the y-axis.
# 
# 4. **Dendrogram:**
#    - Clusters **A**, **B**, and **C** based on their co-occurrence.
# 
# 4. The dendrogram is generated using a clustering technique based on the co-occurrence matrix, grouping similar codes for visualization.
# 
# 
# 
# 
# #### Mathematical Explanation of Dendrogram Generation
# 
# 1. **Co-Occurrence Matrix:**
# Co-occurrence values indicate how often two codes appear together in a dataset. High co-occurrence values suggest a strong relationship, while low values indicate weaker connections.
# Codes with high co-occurrence values will be near.
# Codes with low co-occurrence values will be far apart, indicating less of a relationship.
#    - Let \( M \) be the co-occurrence matrix where \( M[i][j] \) represents the frequency of co-occurrence between code \( i \) and code \( j \). 
#    - For example, for codes \( A, B, C \), the matrix \( M \) can be represented as:
# 
#    \[
#    M = 
#    \begin{bmatrix}
#    5 & 3 & 1 \\
#    3 & 4 & 2 \\
#    1 & 2 & 3
#    \end{bmatrix}
#    \]
# 
# 2. **Distance Calculation:**
#    - Calculate the distance (or dissimilarity) between codes based on the co-occurrence values. A common measure is the **Euclidean distance**:
# 
# To calculate the distance between codes \(i\) and \(j\) in a co-occurrence matrix, we can again use the Euclidean distance formula. In the context of three codes (let's say \(i\), \(j\), and \(k\)), the distance \(d(i,j)\) will depend on the values in the co-occurrence matrix for all three codes.
# 
# ##### Example Co-Occurrence Matrix
# 
# Let’s define a co-occurrence matrix \(M\) for three codes \(A\), \(B\), and \(C\):
# 
# $$
# M = 
# \begin{bmatrix}
# 5 & 3 & 1 \\
# 3 & 4 & 2 \\
# 1 & 2 & 3
# \end{bmatrix}
# $$
# 
# ##### Distance Calculation
# 
# The distance \(d(i,j)\) between two codes \(i\) and \(j\) can be computed as follows:
# 
# $$
# d(i,j) = \sqrt{\sum_k (M[i][k] - M[j][k])^2}
# $$
# 
# Where:
# - \(M[i][k]\) is the co-occurrence value for code \(i\) with respect to code \(k\).
# - The sum is taken over all possible codes \(k\).
# 
# 
# ##### Identify the Rows in the Matrix:
# 
# For Code \(A\): 
# $$
# M[A] = [5, 3, 1]
# $$
# 
# For Code \(B\): 
# $$
# M[B] = [3, 4, 2]
# $$
# 
# ##### Subtract the Values:
# 
# If the co-occurrence of \( A \) with \( B \) (3) is much higher than the co-occurrence of \( C \) with \( B \) (2), then \( A \) is more closely related to \( B \) than \( C \) is. The subtraction will yield a positive value, indicating this relative closeness.
# 
# $$
# M[A] - M[B] = 
# \begin{bmatrix}
# 5 - 3 \\
# 3 - 4 \\
# 1 - 2
# \end{bmatrix}
# =
# \begin{bmatrix}
# 2 \\
# -1 \\
# -1
# \end{bmatrix}
# $$
# 
# ##### Square Each Difference:
# 
# $$
# \begin{bmatrix}
# 2^2 \\
# (-1)^2 \\
# (-1)^2
# \end{bmatrix}
# #### Square Each Difference:
# 
# \[
# \begin{bmatrix}
# 2^2 \\
# (-1)^2 \\
# (-1)^2
# \end{bmatrix}
# =
# \begin{bmatrix}
# 4 \\
# 1 \\
# 1
# \end{bmatrix}
# \]
# 
# #### Sum the Squared Differences:
# 
# \[
# 4 + 1 + 1 = 6
# \]
# 
# #### Take the Square Root:
# 
# $
# d(A,B) = \sqrt{6} \approx 2.45
# $
# 
# #### Summary of Distances
# 
# You can also calculate the distances for the other pairs, \(d(A,C)\) and \(d(B,C)\).
# 
# - $ d(A,B) \approx 2.45 $
# - $ d(A,C) \approx 4.58 $
# - $ d(B,C) = 3 $
# 
# 
# 
# 3. **Clustering:**
#    - Use a clustering algorithm, such as **Agglomerative Clustering**:
#         - Initialization: Treat each data point (or code) as a single cluster.
#         - Distance Calculation: Calculate the distance between all clusters using a specified distance metric (e.g., Euclidean distance).
#         - Merging Clusters: Identify the two closest clusters and merge them into one.
#         - Repeat: Update the distance matrix to reflect the new cluster structure and repeat the merging process until all points are clustered or the desired number of clusters is achieved.
# 
# 4. **Ward's method - Linkage Criteria:**
#    - Define a linkage criterion (e.g., **Ward's method**) to decide how to merge clusters:
#      - This minimizes the total within-cluster variance. The distance between clusters can be computed using:
# 
# The distance between two clusters \( C_a \) and \( C_b \) in Agglomerative Clustering can be calculated using the following equation:
# 
# $$ 
# D(C_a, C_b) = \frac{1}{|C_a| \times |C_b|} \sum_{i \in C_a, j \in C_b} d(i, j) 
# $$
# 
# 5. **Dendrogram Construction:**
#    - Construct the dendrogram by plotting the clusters:
#      - The y-axis represents the distance or dissimilarity, while the x-axis represents the codes.
#      - Each merge is represented by a horizontal line, connecting the merged clusters.
# 
# 
# 

# In[14]:


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
    if not selected_code or selected_code == 'ALL_CODES':
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
    selected_code_label = flat_df.loc[flat_df['Codes'] == selected_code, 'Displays'].iloc[0] if 'show' in show_labels else selected_code

    # Ensure codes_of_interest is a list
    codes_of_interest = codes_of_interest.get('codes_of_interest', [])
   #print("Codes of interest:", codes_of_interest)

    # Prepare bar chart data
    bar_data = []
    x_labels = []
    y_values = []
    line_widths = []
    bar_colors = []

    for neighbor in codes_of_interest:
        occurrence_count = total_freq_dist.get(neighbor, 0)
        neighbor_label = flat_df.loc[flat_df['Codes'] == neighbor, 'Displays'].iloc[0] if 'show' in show_labels else neighbor
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

    # Drop duplicates from sorted_x and keep corresponding y values
    unique_labels = []
    unique_y_values = []
    unique_colors = []
    unique_line_widths = []

    for x, y, color, line_width in zip(sorted_x, sorted_y, sorted_colors, sorted_line_widths):
        if x not in unique_labels:
            unique_labels.append(x)
            unique_y_values.append(y)
            unique_colors.append(color)
            unique_line_widths.append(line_width)
    

    # Create the bar chart
    bar_chart_figure = {
        'data': [{
            'x': unique_labels if 'show' in show_labels else str(unique_labels),  # Conditional for 'x'
            'y': unique_y_values,
            'type': 'bar',
            'name': 'Occurrences',
            'marker': {'color': unique_colors},
            'line': {'width': unique_line_widths},
            'text': unique_labels if 'show' in show_labels else [flat_df.loc[flat_df['Codes'] == label, 'Codes'].iloc[0] for label in unique_labels],#unique_labels,
            'textposition': 'auto'#'none' if 'show' in show_labels else 'auto'
        }],
        'layout': {
            'title': f'Frequency Distribution',
            'xaxis': {
                'title': '',
                'tickangle': -45,  # Rotate x-tick labels for better readability
                'showticklabels': True,  # Show only the labels, no numbers
            },
            'yaxis': {'title': 'Frequency'}
        }
    }


    # Create dendrogram figure
    try:
        if len(codes_of_interest) < 1:
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

       #print("Co-occurrence matrix:\n", cooccurrence_matrix)
       #print("Co-occurrence array shape:", cooccurrence_array.shape)

        clustering = AgglomerativeClustering(n_clusters=1, metric='euclidean', linkage='ward')
        cluster_labels = clustering.fit_predict(cooccurrence_array)
        cooccurrence_matrix['Cluster'] = cluster_labels

        # Generate dendrogram plot
        dendrogram_figure = create_dendrogram_plot(cooccurrence_array, cooccurrence_matrix.index.tolist(), flat_df, show_labels)

        return bar_chart_figure, dendrogram_figure

    except Exception as e:
       #print(f"Error in generating dendrogram: {e}")
        return bar_chart_figure, {'data': [], 'layout': {'title': 'Error generating dendrogram'}}
    
    
if __name__ == '__main__':
    app.run_server(debug=True, port=8052)



# In[ ]:





# # NODE DEGREE DISTRIBUTION

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io

# Use the correct path to read the parquet file
file_path = 'C:/dataset/FHIR_real_data.parquet'
flat_df = pd.read_parquet(file_path)

# Continue with the rest of your code


# Define resource type functions
def is_icd_code(code):
    """Check if the given code is a valid ICD code."""
    if not isinstance(code, str) or not code:  # Check for string type and non-empty
        return False
    return bool(re.match(r"^[A-Z]", code))

def is_loinc_code(code):
    """Check if the given code is a valid LOINC code with a hyphen at [-2]."""
    if not isinstance(code, str) or len(code) < 2:  # Check for string type and minimum length
        return False
    return code[-2] == '-'

def is_ops_code(code):
    """Check if the given code is a valid OPS code."""
    if not isinstance(code, str) or len(code) < 2:  # Check for string type and minimum length
        return False
    return code[1] == '-'

def get_resource_type(code):
    """Determine the resource type based on the code."""
    if is_icd_code(code):
        return "ICD"
    elif is_loinc_code(code):
        return "LOINC"
    elif is_ops_code(code):
        return "OPS"
    else:
        return "Unknown"  # Default case for unrecognized codes



# Create co-occurrence matrices
def create_co_occurrence_matrix(df):
    if df.empty:
        return pd.DataFrame()
    patient_matrix = df.pivot_table(index='PatientID', columns='Codes', aggfunc='size', fill_value=0)
    print("patient_matrix:\n", patient_matrix)  # Displaying patient matrix for debugging
    patient_matrix = patient_matrix.loc[:, (patient_matrix != 0).any(axis=0)]
    co_occurrence_matrix = patient_matrix.T.dot(patient_matrix)
    np.fill_diagonal(co_occurrence_matrix.values, 0)  # Filling diagonal with 0
    return co_occurrence_matrix

# Create co-occurrence matrix from flat_df
main_df = create_co_occurrence_matrix(flat_df)

# Get the degree for each code from the co-occurrence matrix
degrees = main_df.sum(axis=1).reset_index()
degrees.columns = ['Code', 'Degree']  # Renaming for clarity

# Assign resource types to degrees DataFrame using the new function
degrees['ResourceType'] = degrees['Code'].apply(get_resource_type)

# Extract degree values for each resource type
icd_degrees = degrees[degrees['ResourceType'] == 'ICD']['Degree']
loinc_degrees = degrees[degrees['ResourceType'] == 'LOINC']['Degree']
ops_degrees = degrees[degrees['ResourceType'] == 'OPS']['Degree']

# Combine degrees from all resource types for overall histogram properties
sorted_degree_values = pd.concat([icd_degrees, loinc_degrees, ops_degrees]).values

# Set fixed bin size for histogram
bins = np.arange(0, max(sorted_degree_values) + 10, 2)  # Adjusted to accommodate maximum degree

# Plot overlapping histograms for each resource type
plt.figure(figsize=(10, 6))
plt.xlim([min(sorted_degree_values) - 15, max(sorted_degree_values) + 15])



# Plot histograms
plt.hist(icd_degrees, bins=bins, alpha=0.5, color="#00bfff", label='ICD')
plt.hist(loinc_degrees, bins=bins, alpha=0.5, color="#ffc0cb", label='LOINC')
plt.hist(ops_degrees, bins=bins, alpha=0.5, color="#9a31a8", label='OPS')

# Add titles and labels
plt.title('Node Degree Distribution by Resource Type (Fixed Bin Size)')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.grid(axis='y')  # Optional: Add grid for better visibility
plt.legend(loc='upper right')  # Add a legend

# Show the plot
plt.show()


# In[ ]:



