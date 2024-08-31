#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
from dash import html
from dash.dependencies import Input, Output
import pyvis
from pyvis.network import Network

# Create Pyvis Network graph
net = Network(notebook=True)
net.add_node('Pelin')
net.add_node('Volkan')
net.add_edge('Pelin', 'Volkan')
net_html = net.show("graph.html")

# Create Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Iframe(srcDoc=open('graph.html', 'r').read(), style={'width': '100%', 'height': '600px'})
])

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




