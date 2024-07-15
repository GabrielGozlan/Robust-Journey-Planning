# # Journey Planning
#
# The goal of this notebook is to implement a __journey planner__ using a modified version of __Dijsktra's Algorithm__.
#
# All the functions necessary to run our algorithm are defined in _helpers_algorithm.py_.

# ## Data Loading & Preprocessing

# %load_ext autoreload
# %autoreload 2

# # +
import os
import networkx as nx
import numpy as np
import pandas as pd
import datetime
from Constants import region_name

from helpers_algorithm import *

# Configurations
pd.set_option("display.max_columns", 50)
# %matplotlib inline
# # +
from pyhive import hive
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

default_db = 'com490final'
hive_server = os.environ.get('HIVE_SERVER','iccluster080.iccluster.epfl.ch:10000')
hadoop_fs = os.environ.get('HADOOP_DEFAULT_FS','hdfs://iccluster067.iccluster.epfl.ch:8020')
username  = os.environ.get('USER', 'anonym')
(hive_host, hive_port) = hive_server.split(':')

conn = hive.connect(
    host=hive_host,
    port=hive_port,
    username=username
)

# create cursor
cur = conn.cursor()

print(f"hadoop hdfs URL is {hadoop_fs}")
print(f"your username is {username}")
print(f"you are connected to {hive_host}:{hive_port}")

region_name, region_id = "lausanne", 2056

# #### Import data for predictions (coefficients)

# region
# we load data from fitted logistic regression model
coefs = pd.read_sql(f"SELECT * FROM {username}.coefficients_lr", conn)

# conventions are used for peak hours and transport mode (not onehot encoded yet?)
convention_hours = rename_col(pd.read_sql(f"SELECT * FROM {username}.convention_hours", conn))
convention_transport = rename_col(pd.read_sql(f"SELECT * FROM {username}.convention_transport", conn))

# we initialize the model
lr = load_logistic(coefs)
# endregion

# #### Import data for route planner

dataGraph = pd.read_sql(f"SELECT * FROM {username}.orc_data_graph_one_day_{region_name}", conn)
dataGraph = rename_col(dataGraph)
dataGraph.head()

# # +
# Add a column with duration time in order to have the edges' weights
dataGraph["duration"] = dataGraph.apply(lambda row: time_diff(str(row['departure_time']), str(row['arrival_time'])), axis=1)
dataGraph["weight"] = dataGraph["duration"].astype(float)
# As keys for the edges, we use the arrival times as strings
dataGraph["key"] = dataGraph["arrival_time"].astype(str)

dataGraph.head()
# -

# Transform string into date time for the arrival and departure times
dataGraph["departure_time"] = pd.to_datetime(dataGraph.departure_time, format="%H:%M", errors="coerce").dt.time
dataGraph["arrival_time"] = pd.to_datetime(dataGraph.arrival_time, format="%H:%M", errors="coerce").dt.time

dataGraph.head()

# add to notebook data
istdaten = pd.read_sql(f"SELECT * FROM {username}.orc_data_istdaten_{region_name}", conn)
istdaten = rename_col(istdaten)

stops_col = ["stop_id", "stop_lat", "stop_lon", "line_text"]
stops = istdaten[stops_col]

stops.head()

stops = stops.drop_duplicates()
len(stops)

# add coord stop_id_from
dataGraph = dataGraph.merge(
    stops.rename(
        columns=dict(
            zip(
                stops.columns,
                list(map(lambda col: f"{col}_from", stops.columns))
            )),
        ),
        on="stop_id_from"
)
# add coord stop_id_to
dataGraph = dataGraph.merge(
    stops.rename(
        columns=dict(
            zip(
                stops.columns,
                list(map(lambda col: f"{col}_to", stops.columns))
            )),
        ),
        on="stop_id_to"
)

dataGraph.head()

# ### Graph creation

# Creation of the graph, using graph_creation defined in helpers_algorithm
graph_attributes = ["departure_time", "arrival_time", "duration", "transport_mode", # transport_mode
                    "stop_name_to", "stop_name_from", "trip_id"]
graph = graph_creation(df=dataGraph, src_nodes="stop_id_from", dst_nodes="stop_id_to", 
                       attributes=graph_attributes, 
                       key="key"
                      )

# region
# nx.draw(graph)
# endregion

nx.is_strongly_connected(graph)

# The graph is neither strongly connected nor even connected. For our algorithm to function properly for every node, we need to be within a connected component. Therefore, we limit our scope to the largest connected component of the graph.

scc = nx.strongly_connected_components(graph)
# Take the largest component --> this gives us only the nodes id (here: stop ids)
graph_connected = list(max(nx.strongly_connected_components(graph), key=len)) 

# We make a copy of our original data
data = dataGraph.copy(deep=True)

# We restrict our dataframe to the selected component
dataGraph = dataGraph[dataGraph['stop_id_to'].isin(graph_connected)]
dataGraph = dataGraph[dataGraph['stop_id_from'].isin(graph_connected)]

# We create a new graph 
graph_2 = graph_creation(df=dataGraph, src_nodes="stop_id_from", dst_nodes="stop_id_to", 
                       attributes=graph_attributes, 
                       key="key"
                      )

#nx.draw(graph_2, node_size=50)
nx.is_strongly_connected(graph_2)

# region
# edges_graph = graph_2.edges(data=True)
# endregion

# ## Algorithm

dataGraph.head()

# region
# pd.DataFrame(dataGraph.iloc[0]).transpose()["departure_time"].values[0].hour
# endregion

# region
# predict_observation(lr, dataGraph.iloc[0], convention_transport, convention_hours)[0][0]
# endregion

# region
# from datetime import datetime, timedelta
# start_node = "8592080"
# departure = datetime.strptime("13:30:30", "%H:%M:%S")
# arrival = datetime.strptime("15:30:30", "%H:%M:%S")
# endregion

# Now, choose a source and detination node present in our connected graph, departure and arrival times and run the algorithm.

# # +
from datetime import datetime, timedelta
## If you want to test the algorithm without the visualization, run this cell and change start_node and end_node 
## By node present in the strongly connected graph.

# region
# start_node = "8501209" # malley
# end_node = "8592050" # lausanne gare
# arrival = "15:30:30" # str
# endregion

# region
#admissible_paths, admissible_min_costs = paths_formatting(dataGraph, 
#                                                         arrival,
#                                                         start_node, 
#                                                         end_node, 
#                                                         10, 
#                                                         3
#                                                        )
#
# admissible_paths[0]
# endregion

# ## Visualization

import plotly.graph_objects as go
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, HTML
import random
import pandas as pd

# region
transport_colors = {
        'Zug': 'blue',
        'Tram': 'green',
        'Bus': 'red',
        'Metro': 'orange'
    }

def plot_route_on_map(route, colors=transport_colors):
    fig = go.Figure()
    for start_node, end_node, params in route:
        lons = []
        lats = []
        names = []
        # lines = []
        
        lons.append(params["lon_from"])
        lats.append(params["lat_from"])
        lons.append(params["lon_to"])
        lats.append(params["lat_to"])
        names.append(start_node)
        names.append(end_node)
        names.append(params["line_text"])
        
        color = colors.get(params["transport_mode"], 'black')
        
        fig.add_trace(go.Scattermapbox(
            mode="markers+lines",
            lon=lons,
            lat=lats,
            hovertext=names,
            marker={'size': 10, "color": color},
            line=dict(color=color, width=4),
            name=params["transport_mode"] # segment["mode"]
        ))
    
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox={'center': center_map, 'zoom': 12},
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        autosize=True
    )
    
    fig.show()
# endregion

def plot_route_timeline(routes, colors=transport_colors):
    timeline_data = []
    for i, route in enumerate(routes):
        for start_node, end_node, params in route:
            timeline_data.append(dict(
                Route=f"Route {i+1} (P={params['probability']*100:.2f}%)",
                Task=f"{start_node} to {end_node}",
                Start=f"{params['departure_time_from']}",
                Finish=f"{params['arrival_time_to']}",
                Resource=params["transport_mode"],
                Line=params["line_text"]
            ))
    
    df = pd.DataFrame(timeline_data)

    fig = px.timeline(
        df, 
        x_start="Start", 
        x_end="Finish", 
        y="Task", 
        color="Resource",
        facet_row="Route",
        color_discrete_map=colors,
        height=600
    )
    
    fig.update_layout(
        title='Route Timelines',
        xaxis_title='Time',
        yaxis_title='Stops',
        yaxis_autorange='reversed',
        autosize=True,
        margin={'l': 0, 't': 50, 'b': 0, 'r': 0},
        showlegend=True
    )

    fig.update_yaxes(categoryorder="total ascending")

    fig.show()

from helpers_algorithm import paths_formatting

def get_mock_routes(start_node:str, end_node:str, arrival_time:str, success_probability:float, max_nb_trips:int=5, top_routes:int=3):
    routes, _ = paths_formatting(dataGraph, arrival_time,
                                 start_node, end_node,
                                 max_nb_trips, top_routes,
                                 lr, convention_transport, convention_hours)
    return routes

def on_search_button_clicked(b):
    start = start_stop.value
    end = end_stop.value
    time = arrival_time.value
    probability = success_probability.value
    num_routes = num_routes_display.value

    print("Searching...")
    routes = get_mock_routes(start, end, time, probability)[:num_routes]
    shortest_route = routes[0]  # Assuming the first route is the shortest/fastest for simplicity
    plot_route_on_map(shortest_route)
    plot_route_timeline(routes)


center_map = {'lon': dataGraph.stop_lon_to.mean(), 'lat': dataGraph.stop_lat_to.mean()}
print(center_map)

# region
## INITIALIZATION ##
start_node = "8501209" # malley
end_node = "8592050" # lausanne gare
arrival_time = '15:30:00'

start_stop = widgets.Text(value=start_node, description='Start Stop:') 
end_stop = widgets.Text(value=end_node, description='End Stop:')
arrival_time = widgets.Text(value=arrival_time, description='Arrival Time:')
success_probability = widgets.FloatSlider(value=0.9, min=0.0, max=1.0, step=0.05, description='Success Probability:', layout=widgets.Layout(width='80%'))
num_routes_display = widgets.IntSlider(value=3, min=1, max=5, step=1, description='Number of routes:', layout=widgets.Layout(width='80%'))
search_button = widgets.Button(description='Search Route')

display(start_stop, end_stop, arrival_time, success_probability, num_routes_display, search_button)
search_button.on_click(on_search_button_clicked)
# endregion




