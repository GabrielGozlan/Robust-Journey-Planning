import pandas as pd
import networkx as nx
import numpy as np
from collections import deque
import heapq 
from scipy.stats import bernoulli, gamma
import random
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression


# +
def load_logistic(coefs):
    """load logistic regression coefficients."""
    # initialize model
    lr = LogisticRegression()

    # load coefficients
    lr.intercept_ = coefs.iloc[0]["coefficients_lr.value"]
    lr.coef_ = coefs.iloc[1:]["coefficients_lr.value"].values.reshape(1, -1)
    lr.classes_ = np.array([0, 1])
    lr.feature_names_in_ = ["rush_hour", "day_cat_index", "transport_mode", "stop_lat", "stop_lon"]
    return lr

def predict_observation(lr, row_new_data, convention_transport, convention_hours):
    """make a prediction. Interest of function is to handle rush hours/transport mode conventions."""
    # select feature columns
    row = pd.DataFrame(row_new_data).transpose()
    row = row[["departure_time", "transport_mode", "stop_lat_to", "stop_lon_to"]]

    # create column hour
    row["hour"] = row["departure_time"].values[0].hour
    
    # add rush hour info
    try:
        hour = row["departure_time"].values[0].hour
        row["rush_hour"] = int(convention_hours[
            (convention_hours["produkt_id"] == row["transport_mode"].values[0]) &
            (convention_hours["hour"] == hour)
        ]["hour"].values)
    except:
        row["rush_hour"] = 0

    # add transport info
    row["transport_mode"] = int(convention_transport[
        (convention_transport["produkt_id"] == row["transport_mode"].values[0])
    ]["produkt_id_index"].values)

    row = row.rename(columns={"stop_lat_to": "stop_lat", "stop_lon_to": "stop_lon", "departure_time": "day_cat_index"})
    row["day_cat_index"] = 0
    row = row[lr.feature_names_in_]
    prediction = lr.predict_proba(row)
    
    return prediction


# -

def rename_col(df):
    columns = df.columns.tolist()
    for i, col in enumerate(columns): 
        x = col.split('.')
        columns[i] = x[1]
    df.columns = columns
    return df


def graph_creation(df, src_nodes, dst_nodes, attributes, key=None):
    """ Create a graph from a given dataframe
    Input:
        - df: dataframe
        - src_nodes: column name for the source nodes
        - dst_nodes: column name for the destination nodes 
        - attributes: array of column names that we want as edge attributes 
        - key: column that we use as key for multigraph edges (by default = None)
    """
    graph = nx.from_pandas_edgelist(df,
                                    src_nodes,
                                    dst_nodes,
                                    edge_key=key,
                                    edge_attr = attributes,
                                    create_using = nx.MultiDiGraph())
    return graph


def string_datetime(time_str):
    return datetime.datetime.strptime(time_str, "%H:%M:%S").time()


def time_diff(time1: str, time2: str):
    """
    Compute the difference in minutes between two times.
    
    Args:
    - time1, time2: times in the format 'HH:MM:SS'.
    
    Returns:
    - float: difference in minutes.
    """
    time_format = "%H:%M:%S"
    if len(time1.split(':'))==3:
        t1 = datetime.strptime(time1, time_format)
    elif len(time1.split(':'))==2:
        t1 = time1 + ":00"
        t1 = datetime.strptime(t1, time_format)
        
    if len(time2.split(':'))==3:
        t2 = datetime.strptime(time2, time_format)
    elif len(time2.split(':'))==2:
        t2 = time2 + ":00"
        t2 = datetime.strptime(t2, time_format)

    delta = t2 - t1
    
    return delta.total_seconds()/60


def dijkstra(dataGraph: pd.DataFrame, arrival_time_: str,
             start_node: str, end_node: str,
             max_nb_trip: int, propositions: int,
             model, convention_transport, convention_hours):
    """
    propositions: number of paths to propose at the end
    """
    # We want a 2h window between the possible departures time and the wanted arrival time
    arrival_time = datetime.strptime(arrival_time_, "%H:%M:%S")
    departure_time = arrival_time - timedelta(hours=1)
    
    possible_departure_times = dataGraph[
            (dataGraph.stop_id_from == start_node) &
            (dataGraph.arrival_time <= arrival_time.time()) &
            (dataGraph.departure_time >= departure_time.time())
        ].sort_values("departure_time", ascending=False)

    admissible_paths = []
    admissible_min_cost = []
    
    for attempt, temp in possible_departure_times.iterrows():
        lines_list = []
        proba_list = []
        
        current_departure_ = datetime.strptime(str(possible_departure_times.loc[attempt].departure_time), "%H:%M:%S")
        # from latest to earliest departures from final time
        pq = [(0, start_node, current_departure_)]
        min_cost = {start_node: 0}
        path = {start_node: None}

        
        while pq:
            current_cost, current_node, current_departure = heapq.heappop(pq)
            
            # If the cost of reaching the current node is higher than the known minimum, skip it
            if current_cost > min_cost[current_node]:
                continue
            
            # Explore the neighbors of the current node
            # the neighbours are the stops reachable in 1 step from current node,
            # look at departures from current node in the next 15min window (next arrival is every 15min roughly)
            neighbours = dataGraph[
                (dataGraph.stop_id_from == current_node) &
                (dataGraph.departure_time >= current_departure.time()) &
                (dataGraph.departure_time <= (current_departure + timedelta(minutes=5)).time()) &
                (dataGraph.arrival_time <= arrival_time.time())
            ].sort_values("departure_time")
    
            cost = current_cost
            for i, neighbour in neighbours.iterrows():

                # probability no delay
                p = predict_observation(model, neighbour, convention_transport, convention_hours)[0][0]
    
                # sample {0, 1} bernoulli
                sample = bernoulli(1-p).rvs(1)
    
                # weight according to rv
                if sample == 0:
                    cost += neighbour.weight
                else:
                    # add 10min wait
                    cost += neighbour.weight + 10 * 60
                
                # If a cheaper path to the neighbor is found
                if neighbour.stop_id_to not in min_cost or cost < min_cost[neighbour.stop_id_to]:
                    # if too many changes, test next edge
                    if len(lines_list) > max_nb_trip:
                        continue

                    proba_list.append(p)

                    min_cost[neighbour.stop_id_to] = cost

                    current_departure = datetime.strptime(str(neighbour.departure_time), "%H:%M:%S")
                    heapq.heappush(pq, (cost, neighbour.stop_id_to, current_departure))
                    params = dict({
                        "type": neighbour.transport_mode,
                        "arrival_time_to": str(neighbour.arrival_time),
                        "departure_time_from": str(neighbour.departure_time),
                        "weight": cost,
                        "probability": np.mean(proba_list),
                        "lat_from": neighbour.stop_lat_from,
                        "lon_from": neighbour.stop_lon_from,
                        "lat_to": neighbour.stop_lat_to,
                        "lon_to": neighbour.stop_lon_to,
                        "transport_mode": neighbour.transport_mode,
                        "line_text": neighbour.line_text_from
                    }
                    )
                    path[neighbour.stop_id_to] = (current_node, params)
                    # if neighbour.line_text_from not in lines_list:
                    #     lines_list.append(neighbour.line_text_from)
                cost = current_cost
                
        # check if end node exists -> valid path
        if end_node in path.keys():
            print("A path was found!")
            admissible_paths.append(path)
            admissible_min_cost.append(min_cost)

        if len(admissible_paths) == propositions:
            break
        
    return admissible_min_cost[:propositions], admissible_paths[:propositions]


def paths_formatting(dataGraph, arrival, start_node, end_node, max_nb_trip, propositions, model, convention_transport, convention_hours):
    
    all_min_costs, all_paths = dijkstra(dataGraph, 
                                        arrival,
                                        start_node, 
                                        end_node, 
                                        max_nb_trip, 
                                        propositions, model, convention_transport, convention_hours)
    admissible_paths = []
    
    for path in all_paths:
        path_ = []
        current_end = end_node
        
        while current_end != start_node:
            (previous, params) = path[current_end]
            path_.append((
                previous,
                current_end,
                params
            )) 
            current_end=previous
            
        path_ = list(reversed(path_))
        admissible_paths.append(path_)
        
    print(f"Number of paths: {len(admissible_paths)}")
    print("Departure time:",  admissible_paths[0][0][-1]["departure_time_from"])
    print("Arrival time:",  admissible_paths[0][-1][-1]["arrival_time_to"])
    print(f'Fastest journey\'s time: {time_diff(admissible_paths[0][0][-1]["departure_time_from"], admissible_paths[0][-1][-1]["arrival_time_to"])} min.')
    
    return admissible_paths, all_min_costs

# +
#def 
# -




