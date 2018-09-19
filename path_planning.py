import math
import random
from string import ascii_lowercase as ascii_l, ascii_uppercase as ascii_u

from graph_tool.all import *
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from treelib import Node, Tree


NUM_LAYERS = 5
BRANCHING_FACTOR = 2
# NUM_VERTICES = 20
def main():
    g = initialize_graph()
    tree = construct_tree(g)
    paths = generate_paths(g, tree)

    tree.show()
    # print(tree.get_node('B').data)
    for path in paths:
        print(path)


def initialize_graph(num_layers=NUM_LAYERS):
    """ initializes the graph with randomized vertex/edge weights using the graph_tool library """
    g = Graph(directed=True)
    g.vp.weights = g.new_vertex_property("double")
    g.ep.weights = g.new_edge_property("double")
    add_vertices(g, num_layers)
    add_edges(g)
    return g


def construct_tree(g):
    # tree.create_node(tag, id, parent, data)
    tree = Tree()
    layers = form_layers(g)
    node_ids = iter(ascii_l + ascii_u)

    node_id = next(node_ids)
    tree.create_node(identifier=node_id, data=layers[0][0])
    prev_ids = [node_id]

    for i in range(1,len(layers),1):
        layer = layers[i]
        curr_ids = []
        prev_ids = iter(prev_ids)
        
        parent_check = True
        for node in layer:
            if parent_check:
                parent_id = next(prev_ids)
            node_id = next(node_ids)
            tree.create_node(identifier=node_id, data=node, parent=parent_id)
            curr_ids.append(node_id)
            parent_check = not parent_check
        prev_ids = curr_ids
    return tree


def generate_paths(g, tree):
    """ A naive implementation where paths are generated from DFS traversal of a tree """
    # tree.add_node(node[, parent])
    # tree.get_node(nid)
    tree_paths = tree.paths_to_leaves()
    vertex_paths = []
    for tree_path in tree_paths:
        vertex_path = generate_path(g,tree,tree_path)
        vertex_paths.append(vertex_path)
    return  vertex_paths


# implement using OR Tools and get_dist_matrix
def generate_path(g, tree, tree_path):
    path = []
    for node_id in tree_path:
        node = tree.get_node(node_id)
        path += node.data
    return optimize_path(g, path)


def add_vertices(g, num_layers, spread=BRANCHING_FACTOR):
    # add vertices with weights to graph
    v = g.add_vertex(1)
    g.vp.weights[v] = 2**num_layers
    for i in range(1,num_layers,1):
        num_vertices = spread**i
        min_weight = 2**(num_layers-i)
        V = g.add_vertex(num_vertices)
        for v in V:
            g.vp.weights[v] = min_weight*(random.random()+1)
    return g.num_vertices()


def add_edges(g):
    # add edges with weights to graph
    for i in range(g.num_vertices()):
        for j in range(g.num_vertices()):
            e = g.add_edge(i,j)
            g.ep.weights[e] = random.randint(1,100)
    return g.num_edges()


def form_layers(g):
    standardize_graph(g)
    wv_min = min(g.vp.weights)
    num_layers = math.ceil(1 - math.log(wv_min,2))
    layers = [[] for _ in range(num_layers)]
    for v in g.vertices():
        wv = g.vp.weights[v]
        tier = math.ceil(-math.log(wv,2))
        layers[tier].append(int(v)) 
    layers = partition_layers(layers)
    return layers


def standardize_graph(g):
    wv_max = max(g.vp.weights)
    for v in g.vertices():
        g.vp.weights[v] /= wv_max


def partition_layers(layers, split=2):
    """ partitions the each layer of a list according to the layer # and number of elements """ 
    layers_partitioned = []
    for i in range(len(layers)):
        num_parts = split ** i
        layer_partitioned = partition_list(layers[i], num_parts)
        layers_partitioned.append(layer_partitioned)
    return layers_partitioned


def partition_list(l, num_sublists):
    """ splits a single list l into even sublists """
    l = iter(l)
    list_new = [[] for _ in range(num_sublists)]
    count = 0
    for item in l:
        index = count % num_sublists
        list_new[index].append(item)
        count += 1
    return list_new


def optimize_path(g, vertex_list):
    """ optimizes the order of nodes visited in the returned path by a TSP approx """
    tsp_size = len(vertex_list)
    dist_callback = create_distance_callback(g, vertex_list)
    routing, search_parameters = create_routing_model(tsp_size, dist_callback)
    _, route = solve_tsp(vertex_list, routing, search_parameters)
    return route
    

def solve_tsp(vertex_list, routing, search_parameters):
    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)
    if assignment:
        route = []
        route_number = 0
        index = routing.Start(route_number)
        while not routing.IsEnd(index):
            route.append(vertex_list[routing.IndexToNode(index)])
            index = assignment.Value(routing.NextVar(index))
        route.append(vertex_list[routing.IndexToNode(index)])
        tot_dist = assignment.ObjectiveValue()
        return tot_dist, route
    else:
        print('No solution found.')


def create_routing_model(tsp_size, dist_callback, depot=0, num_routes=1):
    routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    return routing, search_parameters


def create_dist_matrix(g, vertex_list):
    """ takes a set of vertices and returns their connected edge costs """
    dist_matrix = []
    for v1 in vertex_list:
        dists = []
        for v2 in vertex_list:
            e = g.edge(v1,v2)
            w = g.ep.weights[e]
            dists.append(w)
        dist_matrix.append(dists)
    # print(line) for line in dist_matrix
    return dist_matrix


def create_distance_callback(g, vertex_list):
    # Create a callback to calculate distances between cities.
    dist_matrix = create_dist_matrix(g, vertex_list)

    def distance_callback(from_node, to_node):
        return int(dist_matrix[from_node][to_node])

    return distance_callback


main()




