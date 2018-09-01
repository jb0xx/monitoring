import math
import ortools
import random
from string import ascii_lowercase as ascii_l, ascii_uppercase as ascii_u

from graph_tool.all import *
from treelib import Node, Tree

NUM_LAYERS = 5
SPREAD = 2
# NUM_VERTICES = 20
def main():
	g = initialize_graph()
	tree = construct_tree(g)
	tree.show()
	# visualization	

	
	# print(g.vp.weights[1])

	# Form Paths
	# m = get_dist_matrix(g, vertex_list)
	# print(m)
	


def initialize_graph(num_layers=NUM_LAYERS):
	""" initializes the graph with randomized vertex/edge weights using the graph_tool library """
	g = Graph(directed=True)
	g.vp.weights = g.new_vertex_property("double")
	g.ep.weights = g.new_edge_property("double")
	add_vertices(g, num_layers)
	print(add_edges(g))
	return g


def add_vertices(g, num_layers):
	# add vertices with weights to graph
	v = g.add_vertex(1)
	g.vp.weights[v] = 2**num_layers
	for i in range(1,num_layers,1):
		num_vertices = SPREAD**i
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


def generate_paths(tree):
	""" A naive implementation where paths are generated from DFS traversal of a tree """
	# tree.add_node(node[, parent])
	# tree.get_node(nid)
	# tree.paths_to_leaves()
	pass


def form_layers(g):
	standardize_graph(g)
	wv_min = min(g.vp.weights)
	num_layers = math.ceil(1 - math.log(wv_min,2))
	layers = [[] for _ in range(num_layers)]

	for v in g.vertices():
		wv = g.vp.weights[v]
		tier = math.ceil(-math.log(wv,2))
		layers[tier].append(int(v))	
	return layers


# implement using OR Tools and get_dist_matrix
def generate_path(vertex_list, dist_matrix):
	pass


def get_dist_matrix(g, vertex_list):
	""" takes a set of vertices and returns their connected edge costs """
	dist_matrix = []
	for v1 in vertex_list:
		dists = []
		for v2 in vertex_list:
			e = g.edge(v1,v2)
			w = g.ep.weights[e]
			dists.append(w)
		dist_matrix.append(dists)
	return dist_matrix


def standardize_graph(g):
	wv_max = max(g.vp.weights)
	for v in g.vertices():
		g.vp.weights[v] /= wv_max


main()




