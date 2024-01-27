import networkx as nx
import igraph as ig
import cdlib
import scipy.io

# path = "dataset/archives/cond-mat/cond-mat.gml"
# path = "dataset/archives/celegansneural/celegansneural.gml"
# path = "dataset/archives/hep-th/hep-th.gml"

name = "fb-75"
path = f"dataset/data/{name}.gml"

# graph_matrix = scipy.io.mmread(path)
# G = nx.Graph(graph_matrix)

# Load the graph
G = nx.read_gml(path, label="id")

print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())

# Convert to igraph
g = ig.Graph.from_networkx(G)

# Apply Community Detection
# Greedy
communities = g.community_fastgreedy()
com_list = [c for c in communities.as_clustering()]
node_cluster = cdlib.NodeClustering(com_list, g)
# Print number of communities
print("Number of communities: ", len(node_cluster.communities))

# Louvain
communities = g.community_multilevel()
com_list = [c for c in communities]
node_cluster = cdlib.NodeClustering(com_list, g)
# Print number of communities
print("Number of communities: ", len(node_cluster.communities))

# walktrap
communities = g.community_walktrap()
com_list = [c for c in communities.as_clustering()]
node_cluster = cdlib.NodeClustering(com_list, g)
# Print number of communities
print("Number of communities: ", len(node_cluster.communities))


path_write = "../../dataset/data/"
# Write the graph in a gml file
# nx.write_gml(G, path_write + f"{name}.gml")
# nx.write_edgelist(G, path_write + f"{name}.mtx", data=False)
