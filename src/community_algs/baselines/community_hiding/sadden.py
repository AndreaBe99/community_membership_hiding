from calendar import c
from scipy.stats import entropy
from typing import List, Set, Tuple

import networkx as nx
import igraph as ig
import numpy as np
import cdlib
import copy
import random


class Safeness:
    """Class that implements the Safeness algorithm"""

    def __init__(
        self,
        budget: int,
        graph: nx.Graph,
        community_target: List[int],
        communities_object: cdlib.NodeClustering,
    ):
        self.budget = budget
        self.graph = graph.copy()
        self.community_target = community_target
        self.community_obj = communities_object

        # Initialize as None the rest of the variable
        self.deg = None
        self.out_deg = None
        self.out_ratio = None
        self.new_adj = None
        self.new_edge_list = None
        self.IG_edgeList = None
        self.pre_computation()

    ############################################################################
    #               PERFORM COMMUNITY DECEPTION with SAFENESS                  #
    ############################################################################
    def run(self):
        """
        Run the Safeness algorithm

        Returns
        -------
        new_graph : nx.Graph
            New graph, where the edges have been added and removed
        steps : int
            Number of edges added or removed
        """
        intra_considered = []
        num_vertices = len(self.community_target)
        beta = self.budget
        add_gain = 0
        del_gain = 0
        while True:
            # print("Beta: ", beta)
            node_list = self.get_min_node_ratio_index(self.out_ratio)
            (add_gain, add_node_ind) = self.min_index_edge_addition(
                node_list, self.deg, self.out_deg
            )
            add_node = self.community_target[add_node_ind]
            add_node_2 = self.find_external_node(
                add_node,
                self.community_target,
                self.community_obj.communities,
                self.IG_edgeList,
            )
            li = self.get_best_del_excl_bridges(
                self.community_target, self.new_edge_list, self.new_adj, num_vertices
            )
            ((del_node, del_node_2), max_gain) = self.deletion_gain(
                li, intra_considered, self.deg, self.out_deg, self.community_target
            )
            del_gain = max_gain
            if add_gain >= del_gain and add_gain > 0:
                self.IG_edgeList.append((add_node, add_node_2))
                # print("Safeness Added edge: ", (add_node, add_node_2))
                for i in self.community_target:
                    deg_ = 0
                    out_deg_ = 0
                    for j in self.IG_edgeList:
                        if i == j[0] or i == j[1]:
                            deg_ = deg_ + 1
                            if (i == j[0] and j[1] not in self.community_target) or (
                                i == j[1] and j[0] not in self.community_target
                            ):
                                out_deg_ = out_deg_ + 1
                    self.deg[self.community_target.index(i)] = deg_
                    self.out_deg[self.community_target.index(i)] = out_deg_

                for i, _ in enumerate(self.out_ratio):
                    self.out_ratio[i] = self.out_deg[i] / self.deg[i]

            elif del_gain > 0:
                self.IG_edgeList.remove((del_node, del_node_2))
                intra_considered.append((del_node, del_node_2))
                # print("Safeness Removed edge: ", (del_node, del_node_2))
                for i in self.community_target:
                    deg_ = 0
                    out_deg_ = 0
                    for j in self.IG_edgeList:
                        if i == j[0] or i == j[1]:
                            deg_ = deg_ + 1
                            if (i == j[0] and j[1] not in self.community_target) or (
                                i == j[1] and j[0] not in self.community_target
                            ):
                                out_deg_ = out_deg_ + 1
                    self.deg[self.community_target.index(i)] = deg_
                    self.out_deg[self.community_target.index(i)] = out_deg_

                for i in enumerate(self.out_ratio):
                    self.out_ratio[i] = self.out_deg[i] / self.deg[i]

                self.new_edge_list.remove((del_node, del_node_2))
                self.new_adj[del_node].remove(del_node_2)
                self.new_adj[del_node_2].remove(del_node)

            beta = beta - 1

            if beta > 0 and (add_gain > 0 or del_gain > 0):
                continue
            else:
                break
        # Build the new graph
        new_graph = nx.Graph()
        new_graph.add_nodes_from(self.graph.nodes())
        new_graph.add_edges_from(self.IG_edgeList)
        return new_graph, self.budget - beta

    def get_min_node_ratio_index(self, out_ratio: List[float]):
        """
        Finds the node with minimum out ratio

        Parameters
        ----------
        out_ratio : List[float]
            List of node ratios

        Returns
        -------
        List[int]
            List of node indices with minimum out ratio
        """
        min_val = min(out_ratio)
        node = []
        for i, _ in enumerate(out_ratio):
            if out_ratio[i] == min_val:
                node.append(i)
        return node

    def min_index_edge_addition(self, node_list, deg, out_deg):
        """
        Finds the node with minimum edge addition gain

        Parameters
        ----------
        node_list : List[int]
            List of node indices
        deg : List[int]
            Degree of nodes
        out_deg : List[int]
            Degree of outgoing edges

        Returns
        -------
        Tuple[float, int]
            Tuple of minimum edge addition gain and node index
        """
        node_ind = 0
        max_gain = 0
        for i in node_list:
            gain = 0.5 * ((out_deg[i] + 1) / (deg[i] + 1) - out_deg[i] / deg[i])
            if gain > max_gain:
                max_gain = gain
                node_ind = i
        return (max_gain, node_ind)

    def find_external_node(self, com_node, com, graph, edges):
        """
        Finds a node (not in C) such that the edge (np, nt) does not exist

        Parameters
        ----------
        com_node : int
            Source node, node in the target community
        com : List[int]
            Target community
        graph : nx.Graph
            Graph
        edges : List[Tuple[int, int]]
            List of edges

        Returns
        -------
        j : int
            Node not in C such that the edge (np, nt) does not exist
        """
        for i in graph:
            if i != com:
                for j in i:
                    if ((com_node, j) or (j, com_node)) not in edges:
                        return j

        raise Exception("No external node found")

    def deletion_gain(
        self, edges, intra_considered, degrees, out_degrees, target_community
    ) -> Tuple[Tuple[int, int], float]:
        """
        Compute the deletion gain for each edge in the list of edges.

        Parameters
        ----------
        edges : List[Tuple[int, int]]
            List of edges in the graph.
        intra_considered : Set[Tuple[int, int]]
            Set of edges already considered.
        degrees : List[int]
            List of degrees of each node in the graph.
        out_degrees : List[int]
            List of out-degrees of each node in the target community.
        target_community : List[int]
            List of integers representing the community membership of each node.

        Returns
        -------
        Tuple[Tuple[int, int], float]
            The edge and its corresponding deletion gain with the highest gain.
        """
        max_gain = 0
        node_u = 0
        node_v = 0
        for edge in edges:
            if edge not in intra_considered:
                u = edge[0]
                v = edge[1]
                gain = (
                    (
                        out_degrees[target_community.index(u)]
                        / (
                            2
                            * degrees[target_community.index(u)]
                            * (degrees[target_community.index(u)] - 1)
                        )
                    )
                    + (
                        out_degrees[target_community.index(v)]
                        / (
                            2
                            * degrees[target_community.index(v)]
                            * (degrees[target_community.index(v)] - 1)
                        )
                    )
                    + (1 / (len(target_community) - 1))
                )
                if gain > max_gain:
                    max_gain = gain
                    node_u = u
                    node_v = v
        return ((node_u, node_v), max_gain)

    def get_best_del_excl_bridges(
        self,
        target_comm: List[int],
        edges: List[Tuple[int, int]],
        adjacency_list: List[List[int]],
        num_vertices: int,
    ) -> List[Tuple[int, int]]:
        """
        Returns the list of edges that, if removed, would disconnect the graph
        and leave only one connected component.

        Parameters
        ----------
        target_comm : List[int]
            List of integers representing the community membership of each node.
        edges : List[Tuple[int, int]]
            List of edges in the graph.
        adjacency_list : List[List[int]]
            The adjacency list of the graph.
        num_vertices : int
            The number of vertices in the graph.

        Returns
        -------
        List[Tuple[int, int]]
            The list of edges that, if removed, would disconnect the graph and
            leave only one connected component.
        """
        best_edges = []
        for edge in edges:
            adj_list_copy = copy.deepcopy(adjacency_list)
            adj_list_copy[edge[0]].remove(edge[1])
            adj_list_copy[edge[1]].remove(edge[0])
            try:
                if (
                    self.connected_components(target_comm, num_vertices, adj_list_copy)
                    == 1
                ):
                    best_edges.append(edge)
            except:
                continue
        return best_edges

    def dfs_util(
        self,
        target_comm: List[int],
        temp: List[int],
        v: int,
        visited: List[bool],
        adjacency_list: List[List[int]],
        excluded_edge: Tuple[int, int] = None,
    ) -> List[int]:
        """
        Utility function for depth-first search algorithm.

        Parameters
        ----------
        target_comm : List[int]
            List of integers representing the community membership of each node.
        temp : List[int]
            List of nodes visited during the search.
        v : int
            The current node being visited.
        visited : List[bool]
            List of boolean values indicating whether a node has been visited or not.
        adjacency_list : List[List[int]]
            The adjacency list of the graph.

        Returns
        -------
        List[int]
            The list of nodes visited during the search.
        """
        # Set current node as visited, to avoid infinite loops
        visited[v] = True
        temp.append(v)
        for i in adjacency_list[target_comm[v]]:
            if not visited[target_comm.index(i)]:
                temp = self.dfs_util(
                    target_comm, temp, target_comm.index(i), visited, adjacency_list
                )
        return temp

    def connected_components(
        self,
        target_comm: List[int],
        num_vertices: int,
        adjacency_list: List[List[int]],
        excluded_edge: Tuple[int, int] = None,
    ) -> int:
        """
        Compute the number of connected components in a graph.

        Parameters
        ----------
        target_comm : List[int]
            List of integers representing the community membership of each node.
        num_vertices : int
            The number of vertices in the graph.
        adjacency_list : List[List[int]]
            The adjacency list of the graph.
        excluded_edge : Tuple[int, int], optional
            An edge to be excluded from the graph, by default None

        Returns
        -------
        int
            The number of connected components in the graph.
        """
        visited = [False] * num_vertices
        # List of connected components
        cc = []
        for v, _ in enumerate(num_vertices):
            if not visited[v]:
                temp = []
                cc.append(self.dfs_util(target_comm, temp, v, visited, adjacency_list))
        return len(cc)

    def vertices_in_connected_components(
        self, target_comm, num_vertices, adjacency_list, node
    ):
        visited = []
        cc = []
        for i in range(num_vertices):
            visited.append(False)
        for v in range(num_vertices):
            if visited[v] == False:
                temp = []
                cc.append(self.dfs_util(target_comm, temp, v, visited, adjacency_list))
        cc_node_list = []
        for i in cc:
            tmp = []
            for j in i:
                tmp.append(target_comm[j])
            cc_node_list.append(tmp)

        for i in cc_node_list:
            if node in i:
                return len(i)
        return 0

    ############################################################################
    #                PRE-COMPUTING FOR COMMUNITY DECEPTION                     #
    ############################################################################
    def pre_computation(self):
        """
        Function to compute the pre-computation for the Safeness algorithm, to
        speed up the Safeness execution.
        """
        e_ = list(self.graph.edges())
        adjacency_list = self.get_adj_list(e_)
        num_vertices = len(self.graph.nodes())
        self.IG_edgeList = []
        for i in e_:
            self.IG_edgeList.append((i[0], i[1]))
        g = ig.Graph(directed=False)
        g.add_vertices(num_vertices)
        g.add_edges(self.IG_edgeList)

        # Get the communities
        communities = self.community_obj.communities

        # pre_neighbours, pre_marked = self.get_target_comm_neighbours(
        #     self.community_target, communities, adjacency_list
        # )

        self.deg = []
        for i in self.community_target:
            self.deg.append(g.vs[i].degree())

        self.out_deg = []
        for i in self.community_target:
            out = 0
            for j in adjacency_list[i]:
                if j not in self.community_target:
                    out = out + 1
            self.out_deg.append(out)

        self.out_ratio = []
        for j, _ in enumerate(self.out_deg):
            self.out_ratio.append(self.out_deg[j] / self.deg[j])

        self.new_adj = {}
        for j in adjacency_list.keys():
            if j in self.community_target:
                self.new_adj[j] = []
                for k in adjacency_list[j]:
                    if k in self.community_target:
                        self.new_adj[j].append(k)

        self.new_edge_list = []
        for j in self.IG_edgeList:
            if j[0] in self.community_target and j[1] in self.community_target:
                self.new_edge_list.append(j)

    def num_comm(
        self, target_comm: List[int], communities: List[List[int]]
    ) -> Tuple[int, List]:
        """
        Find the communities in which the nodes of the target community are present.

        Parameters
        ----------
        target_comm : List[int]
            Target community, list of nodes
        communities : List[List[int]]
            List of communities, each community is a list of nodes

        Returns
        -------
        len(uni_comm) : int
            Number of communities in which the nodes of the target community are present
        comm_list : List[List[int]]
            List of communities in which the nodes of the target community are present

        """
        uni_comm = []
        comm_list = []
        for node in target_comm:
            for c in communities:
                if node in c:
                    comm_list.append(c)
                    if c not in uni_comm:
                        uni_comm.append(c)
                        break
        return len(uni_comm), comm_list

    def get_target_comm_neighbours(
        self,
        target_comm: List[int],
        communities: List[List[int]],
        adjacency_list: List[int],
    ) -> Tuple[List[int], List[int]]:
        """
        Given a target community, a list of communities, and an adjacency list,
        returns a list of the indices of the communities that have at least one
        neighbor in the target community, as well as a dictionary of marked nodes.

        Parameters
        ----------
        target_comm : List[int]
            A list of nodes representing the target community
        communities : List[List[int]]
            A list of lists, where each list contains the nodes of a community
        adjacency_list : dict
            A dictionary where the keys are nodes and the values are lists of the neighbors of the corresponding node

        Returns
        -------
        List : List
            A list of integers representing the indices of the communities that have at least one neighbor in the target community
        Marked : dict
            A dictionary where the keys are nodes that have been marked and the values are the same as the keys
        """
        neighbors_list = []
        marked = dict()
        for node in target_comm:
            for j in adjacency_list[node]:
                if j not in marked:
                    for k, _ in enumerate(communities):
                        if j in communities[k]:
                            neighbors_list.append(k)
                            marked[j] = j
        return neighbors_list, marked

    def check_neighbours(self, neighbours: List[int], communities: List[List[int]]):
        """
        Given a list of node IDs `neighbours` and a list of communities `communities`,
        returns a list of indices of the communities that contain at least one of the
        nodes in `neighbours`.

        Parameters
        ----------
        neighbours : List[int]
            A list of node IDs.
        communities : List[List[int]]
            A list of lists, where each inner list contains the node IDs of a community.

        Returns
        -------
        list
            A list of integers representing the indices of the communities that contain
            at least one of the nodes in `neighbours`.
        """
        ctr = 0
        community_list = []
        for i, community in enumerate(communities):
            for node in community:
                if node in neighbours:
                    community_list.append(community)
                    ctr += 1
                if ctr == len(neighbours):
                    return community_list
        return community_list

    def get_entropy(self, labels, base=None):
        """
        Calculate the entropy of a given set of labels.

        Parameters
        ----------
        labels : array_like
            A 1-D array of labels.
        base : float, optional
            The logarithmic base to use for calculating the entropy. Default is None, which uses the natural logarithm.

        Returns
        -------
        float
            The entropy of the label distribution.
        """
        values, counts = np.unique(labels, return_counts=True)
        return entropy(counts, base=base)

    def get_adj_list(self, edge_list: List[Tuple[int, int]]):
        """
        Returns the adjacency list of a graph given its edge list.

        Parameters
        ----------
        edge_list : list of tuples
            A list of tuples representing the edges of the graph.

        Returns
        -------
        dict
            A dictionary where the keys are the nodes of the graph and the values are lists of the nodes adjacent to the key node.
        """
        adjacency_list = {}
        for i, _ in enumerate(edge_list):
            edge = edge_list[i]
            # Source Node
            s = edge[0]
            # Target Node
            t = edge[1]
            if s in adjacency_list:
                adjacency_list[s].append(t)
            else:
                adjacency_list[s] = [t]
            if t in adjacency_list:
                adjacency_list[t].append(s)
            else:
                adjacency_list[t] = [s]
        return adjacency_list


if __name__ == "__main__":
    # Load graph from edgelist from file
    graph = nx.read_edgelist(path="dataset/data/fb-75.txt", nodetype=int)
    graph = nx.convert_node_labels_to_integers(
        graph, first_label=0, ordering="sorted", label_attribute="node_type"
    )
    for node in graph.nodes:
        # graph.nodes[node]['name'] = node
        graph.nodes[node]["num_neighbors"] = len(list(graph.neighbors(node)))

    def get_deception_score(
        graph, community_structure: List[List[int]], community_target: List[int]
    ):
        number_communities = len(community_structure)

        # Number of the target community members in the various communities
        member_for_community = np.zeros(number_communities, dtype=int)

        for i in range(number_communities):
            for node in community_structure[i]:
                if node in community_target:
                    member_for_community[i] += 1

        # ratio of the targetCommunity members in the various communities
        ratio_community_members = [
            members_for_c / len(com)
            for (members_for_c, com) in zip(member_for_community, community_structure)
        ]

        # In how many commmunities are the members of the target spread?
        spread_members = sum([1 if mc > 0 else 0 for mc in member_for_community])

        second_part = 1 / 2 * ((spread_members - 1) / number_communities) + 1 / 2 * (
            1 - sum(ratio_community_members) / spread_members
        )

        # induced subraph only on target community nodes
        num_components = nx.number_connected_components(
            graph.subgraph(community_target)
        )

        denominator = len(community_target) - 1
        if denominator == 0:
            denominator = 1
        first_part = 1 - ((num_components - 1) / denominator)
        dec_score = first_part * second_part
        return dec_score

    def get_communities(graph):
        # Compute communities structure with igraph
        # First, convert the graph to igraph
        ig_graph = ig.Graph.from_networkx(graph)
        # Then, compute the communities
        communities = ig_graph.community_fastgreedy().as_clustering()
        com_list = [c for c in communities]
        # Create a NodeClustering object
        communities_obj = cdlib.NodeClustering(communities=com_list, graph=graph)
        return communities_obj

    communities_obj = get_communities(graph)
    # Choose the target community
    communities = communities_obj.communities
    max_nodes = max([len(c) for c in communities])
    # Filter the communities with the maximum number of nodes

    # MODE 2
    communities_len = [len(c) for c in communities_obj.communities]
    preferred_size = int(np.ceil(max(communities_len) * 0.5)) / 2
    # Find the 10 communities with the closest size to the preferred size
    communities_len = np.array(communities_len)
    array = np.asarray(communities_len)
    idx = (np.abs(communities_len - preferred_size)).argsort()[:10]
    nearest_communities = [communities[i] for i in idx]

    print("Number of communities: ", len(nearest_communities))
    for i in range(10):
        print("-------- Iteration: ", i)
        community_target = random.choice(nearest_communities)

        print("Target Community Length: ", len(community_target))
        safeness = Safeness(
            budget=1,
            graph=graph.copy(),
            community_target=community_target,
            communities_object=communities_obj,
        )

        graph, steps = safeness.run()

        communities_obj = get_communities(graph)
        communities = communities_obj.communities

        # print deceptio score
        print(
            "Deception Score: ",
            get_deception_score(graph, communities, community_target),
        )
