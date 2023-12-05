from calendar import c
from math import e
import random
from re import sub
import networkx as nx
import cdlib
from typing import List, Tuple, Dict, Union

from concurrent.futures import ThreadPoolExecutor

from sympy import sec


class Modularity:
    """Class that implements the Modularity algorithm"""

    def __init__(
        self,
        beta: int,
        graph: nx.Graph,
        community_target: List[int],
        communities_object: cdlib.NodeClustering,
        detection_alg: callable,
    ):
        self.beta = beta
        self.graph = graph.copy()
        self.community_target = community_target
        self.community_obj = communities_object
        self.detection_alg = detection_alg

        self.old_modularity = nx.community.modularity(
            graph, communities_object.communities
        )

        # Randomly select an intra-community edge
        self.intra_community_edges = [
            edge
            for edge in self.graph.edges()
            if edge[0] in self.community_target and edge[1] in self.community_target
        ]

    def process_edge(self, edge):
        gain, communities_del, mod_after_del = self.get_del_loss(edge[0], edge[1])
        return edge, {
            "gain": gain,
            "communities": communities_del,
            "mod_after": mod_after_del,
        }

    def compute_and_sort_com_degrees(
        self, graph: nx.Graph, communities: List[List[int]]
    ) -> List[Tuple[int, Dict[str, Union[List[int], int]]]]:
        """
        Compute and sort the degree of each community in the given graph.

        Parameters
        ----------
        graph : nx.Graph
            The input graph.
        communities : List[List[int]]
            The list of communities.

        Returns
        -------
        List[Tuple[int, Dict[str, Union[List[int], int]]]]
            The list of communities sorted by their degree, where each
            community is represented as a tuple containing the community
            index and a dictionary with the community nodes and their degree.
        """
        community_degree = {}
        for i, community in enumerate(communities):
            community_degree[i] = {
                "community": community,
                "degree": sum(graph.degree(n) for n in community),
            }
        community_degree = sorted(
            community_degree.items(), key=lambda x: x[1]["degree"], reverse=True
        )
        return community_degree

    def get_eta(self):
        # Given the community structure, compute for each community the sum
        # of internal and external edges
        eta = 0
        for community in self.community_obj.communities:
            # Compute subgraph of the community
            subgraph = self.graph.subgraph(community)
            # Compute the number of edges in the subgraph
            num_edges = subgraph.number_of_edges()
            # Compute the external edges of the community
            for node in community:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in community:
                        num_edges += 1
            eta += num_edges
        return eta

    def get_delta(self):
        delta = 0
        for community in self.community_obj.communities:
            delta += sum(self.graph.degree(node) ** 2 for node in community)
        return delta

    def get_add_loss_fast(self, Ci, Cj) -> float:
        Ci_deg = sum(self.graph.degree(n) for n in Ci)
        Cj_deg = sum(self.graph.degree(n) for n in Cj)
        m = self.graph.number_of_edges()
        first_component = self.get_eta() / m * (m + 1)
        second_component = (
            2 * m**2 * (Ci_deg + Cj_deg + 1) - self.get_delta() * (2 * m + 1)
        ) / (4 * m**2 * (m + 1) ** 2)
        return first_component + second_component

    def get_del_loss_fast(self, Ci):
        Ci_deg = (
            sum(self.graph.degree(n) for n in Ci) - 2
        )  # subtract the edge to remove
        m = self.graph.number_of_edges()
        first_component = m - self.get_eta() / m * (m - 1)
        second_component = (
            self.get_delta() * (2 * m - 1) - 4 * m**2 * (Ci_deg - 1)
        ) / (4 * m**2 * (m - 1) ** 2)
        return first_component + second_component

    def get_add_loss(
        self, np: int, nt: int
    ) -> Tuple[float, cdlib.NodeClustering, float]:
        """
        Computes the modularity gain, new communities and new modularity
        after adding an edge between two nodes.

        Parameters
        ----------
        np : int
            The index of the first node.
        nt : int
            The index of the second node.

        Returns
        -------
        Tuple[float, cdlib.NodeClustering, float]
            A tuple containing the modularity gain, the new communities and
            the new modularity.
        """
        graph = self.graph.copy()
        graph.add_edge(np, nt)
        communities_after = self.detection_alg.compute_community(graph)
        mod_after = nx.community.modularity(graph, communities_after.communities)
        gain = mod_after - self.old_modularity
        graph.remove_edge(np, nt)
        return gain, communities_after, mod_after

    def get_del_loss(
        self, nk: int, nl: int
    ) -> Tuple[float, cdlib.NodeClustering, float]:
        """
        Compute the modularity gain obtained by removing the edge between nodes nk and nl.

        Parameters
        ----------
        nk : int
            The first node of the edge to remove.
        nl : int
            The second node of the edge to remove.

        Returns
        -------
        Tuple[float, cdlib.NodeClustering, float]
            A tuple containing the modularity gain obtained by removing the edge, the new node clustering, and the new modularity value.
        """
        graph = self.graph.copy()
        mod_before = self.old_modularity

        graph.remove_edge(nk, nl)
        communities_after = self.detection_alg.compute_community(graph)
        mod_after = nx.community.modularity(graph, communities_after.communities)

        gain = mod_after - mod_before
        graph.add_edge(nk, nl)
        return gain, communities_after, mod_after

    def run(self) -> Tuple[nx.Graph, int, cdlib.NodeClustering]:
        """
        Executes the community hiding algorithm on the input graph.

        Returns
        -------
        Tuple[nx.Graph, int, cdlib.NodeClustering] : [graph, iterations, communities]
            A tuple containing the modified graph, the number of iterations
            performed, and the resulting node clustering.
        """
        graph = self.graph
        beta = self.beta
        communities = self.community_obj

        while beta > 0:
            deg_C = self.compute_and_sort_com_degrees(graph, communities.communities)
            MLadd = -1
            if len(deg_C) > 1:
                Ci = deg_C[0][1]["community"]
                Cj = deg_C[1][1]["community"]
                # Find the best edge to add
                np, nt = next(
                    (
                        (np, nt)
                        for np in Ci
                        for nt in Cj
                        if np != nt and not graph.has_edge(np, nt)
                    ),
                    (None, None),
                )
                if np is not None and nt is not None:
                    MLadd = self.get_add_loss_fast(Ci, Cj)
                    # MLadd, communities_add, mod_after_add = self.get_add_loss(np, nt)

            MLdel = self.get_del_loss_fast(self.community_target)

            if MLdel >= MLadd and MLdel > 0:
                # Randomly select an intra-community edge
                subgraph = graph.subgraph(self.community_target)
                nk, nl = random.choice(list(subgraph.edges()))
                graph.remove_edge(nk, nl)
                self.old_modularity += MLdel
            elif MLadd > 0:
                graph.add_edge(np, nt)
                self.old_modularity += MLadd

            beta -= 1
            if MLadd <= 0 and MLdel <= 0:
                break

        # Compute the new communities
        communities = self.detection_alg.compute_community(graph)
        return graph, self.beta - beta, communities
