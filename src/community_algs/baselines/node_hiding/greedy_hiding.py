# import sys
# sys.path.append("../../../")
from src.environment.graph_env import GraphEnvironment
from src.utils.utils import DetectionAlgorithmsNames, Utils, FilePaths
from src.community_algs.detection_algs import CommunityDetectionAlgorithm

import networkx as nx
from typing import List, Callable, Tuple
import random
import copy


class GreedyHiding:
    def __init__(self, env: GraphEnvironment, steps: int, target_community: List[int]):
        self.env = env
        self.graph = self.env.original_graph
        self.steps = steps
        self.target_node = self.env.node_target
        self.target_community = target_community
        self.detection_alg = self.env.detection
        self.original_community_structure = copy.deepcopy(
            self.env.original_community_structure
        )
        self.possible_edges = self.get_possible_action()

        self.alpha_metric = 0.7

    def get_intra_community_node(
        self,
        community: List[int],
        graph: nx.Graph,
    ) -> List[int]:
        """
        For each node in the community, different from the target node,
        compute the intra-community degree, and return the node with the
        highest intra-community degree.

        Parameters
        ----------
        community : List[int]
            Community of the target node
        graph : nx.Graph

        Returns
        -------
        node : int
            Node with the highest intra-community degree
        """
        # Set the max degree to - infinity
        max_degree = -float("inf")
        node = None

        # Get the neighbors of the target node in the community
        neighbors = list(graph.neighbors(self.target_node))
        community_neighbors = [n for n in neighbors if n in community]

        for n in community_neighbors:
            if n == self.target_node:
                continue

            # Get the neighbors of the node n
            neighbors = list(graph.neighbors(n))

            # Compute the intra-community degree
            intra_degree = sum(1 for neighbor in neighbors if neighbor in community)

            # Normalize the intra-community degree
            intra_degree /= len(neighbors)

            # Check if the intra-community degree is higher than the current max
            if intra_degree > max_degree:
                max_degree = intra_degree
                node = n
        return node

    def get_inter_community_node(
        self, community: List[int], graph: nx.Graph
    ) -> List[int]:
        """
        Get the inter-community node with the highest degree.

        Parameters
        ----------
        community : List[int]
            Community of the target node

        Returns
        -------
        int
            Node with the highest degree
        """
        # Get the inter-community nodes
        inter_community_nodes = set(graph.nodes()) - set(community)

        # Remove the neighbors of the target node from the list
        inter_community_nodes -= set(graph.neighbors(self.target_node))

        # Get the node with the highest degree
        node = max(inter_community_nodes, key=lambda x: graph.degree(x))
        return node

    def compute_loss(
        self,
        original_community: List[List[int]],
        new_community: List[List[int]],
        original_graph: nx.Graph,
        new_graph: nx.Graph,
    ) -> float:
        """
        Compute the loss between the original and the new community structure

        Parameters
        ----------
        original_community : List[List[int]]
            Original community structure
        new_community : List[List[int]]
            New community structure
        original_graph : nx.Graph
            Original graph
        new_graph : nx.Graph

        Returns
        -------
        loss : float
            Loss between the original and the new community structure
        """
        community_distance = (
            1 - new_community.normalized_mutual_information(original_community).score
        )
        graph_distance = self.env.graph_similarity(original_graph, new_graph)
        loss = (
            self.alpha_metric * community_distance
            + (1 - self.alpha_metric) * graph_distance
        )
        return loss

    def get_possible_action(self):
        # Put all edge between the target node and its neighbors in a list
        possible_actions_remove = []
        for neighbor in self.graph.neighbors(self.target_node):
            possible_actions_remove.append((self.target_node, neighbor))

        # Put all the edges that aren't neighbors of the target node in a list
        possible_actions_add = []
        for node in self.graph.nodes():
            if node != self.target_node and node not in self.graph.neighbors(
                self.target_node
            ):
                possible_actions_add.append((self.target_node, node))
        possible_action = possible_actions_add + possible_actions_remove
        return possible_action

    def hide_target_node_from_community(self) -> Tuple[nx.Graph, List[int], int]:
        """
        Hide the target node from the target community by rewiring its edges,
        choosing the node with the highest degree between adding or removing an edge.

        Returns
        -------
        Tuple[nx.Graph, List[int], int]
            The new graph, the new community structure and the number of steps
        """
        graph = self.graph.copy()
        communities = self.original_community_structure
        steps = self.steps
        target_community = self.target_community.copy()

        while steps > 0:
            # Get the inter-community node with the highest degree, (add edge)
            candidate_1 = self.get_inter_community_node(target_community, graph)

            if candidate_1 is not None:
                # Compute the community structure after adding the edge, and restore the graph
                graph_1 = graph.copy()
                graph_1.add_edge(self.target_node, candidate_1)
                communities_1 = self.detection_alg.compute_community(graph)
                # Compute the loss with the new community structures
                loss_1 = self.compute_loss(
                    communities,
                    communities_1,
                    graph,
                    graph_1,
                )

            # Get the intra-community node with the highest degree, (remove edge)
            candidate_2 = self.get_intra_community_node(target_community, graph)
            if candidate_2 is not None:
                # Compute the community structure after removing the edge, and restore the graph
                graph_2 = graph.copy()
                graph_2.remove_edge(self.target_node, candidate_2)
                communities_2 = self.detection_alg.compute_community(graph)
                loss_2 = self.compute_loss(
                    communities,
                    communities_2,
                    graph,
                    graph_2,
                )

            if candidate_1 is None:
                graph = graph_2
                communities = communities_2
                target_community = self.get_new_community(communities_2)
            elif candidate_2 is None:
                graph = graph_1
                communities = communities_1
                target_community = self.get_new_community(communities_1)
            elif candidate_1 is None and candidate_2 is None:
                break

            if loss_1 < loss_2:
                graph = graph_1
                communities = communities_1
                target_community = self.get_new_community(communities_1)
            else:
                graph = graph_2
                communities = communities_2
                target_community = self.get_new_community(communities_2)

            steps -= 1

            if len(target_community) < 2:
                break

        step = self.steps - steps
        return graph, communities, step

    def get_new_community(self, new_community_structure: List[List[int]]) -> List[int]:
        """
        Search the community target in the new community structure after
        deception. As new community target after the action, we consider the
        community that contains the target node, if this community satisfies
        the deception constraint, the episode is finished, otherwise not.

        Parameters
        ----------
        node_target : int
            Target node to be hidden from the community
        new_community_structure : List[List[int]]
            New community structure after deception

        Returns
        -------
        List[int]
            New community target after deception
        """
        if new_community_structure is None:
            # The agent did not perform any rewiring, i.e. are the same communities
            return self.target_community
        for community in new_community_structure.communities:
            if self.target_node in community:
                return community
        raise ValueError("Community not found")


# Example usage:
if __name__ == "__main__":
    pass
    # Import karate club graph
    # env = GraphEnvironment(graph_path=FilePaths.KAR.value)

    # detection_alg_name = DetectionAlgorithmsNames.GRE.value
    # detection_alg = CommunityDetectionAlgorithm(detection_alg_name)
    # communities = detection_alg.compute_community(graph)

    # # Choose randomly a community
    # community = communities.communities[
    #     random.randint(0, len(communities.communities) - 1)
    # ]
    # # Choose randomly a node from the community
    # node = community[random.randint(0, len(community) - 1)]

    # edge_budget = graph.number_of_edges() * 0.1

    # random_hiding = GreedyHiding(graph, edge_budget,community)

    # new_graph = random_hiding.hide_target_node_from_community()

    # # Compute the new community structure
    # new_communities = detection_alg.compute_community(new_graph)

    # print("Original community: ", communities.communities)
    # print("New community: ", new_communities.communities)
