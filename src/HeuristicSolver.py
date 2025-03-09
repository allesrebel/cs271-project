import random


class HeuristicSolver:
    """
    A heuristic solver for the Traveling Salesman Problem using the Nearest Neighbor heuristic.
    """

    def set_meta_config():
        pass

    def __init__(self):
        pass

    def solve(self, graph):
        """
        Solves the TSP using the Nearest Neighbor heuristic.

        Parameters:
            graph: A TSP instance (e.g. loaded using tsplib95) that provides:
                   - get_nodes() returning the set/list of nodes
                   - get_weight(i, j) returning the distance between nodes i and j.

        Returns:
            A dictionary with:
              - "tour": A list representing the order in which nodes are visited,
                        starting and ending at the same node.
              - "cost": The total cost (distance) of the computed tour.
        """

        # a list of nodes we've got to process
        unvisited = set( graph.get_nodes() )

        distance = 0
        current = random.choice(tuple(unvisited))
        tour = [current]
        unvisited.remove(current)

        # Iteratively visit the nearest unvisited neighbor
        while unvisited:
            next_node = min(unvisited, key=lambda node: graph.get_weight(current, node))
            distance += graph.get_weight(current, next_node)
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        # Complete the tour by returning to the starting node
        tour.append(tour[0])
        distance += graph.get_weight(current, tour[0])

        return {"tour": tour, "cost": distance, "meta": vars(self)}
