class HeuristicSolver:
    """
    A heuristic solver for the Traveling Salesman Problem using the Nearest Neighbor heuristic.
    """

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
        # Obtain a list of nodes from the graph
        nodes = list(graph.get_nodes())
        if not nodes:
            return {"tour": [], "cost": 0}

        # Start at an arbitrary node (here, the first node in the list)
        start = nodes[0]
        tour = [start]
        unvisited = set(nodes)
        unvisited.remove(start)
        current = start

        # Iteratively visit the nearest unvisited neighbor
        while unvisited:
            next_node = min(unvisited, key=lambda node: graph.get_weight(current, node))
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        # Complete the tour by returning to the starting node
        tour.append(start)

        # Compute the total cost of the tour
        total_cost = sum(graph.get_weight(tour[i], tour[i + 1]) for i in range(len(tour) - 1))

        return {"tour": tour, "cost": total_cost}