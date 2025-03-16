import time

class LocalSearchSolver:
    """
    A solver for the Traveling Salesman Problem using local search.
    This implementation generates neighbors by swapping two nodes in the tour.
    """

    def set_meta_config(self):
        pass

    def __init__(self):
        pass

    def solve(self, graph, timestart, timelimit=None):
        """
        Solves the TSP using a local search algorithm with two-node swaps.

        Parameters:
            graph: A TSP instance (e.g. loaded using tsplib95) that provides:
                   - get_nodes() returning the set/list of nodes.
                   - get_weight(i, j) returning the distance between nodes i and j.

        Returns:
            A dictionary with:
              - "tour": A list representing the order in which nodes are visited,
                        starting and ending at the same node.
              - "cost": The total cost (distance) of the computed tour.
              - "meta": Additional metadata about the solver (vars(self)).
        """
        # Just to keep things finishing in a reasonable timeline
        # we'll kill all jobs that take longer than 10 mins (Since they'd take too long
        # in a realworld applications)
        if not timelimit:
            timelimit = 10 * 60

        # Get nodes and create a stable ordering.
        nodes = list(graph.get_nodes())
        n = len(nodes)
        if n == 0:
            return {"tour": [], "cost": 0, "meta": vars(self)}

        # Create an initial tour.
        # We fix the starting node at index 0.
        current_tour = nodes[:]  # initial order as given
        # Helper: compute the total cost of a tour (cyclic: returning to the start)
        def compute_cost(tour):
            cost = 0
            for i in range(len(tour) - 1):
                cost += graph.get_weight(tour[i], tour[i+1])
            # add the cost to return to the starting node
            cost += graph.get_weight(tour[-1], tour[0])
            return cost

        current_cost = compute_cost(current_tour)
        improvement = True

        # Local search: try swapping two nodes (except the first, which is fixed)
        while improvement:
            # enforce time limit
            if timelimit and (timelimit < time.time() - timestart):
                break;

            improvement = False
            # Iterate over all pairs (i, j) with i < j, starting from 1
            for i in range(1, n):
                for j in range(i + 1, n):
                    # Create a new tour by swapping nodes at positions i and j.
                    new_tour = current_tour[:]
                    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                    new_cost = compute_cost(new_tour)
                    # If the new tour improves the cost, accept it and restart search.
                    if new_cost < current_cost:
                        current_tour = new_tour
                        current_cost = new_cost
                        improvement = True
                        # Break out to restart scanning for improvements.
                        break
                if improvement:
                    break

        # Complete the cycle by appending the starting node at the end.
        final_tour = current_tour + [current_tour[0]]

        return {"tour": final_tour, "cost": current_cost, "meta": vars(self)}
