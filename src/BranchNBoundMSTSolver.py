import math
import heapq
import time

class BranchAndBoundMSTSolver:
    """
    A solver for the Traveling Salesman Problem using a Branch-and-Bound approach
    with an MST-based bounding function.

    Adapted from https://www.sciencedirect.com/science/article/pii/S1572528616000062
    """

    def set_meta_config(self):
        pass

    def __init__(self):
        pass

    def solve(self, graph, timestart, timelimit=None):
        """
        Solves the TSP using a Branch-and-Bound approach with an MST-based bound.

        Parameters:
            graph: A TSP instance (e.g. loaded using tsplib95) that provides:
                   - get_nodes() returning the set/list of nodes
                   - get_weight(i, j) returning the distance between nodes i and j.

        Returns:
            A dictionary with:
              - "tour": A list representing the order in which nodes are visited,
                        starting and ending at the same node.
              - "cost": The total cost (distance) of the computed tour.
              - "meta": Additional metadata about the solver (vars(self)).
        """

        # Convert the node set to a list for stable indexing
        nodes = list(graph.get_nodes())
        n = len(nodes)
        if n == 0:
            return {"tour": [], "cost": 0, "meta": vars(self)}

        # Build mappings between node labels and their indices
        node_to_index = {node: i for i, node in enumerate(nodes)}
        index_to_node = {i: node for i, node in enumerate(nodes)}

        # Helper to get the distance between two *indices*
        def distance(i, j):
            return graph.get_weight(index_to_node[i], index_to_node[j])

        # For convenience, we treat the first node in `nodes` as the TSP start
        start_node_index = 0

        # ---------------------------------------------------------------------
        # 1) Compute an initial feasible solution for an upper bound
        #    (we’ll just visit nodes in the trivial order 0..n-1).
        # ---------------------------------------------------------------------
        def compute_path_cost(path_indices):
            cost_ = 0
            for k in range(len(path_indices) - 1):
                cost_ += distance(path_indices[k], path_indices[k+1])
            # Add the cost to return to start
            cost_ += distance(path_indices[-1], path_indices[0])
            return cost_

        trivial_path = list(range(n))  # [0, 1, 2, ..., n-1]
        best_cost = compute_path_cost(trivial_path)
        best_path_indices = trivial_path[:]

        # ---------------------------------------------------------------------
        # 2) MST-based bounding function
        #    bound = current_cost + MST_of_unvisited + link_from_current + link_to_start
        # ---------------------------------------------------------------------
        def prim_mst_cost(unvisited_list):
            """Compute MST cost (Prim’s algorithm) over the given unvisited indices."""
            if len(unvisited_list) <= 1:
                return 0

            in_mst = {unvisited_list[0]}
            total_mst_cost = 0
            # O(|unvisited|^2) Prim's for simplicity
            while len(in_mst) < len(unvisited_list):
                min_edge = math.inf
                min_node = None
                for u in in_mst:
                    for v in unvisited_list:
                        if v not in in_mst:
                            w = distance(u, v)
                            if w < min_edge:
                                min_edge = w
                                min_node = v
                in_mst.add(min_node)
                total_mst_cost += min_edge
            return total_mst_cost

        def compute_bound(current_cost, visited, current_node):
            """Compute MST-based lower bound for the partial solution."""
            unvisited_list = [i for i in range(n) if i not in visited]

            # If there are no unvisited nodes, we only need to return to start
            if not unvisited_list:
                return current_cost + distance(current_node, start_node_index)

            # MST of unvisited nodes
            mst_cost = prim_mst_cost(unvisited_list)

            # Minimal link from the current node to any unvisited
            min_current_to_unvisited = min(distance(current_node, u) for u in unvisited_list)

            # Minimal link from unvisited to the start node
            min_unvisited_to_start = min(distance(u, start_node_index) for u in unvisited_list)

            # Total bound
            return current_cost + mst_cost + min_current_to_unvisited + min_unvisited_to_start

        # ---------------------------------------------------------------------
        # 3) Prepare the priority queue (min-heap) for subproblems:
        #    (bound, current_cost, path, visited)
        # ---------------------------------------------------------------------
        initial_path = [start_node_index]
        initial_visited = {start_node_index}
        initial_bound = compute_bound(0, initial_visited, start_node_index)

        L = []
        heapq.heappush(L, (initial_bound, 0, initial_path, initial_visited))

        # ---------------------------------------------------------------------
        # 4) Main Branch-and-Bound loop
        # ---------------------------------------------------------------------
        while L:
            # Enforce the time limit
            if timelimit and (timelimit < time.time() - timestart):
                break

            bound, current_cost, path, visited = heapq.heappop(L)

            # Prune if bound >= best_cost
            if bound >= best_cost:
                continue

            # If path already includes all nodes, finalize the cost
            if len(path) == n:
                total_cost = current_cost + distance(path[-1], path[0])
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path_indices = path[:]
                continue

            # Otherwise, expand to each unvisited node
            for next_city in range(n):
                if next_city not in visited:
                    new_path = path + [next_city]
                    new_visited = visited | {next_city}
                    new_cost = current_cost + distance(path[-1], next_city)

                    new_bound = compute_bound(new_cost, new_visited, next_city)
                    if new_bound < best_cost:
                        heapq.heappush(L, (new_bound, new_cost, new_path, new_visited))

        # ---------------------------------------------------------------------
        # 5) Reconstruct final tour with node labels, including return to start
        # ---------------------------------------------------------------------
        tour = [index_to_node[i] for i in best_path_indices]
        tour.append(tour[0])  # close the cycle

        return {
            "tour": tour,
            "cost": best_cost,
            "meta": vars(self)
        }
