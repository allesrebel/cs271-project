import math
import heapq
import time

class AStarSolver:
    """
    A solver for the Traveling Salesman Problem using the A* search algorithm.
    Uses an MST-based heuristic for estimating the cost to complete a tour.
    """

    def set_meta_config(self):
        pass

    def __init__(self):
        pass

    def solve(self, graph, timestart, timelimit):
        """
        Solves the TSP using an A* search algorithm.

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

        # Convert nodes to a list to have a stable ordering
        nodes = list(graph.get_nodes())
        n = len(nodes)
        if n == 0:
            return {"tour": [], "cost": 0, "meta": vars(self)}

        # Create mapping for convenience: node label <-> index.
        node_to_index = {node: i for i, node in enumerate(nodes)}
        index_to_node = {i: node for i, node in enumerate(nodes)}

        # Helper function: get the distance between two indices.
        def distance(i, j):
            return graph.get_weight(index_to_node[i], index_to_node[j])

        # We'll treat the first node in the list as the start.
        start = 0

        # ---------------------------------------------------------------------
        # MST helper using Prim's algorithm (O(n^2) for the unvisited subset)
        # ---------------------------------------------------------------------
        def prim_mst_cost(unvisited_list):
            if len(unvisited_list) <= 1:
                return 0
            in_mst = {unvisited_list[0]}
            total_cost = 0
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
                total_cost += min_edge
            return total_cost

        # ---------------------------------------------------------------------
        # Heuristic function:
        #   h = MST cost of unvisited nodes +
        #       min_edge from current node to any unvisited +
        #       min_edge from any unvisited to the start node.
        # ---------------------------------------------------------------------
        def heuristic(visited, current):
            unvisited = [i for i in range(n) if i not in visited]
            # If no nodes left unvisited, estimate is simply cost to return to start.
            if not unvisited:
                return distance(current, start)
            mst_cost = prim_mst_cost(unvisited)
            min_current_to_unvisited = min(distance(current, u) for u in unvisited)
            min_unvisited_to_start = min(distance(u, start) for u in unvisited)
            return mst_cost + min_current_to_unvisited + min_unvisited_to_start

        # ---------------------------------------------------------------------
        # A* search: each state is (f, g, path, visited)
        # f = g + h where g is cost so far, and h is our heuristic.
        # ---------------------------------------------------------------------
        initial_path = [start]
        initial_visited = {start}
        g_initial = 0
        h_initial = heuristic(initial_visited, start)
        f_initial = g_initial + h_initial

        # Priority queue (min-heap) to expand states with the smallest f first.
        queue = []
        heapq.heappush(queue, (f_initial, g_initial, initial_path, initial_visited))

        # Use a dictionary to record best cost (g value) found for a state (current, visited)
        best_state_cost = {}

        while queue:
            # Enforce time limit
            if timelimit and (timelimit < time.time() - timestart):
                break

            f, g, path, visited = heapq.heappop(queue)
            current = path[-1]

            # Use (current, frozenset(visited)) as state key.
            state_key = (current, frozenset(visited))
            if state_key in best_state_cost and best_state_cost[state_key] <= g:
                continue
            best_state_cost[state_key] = g

            # If all nodes have been visited, finish by returning to start.
            if len(path) == n:
                total_cost = g + distance(current, start)
                final_path = path + [start]
                # Convert indices back to node labels.
                tour = [index_to_node[i] for i in final_path]
                return {
                    "tour": tour,
                    "cost": total_cost,
                    "meta": vars(self)
                }

            # Expand current state to each unvisited node.
            for next_city in range(n):
                if next_city not in visited:
                    new_path = path + [next_city]
                    new_visited = visited | {next_city}
                    new_g = g + distance(current, next_city)
                    new_h = heuristic(new_visited, next_city)
                    new_f = new_g + new_h
                    heapq.heappush(queue, (new_f, new_g, new_path, new_visited))

        # If no complete tour was found (should not occur for a connected graph)
        return {"tour": [], "cost": math.inf, "meta": vars(self)}
