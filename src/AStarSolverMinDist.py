import math
import heapq
import time

class AStarSolverMinDist:
    """
    A solver for the Traveling Salesman Problem using the A* search algorithm.
    Uses an MST-based heuristic for estimating the cost to complete a tour.
    Augmented to handle deadlines to return a valid solution using greedy completion.
    """

    def set_meta_config(self):
        pass

    def __init__(self):
        pass

    def solve(self, graph, timestart, timelimit):
        """
        Solves the TSP using an A* search algorithm. For strict time limits
        Returns a greedy solution from the best path know (so far)

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

        # Convert nodes to a list for a stable ordering.
        nodes = list(graph.get_nodes())
        n = len(nodes)
        if n == 0:
            return {"tour": [], "cost": 0, "meta": vars(self)}

        # Create mapping for convenience: index -> node.
        index_to_node = {i: node for i, node in enumerate(nodes)}

        # Helper function: get the distance between two nodes by their indices.
        def distance(i, j):
            return graph.get_weight(index_to_node[i], index_to_node[j])

        # We treat the first node in the list as the start.
        start = 0

        # ---------------------------------------------------------------------
        # MST helper using Prim's algorithm for the unvisited nodes.
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
        # Heuristic function: estimate remaining cost as MST of unvisited nodes +
        # minimal cost from current to any unvisited and from any unvisited to start.
        # ---------------------------------------------------------------------
        def heuristic(visited, current):
            unvisited = [i for i in range(n) if i not in visited]
            if not unvisited:
                return distance(current, start)
            mst_cost = prim_mst_cost(unvisited)
            min_current_to_unvisited = min(distance(current, u) for u in unvisited)
            min_unvisited_to_start = min(distance(u, start) for u in unvisited)
            return mst_cost + min_current_to_unvisited + min_unvisited_to_start

        # ---------------------------------------------------------------------
        # A* search: each state is (f, g, path, visited)
        # f = g + h, where g is the cost so far and h is the heuristic.
        # ---------------------------------------------------------------------
        initial_path = [start]
        initial_visited = {start}
        g_initial = 0
        h_initial = heuristic(initial_visited, start)
        f_initial = g_initial + h_initial

        # Priority queue (min-heap) to expand states with the smallest f first.
        queue = []
        heapq.heappush(queue, (f_initial, g_initial, initial_path, initial_visited))

        # Dictionary to record the best g cost for each state (current, visited).
        best_state_cost = {}
        # Track the best partial state seen so far (most nodes visited; break ties by lower cost).
        best_partial = (g_initial, initial_path, initial_visited)

        while queue:
            # Enforce the time limit.
            if time.time() - timestart > timelimit:
                break

            f, g, path, visited = heapq.heappop(queue)
            current = path[-1]

            # Update best_partial based on number of nodes visited (and lower cost on tie).
            if (len(visited) > len(best_partial[2])) or (len(visited) == len(best_partial[2]) and g < best_partial[0]):
                best_partial = (g, path, visited)

            state_key = (current, frozenset(visited))
            if state_key in best_state_cost and best_state_cost[state_key] <= g:
                continue
            best_state_cost[state_key] = g

            # If all nodes have been visited, complete the tour by returning to start.
            if len(path) == n:
                total_cost = g + distance(current, start)
                final_path = path + [start]
                tour = [index_to_node[i] for i in final_path]
                return {"tour": tour, "cost": total_cost, "meta": vars(self)}

            # Expand current state by exploring each unvisited node.
            for next_city in range(n):
                if next_city not in visited:
                    new_path = path + [next_city]
                    new_visited = visited | {next_city}
                    new_g = g + distance(current, next_city)
                    new_h = heuristic(new_visited, next_city)
                    new_f = new_g + new_h
                    heapq.heappush(queue, (new_f, new_g, new_path, new_visited))

        # Time limit reached: Greedy completion using best_partial.
        g_partial, path_partial, visited_partial = best_partial
        remaining = set(range(n)) - visited_partial
        current = path_partial[-1]
        additional_cost = 0
        greedy_path = list(path_partial)

        while remaining:
            # Select the nearest unvisited node.
            next_city = min(remaining, key=lambda city: distance(current, city))
            additional_cost += distance(current, next_city)
            greedy_path.append(next_city)
            current = next_city
            remaining.remove(next_city)

        # Finally, add the cost to return to start.
        additional_cost += distance(current, start)
        greedy_path.append(start)
        total_cost = g_partial + additional_cost
        tour = [index_to_node[i] for i in greedy_path]

        # Return the tour generated by the greedy completion.
        return {"tour": tour, "cost": total_cost, "meta": vars(self)}
