import random
import math
import time

class SimulatedAnnealingSolver:
    """
    A simulated annealing solver for the Traveling Salesman Problem.
    The endpoints (first and last node) are fixed to make it work!
    """

    def set_meta_config( self, initial_temperature, cooling_rate, iterations_per_temp, temperature_threshold ):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.temperature_threshold = temperature_threshold

    def __init__(self):
        # Randomly generate meta parameters for simulated annealing.
        self.initial_temperature = random.uniform(1000, 5000)   # e.g., between 1000 and 5000
        self.cooling_rate = random.uniform(0.95, 0.99)          # e.g., cooling rate between 0.95 and 0.99
        self.iterations_per_temp = random.randint(100, 500)     # iterations per temperature level
        self.temperature_threshold = 1e-3                       # stopping condition for temperature

    def solve(self, graph, timestart, timelimit=None):
        """
        Solves the TSP using simulated annealing.

        Parameters:
            graph: A TSP instance (e.g. loaded using tsplib95) that provides:
                   - get_nodes() returning the set/list of nodes.
                   - get_weight(i, j) returning the distance between nodes i and j.

        Returns:
            A dictionary with:
              - "tour": A list representing the order in which nodes are visited,
                        starting and ending at the same node.
              - "cost": The total cost (distance) of the computed tour.
        """
        # Obtain nodes and verify that the list is not empty.
        nodes = list(graph.get_nodes())
        if not nodes:
            return {"tour": [], "cost": 0}
        
        # Create an initial solution:
        # Fix the starting node, randomize the order of the remaining nodes,
        # and append the starting node at the end.
        start = nodes[0]
        middle = random.sample(nodes[1:], len(nodes) - 1)
        current_solution = [start] + middle + [start]

        # Helper function to compute the total cost of a tour.
        def compute_cost(tour):
            return sum(graph.get_weight(tour[i], tour[i+1]) for i in range(len(tour) - 1))
        
        current_cost = compute_cost(current_solution)
        best_solution = current_solution[:]
        best_cost = current_cost

        # Set initial temperature.
        T = self.initial_temperature
        
        # Main simulated annealing loop.
        while T > self.temperature_threshold:
            for _ in range(self.iterations_per_temp):
                # Have we hit our time control?
                if timelimit and (timelimit < time.time() - timestart):
                    return {"tour": best_solution, "cost": best_cost, "meta": vars(self) }

                # Generate a neighboring solution by swapping two interior nodes.
                i, j = random.sample(range(1, len(current_solution) - 1), 2)
                neighbor = current_solution[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                
                neighbor_cost = compute_cost(neighbor)
                delta = neighbor_cost - current_cost

                # Accept neighbor if it's better or probabilistically if worse.
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_solution = neighbor
                    current_cost = neighbor_cost
                    # Update best solution found so far.
                    if current_cost < best_cost:
                        best_solution = current_solution[:]
                        best_cost = current_cost

            # Cool down the temperature.
            T *= self.cooling_rate

        return {"tour": best_solution, "cost": best_cost, "meta": vars(self) }
