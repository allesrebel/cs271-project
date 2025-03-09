import collections
import tsplib95
import time
import math

def verify_solution(graph, solution):
    """
    Verifies a TSP solution.

    Checks:
      - The tour starts and ends at the same node.
      - Each node (other than the endpoints) is visited exactly once.
      - The reported cost matches the sum of edge weights computed along the tour.

    Parameters:
      graph: A TSP instance with methods get_nodes() and get_weight(i, j).
      solution: A dictionary with keys "tour" (a list of nodes) and "cost" (a numeric value).

    Returns:
      (bool, str): A tuple where the first element is True if the solution is valid,
                   otherwise False. The second element is a message describing the result.
    """
    tour = solution.get("tour", [])
    reported_cost = solution.get("cost", None)

    if not tour:
        return False, "Tour is empty."

    # Check that the start and end are the same.
    if tour[0] != tour[-1]:
        return False, "Tour does not start and end at the same node."

    # Get the list of nodes in the graph.
    nodes = list(graph.get_nodes())

    # Exclude the fixed endpoints to check the interior.
    interior = tour[1:-1]

    # Ensure each interior node is unique.
    if len(interior) != len(set(interior)):
        return False, "Some nodes are visited more than once in the interior of the tour."

    # In a valid TSP tour, the interior nodes should be exactly the graph's nodes minus the starting node.
    expected_interior = set(nodes)
    try:
        expected_interior.remove(tour[0])
    except KeyError:
        return False, "The starting node is not in the graph's node set."

    if set(interior) != expected_interior:
        return False, "Interior nodes do not match the expected set of nodes."

    # Compute the cost of the tour.
    computed_cost = sum(graph.get_weight(tour[i], tour[i+1]) for i in range(len(tour) - 1))

    # Allow for a small floating-point tolerance.
    if abs(computed_cost - reported_cost) > 1e-6:
        return False, f"Computed cost ({computed_cost}) does not match reported cost ({reported_cost})."

    return True, "Solution is valid."


from src.GeneticAlgorithmSolver import GeneticAlgorithmSolver
from src.HeuristicSolver import HeuristicSolver
from src.SimulatedAnnealingSolver import SimulatedAnnealingSolver

files = ["att48.tsp", "dantzig42.tsp", "fri26.tsp", "gr17.tsp", "p01.tsp"]
tsp_optimal = {
    "att48.tsp": 10628,
    "dantzig42.tsp": 699,
    "fri26.tsp": 937,
    "gr17.tsp": 2085,
    "p01.tsp": 291
}
tsp_data = {}

for file in files:
    file_path = 'dataset/' + file
    with open(file_path, 'r') as f:
        file_content = f.read()
    # Add an extra newline to the end of the file content
    # for compatibility with the TSPLIB parser (specifically p01.tsp)
    file_content += "\n"
    # Parse the file content into a TSPLIB graph
    tsp_data[file] = tsplib95.parse(file_content)

solvers = [HeuristicSolver, GeneticAlgorithmSolver, SimulatedAnnealingSolver]

results = collections.defaultdict(list)

for solver_cls in solvers:
    for file, graph in tsp_data.items():
        solver_instance = solver_cls()

        if solver_cls == GeneticAlgorithmSolver:
            solver_instance.set_meta_config(2,0.5,0.5,0.5,10)

        start_time = time.time()
        solution = solver_instance.solve(graph)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # validate the solution
        valid, message = verify_solution(graph, solution)
        if not valid:
            print(f"Verification failed for {solver_cls.__qualname__} on {file}: {message}")
            exit(2)

        results[file].append({
            "solver" : solver_cls.__qualname__,
            "solution": solution,
            "time": elapsed_time,
            "optimal": tsp_optimal[file]
        })

print(results)
