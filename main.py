import collections
import tsplib95
import time
import multiprocessing
import os
import json

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

    if tour[0] != tour[-1]:
        return False, "Tour does not start and end at the same node."

    nodes = list(graph.get_nodes())
    interior = tour[1:-1]

    if len(interior) != len(set(interior)):
        return False, "Some nodes are visited more than once in the interior of the tour."

    expected_interior = set(nodes)
    try:
        expected_interior.remove(tour[0])
    except KeyError:
        return False, "The starting node is not in the graph's node set."

    if set(interior) != expected_interior:
        return False, "Interior nodes do not match the expected set of nodes."

    computed_cost = sum(graph.get_weight(tour[i], tour[i+1]) for i in range(len(tour) - 1))

    if abs(computed_cost - reported_cost) > 1e-6:
        return False, f"Computed cost ({computed_cost}) does not match reported cost ({reported_cost})."

    return True, "Solution is valid."


from src.GeneticAlgorithmSolver import GeneticAlgorithmSolver
from src.HeuristicSolver import HeuristicSolver
from src.SimulatedAnnealingSolver import SimulatedAnnealingSolver
from src.BranchNBoundMSTSolver import BranchAndBoundMSTSolver
from src.AStarSolver import AStarSolver
from src.LocalSearchSolver import LocalSearchSolver


# List of TSPLIB instance files and their known optimal costs.
files = ["att48.tsp", "dantzig42.tsp", "fri26.tsp", "gr17.tsp", "p01.tsp"]

tsp_optimal = {
    "att48.tsp": 10628,
    "dantzig42.tsp": 699,
    "fri26.tsp": 937,
    "gr17.tsp": 2085,
    "p01.tsp": 291
}

solvers = [
    HeuristicSolver, 
    BranchAndBoundMSTSolver, 
    GeneticAlgorithmSolver, 
    SimulatedAnnealingSolver,
    AStarSolver,
    LocalSearchSolver
]

# Prepare tasks with picklable parameters: (solver class, file name, optimal cost).
tasks = []
for solver_cls in solvers:
    # do each run 1 times, with no timelimit
    for _ in range(1):#10):
        for file in files:
            tasks.append((solver_cls, file, tsp_optimal[file], None))
    # do each run 1 times, with 1 min of limit
    for _ in range(1):#10):
        for file in files:
            tasks.append((solver_cls, file, tsp_optimal[file], 60))
    # do each run 1 times, with 2 min of limit
    for _ in range(1):#10):
        for file in files:
            tasks.append((solver_cls, file, tsp_optimal[file], 60*2))
    # do each run 1 times, with 5 min of limit
    for _ in range(1):#100):
        for file in files:
            tasks.append((solver_cls, file, tsp_optimal[file], 60*5))

def run_solver_task(args):
    solver_cls, file, optimal_value, timelimit = args
    file_path = os.path.join('dataset', file)
    
    # Read and parse the file content inside the worker.
    with open(file_path, 'r') as f:
        file_content = f.read()
    # Add an extra newline for TSPLIB compatibility (especially for p01.tsp)
    file_content += "\n"
    graph = tsplib95.parse(file_content)
    
    solver_instance = solver_cls()
    start_time = time.time()
    solution = solver_instance.solve(graph, start_time, timelimit)
    elapsed_time = time.time() - start_time

    valid, message = verify_solution(graph, solution)
    if not valid:
        if timelimit:
            print(f"Verification failed for {solver_cls.__qualname__} on {file} with {timelimit}s limit: {message}")
        else:
            print(f"Verification failed for {solver_cls.__qualname__} on {file}: {message}")

    return file, solver_cls.__qualname__, solution, elapsed_time, optimal_value, timelimit

if __name__ == "__main__":
    # Spawn only as many processes as there are CPU cores.
    num_processors = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processors) as pool:
        results_list = pool.map(run_solver_task, tasks)

    # Aggregate results in a thread-safe manner in the main process.
    results = collections.defaultdict(list)
    for file, solver_name, solution, elapsed_time, optimal, timelimit in results_list:
        results[file].append({
            "solver": solver_name,
            "solution": solution,
            "time": elapsed_time,
            "optimal": optimal,
            "timelimit": timelimit if timelimit else 0
        })

    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4, sort_keys=True)
