import collections
import tsplib95
import time
import multiprocessing
import os
import json
import random

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
from src.AStarSolverMinDist import AStarSolverMinDist


# List of TSPLIB instance files and their known optimal costs.
files = ['p01.tsp', 'gr17.tsp', 'fri26.tsp', 'dantzig42.tsp', 'att48.tsp']

tsp_optimal = {
    "p01.tsp": 291,
    "gr17.tsp": 2085,
    "fri26.tsp": 937,
    "dantzig42.tsp": 699,
    "att48.tsp": 10628,
}

solvers = [
    HeuristicSolver, 
    BranchAndBoundMSTSolver, 
    GeneticAlgorithmSolver, 
    SimulatedAnnealingSolver,
    AStarSolver,
    AStarSolverMinDist,
    LocalSearchSolver
]

# Prepare tasks with picklable parameters: (solver class, file name, optimal cost).
tasks = []
for solver_cls in solvers:
    import numpy as np
    num_points = 50
    values = np.linspace(1, 300, num=num_points)
    values_int = np.rint(values).astype(int)

    # Loop over the integer values
    for timelimit in values_int:
        for _ in range(10):
            for file in files:
                tasks.append((solver_cls, file, tsp_optimal[file], timelimit))

    if solver_cls == 'AStarSolver':
        # only do A Star on time bound, since it takes 24 hours(Or more)
        # larger city files, which isn't worth it given more optimal algs
        continue

    for _ in range(10):
        for file in files:
            tasks.append((solver_cls, file, tsp_optimal[file], None))

results = collections.defaultdict(list)
delta_file = 'results_delta.jsonl'
master_file = 'results.json'

# Create a lock for synchronizing file writes
lock = multiprocessing.Lock()

def append_delta(update):
    with lock:
        with open(delta_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(update) + "\n")

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
    print("Working Directory:", os.getcwd())

    # Spawn only as many processes as there are CPU cores.
    num_processors = multiprocessing.cpu_count()
    # Shuffle Task list to remove bias
    random.shuffle(tasks)
    print(tasks)

    with multiprocessing.Pool(processes=num_processors) as pool:
        for result in pool.imap_unordered(run_solver_task, tasks):
            file, solver_name, solution, elapsed_time, optimal, timelimit = result
            update = {
                "file" : file,
                "solver": solver_name,
                "solution": solution,
                "time": elapsed_time,
                "optimal": optimal,
                "timelimit": int(timelimit) if timelimit else 0
            }
            # Update the in-memory dictionary
            results[file].append(update)
            # Append just the new update to the delta file
            append_delta(update)

    with open(master_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4, sort_keys=True)
