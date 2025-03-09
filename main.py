import tsplib95
import time
import math

from src.GeneticAlgorithmSolver import GeneticAlgorithmSolver
from src.HeuristicSolver import HeuristicSolver
from src.SimulatedAnnealingSolver import SimulatedAnnealingSolver

files = ["att48.tsp"]#, "dantzig42.tsp", "fri26.tsp", "gr17.tsp", "p01.tsp"]
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

solvers = [HeuristicSolver] #, GeneticAlgorithmSolver, SimulatedAnnealingSolver]

results = {}

for solver_cls in solvers:
    for file, graph in tsp_data.items():
        solver_instance = solver_cls()
        start_time = time.time()
        solution = solver_instance.solve(graph)
        end_time = time.time()
        elapsed_time = end_time - start_time
        results[file] = {
            "solution": solution,
            "time": elapsed_time,
            "optimal": tsp_optimal[file]
        }

#print(list( tsp_data["att48.tsp"].get_nodes()))


print(results)
