import tsplib95
import time

from src.GeneticAlgorithmSolver import GeneticAlgorithmSolver
from src.HeuristicSolver import HeuristicSolver
from src.SimulatedAnnealingSolver import SimulatedAnnealingSolver

files = ["att48.tsp", "dantzig42.tsp", "fri26.tsp", "gr17.tsp", "p01.tsp"]
tsp_optimal = {
    "att48.tsp": 33523,
    "dantzig42.tsp": 699,
    "fri26.tsp": 937,
    "gr17.tsp": 2085,
    "p01.tsp": 291
}
tsp_data = {}

for file in files:
    tsp_data[file] = tsplib95.load('dataset/'+file)

solvers = [GeneticAlgorithmSolver, HeuristicSolver, SimulatedAnnealingSolver]

results = {}

for solver in solvers:
    for file, graph in tsp_data.items():
        start_time = time.time()
        solution = solver.solve(graph)
        end_time = time.time()
        elapsed_time = end_time - start_time
        results[file] = {
            "solution": solution,
            "time": elapsed_time,
            "optimal": tsp_optimal[file]
        }

print(results)