import tsplib95

files = ["att48.tsp", "dantzig42.tsp", "fri26.tsp", "gr17.tsp", "p01.tsp"]
tsp_data = {}

for file in files:
    tsp_data[file] = tsplib95.load('dataset/'+file)

print(tsp_data["att48.tsp"])
