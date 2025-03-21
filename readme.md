# CS271-Project!
Comparing and contrasting TSP solutions - using different solvers!
Looking at time complexity (measured indirectly with wall time) against the accuracy of different methods.
Doing multiple runs to get an idea of how accurate each solution would be. 

Requires python3, cmake, wget to be installed. Running cmake will download all dependencies if they aren't present.

## Installing dependencies
On Ubuntu or other Debian based systems:
* apt install cmake python3 python3-venv wget

# Commands
```cmake -B build```
- Creates a directory called 'build', downloads the dataset, and sets up the python virtual environment
```cmake --build build --target run```
- Uses the 'build' folder scripts to run the python code in the project using the virtual environment
```cmake --build build --target analyze```
- Uses the 'build' folder's virtual environment to run the analysis script on a result.json (in the root dir). Outputs images

# Manual Usage (without cmake)
Pip install requirements.txt, and then run ```main.py``

# Algorithms Implemented
## Heuristic Based
* Minimum Distance
* A Star (with MST)
* Branch and Bound (with MST)
## Local Search Based
* Simple Local Search
* Simulated Annealing
* Genetic Algorithm