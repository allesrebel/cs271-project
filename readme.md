# CS271-Project!
Comparing and contrasting TSP solutions - using different solvers!
Looking at time complexity (measured indirectly with wall time) against the accuracy of different methods.
Doing multiple runs to get an idea of how accurate each solution would be. 

Requires python3, cmake, wget to be installed. Running cmake will download all dependencies if they aren't present.

## Installing dependencies
On Ubuntu or other Debian based systems:
* apt install cmake python3 python3-venv wget

## Additional Heuristic Search
TODO:
- Also add time constraints

Linear Memory - Provably Optimal

Add A* to the hueristic search

Unlike A*, branch and bound is any time (meaning you can pick any timescale)

Minimum spanning tree is probably the best hueristic

Local Search as another method, tour

# Commands
```cmake -B build```
- Creates a directory called 'build', downloads the dataset, and sets up the python virtual environment
```cmake --build build --target run```
- Uses the 'build' folder scripts to run the python code in the project using the virtual environment
```cmake --build build --target analyze```
- Uses the 'build' folder's virtual environment to run the analysis script on a result.json (in the root dir). Outputs images

# Manual Usage (without cmake)
Pip install requirements.txt, and then run ```main.py``
