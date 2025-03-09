# CS271-Project!
Requires python3, cmake, wget to be installed. Running cmake will download all dependencies if they aren't present.

## Installing dependencies
On Ubuntu or other Debian based systems:
* apt install cmake python3 python3-venv wget

# Commands
```cmake -B build```
- Creates a directory called 'build', downloads the dataset, and sets up the python virtual environment
```cmake --build build --target run```
- Uses the 'build' folder scripts to run the python code in the project using the virtual environment
