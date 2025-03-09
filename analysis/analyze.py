import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Process TSP solver results.')
parser.add_argument('input_file', type=str, help='Input JSON file containing solver results')
parser.add_argument('--save', action='store_true', help='Save to disk instead of displaying plots')
args = parser.parse_args()

# Load data from JSON file
with open(args.input_file, 'r') as f:
    data = json.load(f)

# Flattening data into DataFrame
data_rows = []
for filename, results in data.items():
    for result in results:
        solver = result['solver']
        cost = result['solution']['cost']
        optimal = result['optimal']
        time_taken = result['time']
        # Avoid division by zero by checking if cost is not zero
        accuracy = optimal / cost if cost != 0 else np.nan

        data_rows.append({
            'File': filename,
            'Solver': solver,
            'Accuracy': accuracy,
            'Time': time_taken
        })

# Create DataFrame
df = pd.DataFrame(data_rows)

# ----------------------------
# Plot 1: Accuracy per File by Solver with error bars
# ----------------------------
# Compute mean and std for Accuracy per File and Solver
solver_accuracy_stats = df.groupby(['File', 'Solver'])['Accuracy'].agg(['mean', 'std']).reset_index()
pivot_mean = solver_accuracy_stats.pivot(index='File', columns='Solver', values='mean')
pivot_std = solver_accuracy_stats.pivot(index='File', columns='Solver', values='std')

ax = pivot_mean.plot(kind='bar', figsize=(10, 6), yerr=pivot_std, capsize=4)
ax.set_title('Accuracy per File by Solver')
ax.set_ylabel('Accuracy (Optimal / Solver Cost)')
ax.set_xlabel('File')
ax.legend(title='Solver')
plt.xticks(rotation=45)
plt.tight_layout()
if args.save:
    plt.savefig('accuracy_per_file_solver.png', bbox_inches='tight')
    plt.close()
else:
    plt.show()

# ----------------------------
# Plot 2: Average Accuracy by Solver with error bars
# ----------------------------
# Compute mean and std for Accuracy by Solver
solver_accuracy_overall = df.groupby('Solver')['Accuracy'].agg(['mean', 'std'])
ax = solver_accuracy_overall['mean'].plot(kind='bar', figsize=(10, 6), 
                                          yerr=solver_accuracy_overall['std'], 
                                          color='skyblue', edgecolor='black', capsize=4)
ax.set_title('Average Accuracy per Solver')
ax.set_ylabel('Average Accuracy')
ax.set_xlabel('Solver')
plt.xticks(rotation=45)
plt.tight_layout()
if args.save:
    plt.savefig('average_accuracy_solver.png', bbox_inches='tight')
    plt.close()
else:
    plt.show()

# ----------------------------
# Plot 3: Execution Time per File by Solver with error bars
# ----------------------------
# Compute mean and std for Time per File and Solver
time_stats = df.groupby(['File', 'Solver'])['Time'].agg(['mean', 'std']).reset_index()
pivot_mean_time = time_stats.pivot(index='File', columns='Solver', values='mean')
pivot_std_time = time_stats.pivot(index='File', columns='Solver', values='std')

ax = pivot_mean_time.plot(kind='bar', figsize=(10, 6), yerr=pivot_std_time, capsize=4)
ax.set_title('Execution Time per File by Solver')
ax.set_ylabel('Time (seconds)')
ax.set_xlabel('File')
ax.legend(title='Solver')
plt.xticks(rotation=45)
plt.tight_layout()
if args.save:
    plt.savefig('execution_time_filewise_solver.png', bbox_inches='tight')
    plt.close()
else:
    plt.show()
