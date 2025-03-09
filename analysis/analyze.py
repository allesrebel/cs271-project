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

# Plot average accuracy per file grouped by Solver
solver_accuracy_df = df.groupby(['File', 'Solver']).mean().reset_index()
pivot_df = solver_accuracy_df.pivot(index='File', columns='Solver', values='Accuracy')

ax = pivot_df.plot(kind='bar', figsize=(10, 6))
ax.set_title('Accuracy per File by Solver')
ax.set_ylabel('Accuracy (Optimal / Solver Cost)')
ax.set_xlabel('File')
ax.legend(title='Solver')
plt.xticks(rotation=45)
plt.tight_layout()
if args.save:
    plt.savefig('accuracy_per_file_solver.png', bbox_inches='tight')
else:
    plt.show()

# Plot average accuracy by Solver
solver_avg_accuracy = df.groupby('Solver')['Accuracy'].mean()
ax = solver_avg_accuracy.plot(kind='bar', figsize=(10, 6), color='skyblue', edgecolor='black')
ax.set_title('Average Accuracy per Solver')
ax.set_ylabel('Average Accuracy')
ax.set_xlabel('Solver')
plt.xticks(rotation=45)
plt.tight_layout()
if args.save:
    plt.savefig('average_accuracy_solver.png', bbox_inches='tight')
else:
    plt.show()

# Plot execution time per file using pivot_table to handle duplicates
pivot_time = df.pivot_table(index='File', columns='Solver', values='Time', aggfunc='mean')
ax = pivot_time.plot(kind='bar', figsize=(10, 6))
ax.set_title('Execution Time per File by Solver')
ax.set_ylabel('Time (seconds)')
ax.set_xlabel('File')
ax.legend(title='Solver')
plt.xticks(rotation=45)
plt.tight_layout()
if args.save:
    plt.savefig('execution_time_filewise_solver.png', bbox_inches='tight')
else:
    plt.show()
