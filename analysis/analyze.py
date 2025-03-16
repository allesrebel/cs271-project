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
        time_limit = result['timelimit']
        # Avoid division by zero by checking if cost is not zero
        accuracy = optimal / cost if cost != 0 else np.nan

        data_rows.append({
            'File': filename,
            'Solver': solver,
            'Accuracy': accuracy,
            'Time': time_taken,
            'Timelimit': time_limit
        })

# Create DataFrame
df = pd.DataFrame(data_rows)

# Define the time limits you want to consider
time_limits = [1, 60, 120, 300, 0]


for t in time_limits:
    # Filter the DataFrame by the current time limit (note the column is 'timelimit' in your JSON)
    df_t = df[df['Timelimit'] == t]
    if df_t.empty:
        print(f"No data for time limit {t} sec, skipping...")
        continue
    # ----------------------------
    # Plot 1: Accuracy per File by Solver with error bars
    # ----------------------------
    # Compute mean and std for Accuracy per File and Solver
    # Use df_t (filtered by time limit) for the aggregation
    solver_accuracy_stats = df_t.groupby(['File', 'Solver'])['Accuracy'].agg(['mean', 'std']).reset_index()
    pivot_mean = solver_accuracy_stats.pivot(index='File', columns='Solver', values='mean')
    pivot_std = solver_accuracy_stats.pivot(index='File', columns='Solver', values='std')

    # Print the underlying accuracy datapoints used for each File and Solver group
    print(f"\nData points for Accuracy per File by Solver for time limit {t} sec:")
    for (file, solver), group in df_t.groupby(['File', 'Solver']):
        print(f"  File: {file}, Solver: {solver}, Accuracies: {group['Accuracy'].tolist()}")


    ax = pivot_mean.plot(kind='bar', figsize=(10, 6), yerr=pivot_std, capsize=4)
    ax.set_title(f'Normalized Accuracy per Dataset by Solver - {t}s Limit')
    ax.set_ylabel('Accuracy (Optimal / Solver Cost)')
    ax.set_xlabel('Dataset (increasing numer of cities)')
    ax.legend(title='Solver')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if args.save:
        plt.savefig(f'accuracy_per_file_solver_{t}s.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # ----------------------------
    # Plot 2: Average Accuracy by Solver with error bars
    # ----------------------------
    # Compute mean and std for Accuracy by Solver
    solver_accuracy_overall = df_t.groupby('Solver')['Accuracy'].agg(['mean', 'std'])
    ax = solver_accuracy_overall['mean'].plot(kind='bar', figsize=(10, 6), 
                                            yerr=solver_accuracy_overall['std'], 
                                            color='skyblue', edgecolor='black', capsize=4)
    ax.set_title(f'Average Accuracy per Solver - {t}s Limit')
    ax.set_ylabel('Average Accuracy')
    ax.set_xlabel('Solver')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if args.save:
        plt.savefig(f'average_accuracy_solver_{t}s.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # ----------------------------
    # Plot 3: Execution Time per File by Solver with error bars
    # ----------------------------
    # Compute mean and std for Time per File and Solver
    time_stats = df_t.groupby(['File', 'Solver'])['Time'].agg(['mean', 'std']).reset_index()
    pivot_mean_time = time_stats.pivot(index='File', columns='Solver', values='mean')
    pivot_std_time = time_stats.pivot(index='File', columns='Solver', values='std')

    ax = pivot_mean_time.plot(kind='bar', figsize=(10, 6), yerr=pivot_std_time, capsize=4)
    ax.set_title(f'Execution Time per File by Solver - {t}s Limit')
    ax.set_ylabel('Time (seconds)')
    ax.set_xlabel('Dataset (increasing numer of cities)')
    ax.legend(title='Solver')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if args.save:
        plt.savefig(f'execution_time_filewise_solver_{t}s.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ----------------------------
# Plot 4: Normalized Accuracy for solver over time limits (grouped by solver) 
# ----------------------------
# Loop over each dataset in the DataFrame
for dataset in df['File'].unique():
    df_dataset = df[df['File'] == dataset]
    
    if df_dataset.empty:
        print(f"No data found for dataset {dataset}.")
        continue

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get unique solvers for the current dataset
    solvers = df_dataset['Solver'].unique()
    
    for solver in solvers:
        # Filter data for the current solver within this dataset
        df_solver = df_dataset[df_dataset['Solver'] == solver]
        # Group by time limit and compute mean and std of Accuracy
        stats = df_solver.groupby('Timelimit')['Accuracy'].agg(['mean', 'std']).reset_index()
        # Ensure the data is ordered by time limit
        stats = stats.sort_values('Timelimit')
        
        # Plot a line graph with error bars
        ax.errorbar(stats['Timelimit'], stats['mean'], yerr=stats['std'],
                    label=solver, marker='o', capsize=4)
    
    ax.set_xlabel('Time Limit (sec)')
    ax.set_ylabel('Accuracy (Optimal / Solver Cost)')
    ax.set_title(f'Normalized Solver Accuracy vs Time Control for Dataset: {dataset}')
    ax.legend(title='Solver')
    ax.grid(True)
    plt.tight_layout()
    
    if args.save:
        plt.savefig(f'accuracy_line_{dataset}.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()
