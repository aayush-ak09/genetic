import pandas as pd
import numpy as np

# Load data from Excel
data = pd.read_excel('PSO/data.xlsx', sheet_name='data.xlsx')


# Extract parameters (replace 'Column' with actual column names or indices)
lambda_values = data['lambda'].values
mu_values = data['mu'].values
availability = data['Availability'].values


# Define objective function (example: minimize the sum of lambda values)
def fitness_function(position):
    return np.sum(position ** 2)  # Example objective function


# PSO parameters
num_particles = 30
dimensions = len(lambda_values)  # Assuming each lambda value is a dimension
max_iterations = 100
w = 0.5  # Inertia weight
c1 = 1.5  # Cognitive component
c2 = 1.5  # Social component

# Initialize particles
positions = np.random.rand(num_particles, dimensions)
velocities = np.random.rand(num_particles, dimensions)
personal_best_positions = positions.copy()
personal_best_scores = np.array([fitness_function(p) for p in positions])
global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
global_best_score = np.min(personal_best_scores)

# Main loop
for t in range(max_iterations):
    for i in range(num_particles):
        velocities[i] = (w * velocities[i] +
                         c1 * np.random.rand() * (personal_best_positions[i] - positions[i]) +
                         c2 * np.random.rand() * (global_best_position - positions[i]))
        positions[i] += velocities[i]

        # Evaluate fitness
        score = fitness_function(positions[i])

        # Update personal best
        if score < personal_best_scores[i]:
            personal_best_positions[i] = positions[i]
            personal_best_scores[i] = score

        # Update global best
        if score < global_best_score:
            global_best_position = positions[i]
            global_best_score = score

    # Print progress (optional)
    print(f"Iteration {t + 1}/{max_iterations}, Best Score: {global_best_score}")

# Output results
print(f"Best Position: {global_best_position}")
print(f"Best Score: {global_best_score}")
