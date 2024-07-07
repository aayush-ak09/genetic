import random
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.1

# Generate random cities
NUM_CITIES = 20
cities = np.random.rand(NUM_CITIES, 2)

# Calculate the distance matrix
def calculate_distance_matrix(cities):
    distance_matrix = np.sqrt(((cities[:, np.newaxis, :] - cities[np.newaxis, :, :]) ** 2).sum(axis=2))
    return distance_matrix

distance_matrix = calculate_distance_matrix(cities)

# Function to create a random chromosome
def create_random_chromosome():
    chromosome = list(np.random.permutation(NUM_CITIES))
    return chromosome

# Initialize population
def initialize_population():
    population = [create_random_chromosome() for _ in range(POPULATION_SIZE)]
    return population

# Fitness function: Total distance of the tour
def fitness(chromosome):
    return sum(distance_matrix[chromosome[i], chromosome[i + 1]] for i in range(NUM_CITIES - 1)) + \
           distance_matrix[chromosome[-1], chromosome[0]]

# Selection: Tournament selection
def selection(population):
    selected = []
    for _ in range(POPULATION_SIZE):
        tournament = random.sample(population, k=5)
        tournament.sort(key=fitness)
        selected.append(tournament[0])
    return selected

# Crossover: Ordered crossover
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(NUM_CITIES), 2))
    offspring = [None] * NUM_CITIES
    offspring[start:end] = parent1[start:end]

    fill_position = end
    for gene in parent2:
        if gene not in offspring:
            if fill_position >= NUM_CITIES:
                fill_position = 0
            offspring[fill_position] = gene
            fill_position += 1

    return offspring

# Mutation: Swap mutation
def mutation(chromosome):
    idx1, idx2 = random.sample(range(NUM_CITIES), 2)
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

# Genetic Algorithm
def genetic_algorithm():
    population = initialize_population()
    best_fitness = float('inf')
    best_solution = None

    for generation in range(GENERATIONS):
        population = sorted(population, key=fitness)
        if fitness(population[0]) < best_fitness:
            best_fitness = fitness(population[0])
            best_solution = population[0]

        parents = selection(population)
        next_generation = []

        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)
            offspring = crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                offspring = mutation(offspring)
            next_generation.append(offspring)

        population = next_generation

    return best_solution, best_fitness

# Example usage
best_solution, best_fitness = genetic_algorithm()
print("Best solution:", best_solution)
print("Fitness of the best solution:", best_fitness)

# Plot the best solution
plt.figure(figsize=(10, 5))
plt.plot(cities[best_solution, 0], cities[best_solution, 1], 'o-')
plt.plot([cities[best_solution[-1], 0], cities[best_solution[0], 0]], [cities[best_solution[-1], 1], cities[best_solution[0], 1]], 'o-')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Best TSP Solution Found by Genetic Algorithm')
plt.show()
