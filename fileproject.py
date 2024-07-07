import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv('data.csv')
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values  # Target
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Parameters
POPULATION_SIZE = 20
GENERATIONS = 30
MUTATION_RATE = 0.1

# Define hyperparameter ranges
HYPERPARAMETER_RANGES = {
    'n_estimators': (10, 200),
    'max_depth': (1, 20),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 20)
}

# Function to create a random chromosome
def create_random_chromosome():
    chromosome = {
        'n_estimators': random.randint(*HYPERPARAMETER_RANGES['n_estimators']),
        'max_depth': random.randint(*HYPERPARAMETER_RANGES['max_depth']),
        'min_samples_split': random.randint(*HYPERPARAMETER_RANGES['min_samples_split']),
        'min_samples_leaf': random.randint(*HYPERPARAMETER_RANGES['min_samples_leaf'])
    }
    return chromosome

# Initialize population
def initialize_population():
    population = [create_random_chromosome() for _ in range(POPULATION_SIZE)]
    return population

# Fitness function: Train and evaluate the model
def fitness(chromosome, X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=chromosome['n_estimators'],
        max_depth=chromosome['max_depth'],
        min_samples_split=chromosome['min_samples_split'],
        min_samples_leaf=chromosome['min_samples_leaf'],
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

# Selection: Tournament selection
def selection(population, fitness_scores):
    selected = []
    for _ in range(POPULATION_SIZE):
        tournament = random.sample(list(zip(population, fitness_scores)), k=5)
        tournament.sort(key=lambda x: x[1], reverse=True)
        selected.append(tournament[0][0])
    return selected

# Crossover: Single-point crossover
def crossover(parent1, parent2):
    crossover_point = random.choice(list(HYPERPARAMETER_RANGES.keys()))
    child = parent1.copy()
    child[crossover_point] = parent2[crossover_point]
    return child

# Mutation: Random mutation within the hyperparameter range
def mutation(chromosome):
    mutation_point = random.choice(list(HYPERPARAMETER_RANGES.keys()))
    chromosome[mutation_point] = random.randint(*HYPERPARAMETER_RANGES[mutation_point])
    return chromosome

# Genetic Algorithm
def genetic_algorithm(X_train, y_train, X_test, y_test):
    population = initialize_population()

    for generation in range(GENERATIONS):
        fitness_scores = [fitness(chromosome, X_train, y_train, X_test, y_test) for chromosome in population]
        best_fitness = max(fitness_scores)
        print(f'Generation {generation}: Best Fitness = {best_fitness}')

        parents = selection(population, fitness_scores)
        next_generation = []

        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)
            offspring = crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                offspring = mutation(offspring)
            next_generation.append(offspring)

        population = next_generation

    best_chromosome = max(population, key=lambda chromo: fitness(chromo, X_train, y_train, X_test, y_test))
    return best_chromosome

# Example usage
file_path = 'your_data.csv'
X_train, X_test, y_train, y_test = load_data(file_path)
best_hyperparameters = genetic_algorithm(X_train, y_train, X_test, y_test)
print("Best hyperparameters:", best_hyperparameters)
