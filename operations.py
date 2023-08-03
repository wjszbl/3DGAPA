import random
from individual import Individual

def crossover(parent1, parent2):
    size = len(parent1.assignment)
    cross_point1 = random.randint(0, size)
    cross_point2 = random.randint(cross_point1, size)
    child1 = parent1.assignment[:cross_point1] + parent2.assignment[cross_point1:cross_point2] + parent1.assignment[cross_point2:]
    child2 = parent2.assignment[:cross_point1] + parent1.assignment[cross_point1:cross_point2] + parent2.assignment[cross_point2:]
    return Individual(child1), Individual(child2)

def mutate(individual, mutation_rate=0.01):
    individual.assignment = [gene if random.random() > mutation_rate else 1-gene for gene in individual.assignment]

def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)
