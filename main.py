from individual import Individual
from operations import crossover, mutate, tournament_selection
from gan import Generator, Discriminator
from train import train_gan

n_items = 100  # number of items
pop_size = 50  # population size
t_max = 1000  # maximum number of generations
noise_dim = 10  # dimension of the random noise
hidden_dim = 50  # dimension of the hidden layer in GAN
batch_size = 10  # batch size for training GAN
n_epochs = 10  # number of epochs for training GAN

# Initialize the population
population = [Individual() for _ in range(pop_size)]

# Initialize GAN
G = Generator(noise_dim, hidden_dim, n_items)
D = Discriminator(n_items, hidden_dim)

t = 0

while t < t_max:
    # Generate synthetic packing assignments using G
    synthetic_population = [Individual(G.generate(torch.randn(noise_dim)).tolist()) for _ in range(pop_size)]
  
    # Combine P(t) and synthetic_population to form combined_population
    combined_population = population + synthetic_population

    # Perform selection on combined_population based on fitness function
    selected_population = [tournament_selection(combined_population, 5) for _ in range(len(combined_population))]

    # Perform crossover operation
    for i in range(0, len(selected_population), 2):
        selected_population[i], selected_population[i+1] = crossover(selected_population[i], selected_population[i+1])

    # Perform mutation operation
    for individual in selected_population:
        mutate(individual)

    # Update population
    population = selected_population

    # Train GAN with the current population
    train_gan(G, D, population, noise_dim, batch_size, n_epochs)

    # Increment t
    t += 1

# Return the best packing assignment in the final population
best_individual = max(population, key=lambda ind: ind.fitness)
print("Best packing assignment:", best_individual.assignment)
