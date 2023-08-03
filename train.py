import torch
from torch import nn
from gan import Generator, Discriminator

def train_gan(generator, discriminator, population, noise_dim, batch_size, n_epochs):
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters())
    optimizer_D = torch.optim.Adam(discriminator.parameters())
  
    for epoch in range(n_epochs):
        for i in range(0, len(population), batch_size):
            # Train discriminator
            real_data = torch.Tensor([ind.assignment for ind in population[i:i+batch_size]])
            noise = torch.randn(batch_size, noise_dim)
            fake_data = generator(noise)
            real_output = discriminator(real_data)
            fake_output = discriminator(fake_data)
            loss_D = criterion(real_output, torch.ones_like(real_output)) + criterion(fake_output, torch.zeros_like(fake_output))
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
    
            # Train generator
            noise = torch.randn(batch_size, noise_dim)
            fake_data = generator(noise)
            output = discriminator(fake_data)
            loss_G = criterion(output, torch.ones_like(output))
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
