import sys
import os

import json
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image

# Enhanced Variational Auto-Encoder with Convolutional Networks

class EnhancedVAE(nn.Module):
    """
    An enhanced version of Variational Auto-Encoder (VAE) utilizing convolutional layers.
    Designed for processing and generating images of even MNIST numbers.
    """
    def __init__(self, input_channels=1, feature_dims=64, image_size=14):
        super(EnhancedVAE, self).__init__()
        # Encoder layers
        self.encoder_conv1 = nn.Conv2d(input_channels, feature_dims, kernel_size=4, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv2d(feature_dims, feature_dims * 2, kernel_size=4, stride=2, padding=1)
        self.encoder_fc1 = nn.Linear(feature_dims * 18, image_size)
        self.encoder_fc2 = nn.Linear(feature_dims * 18, image_size)

        # Decoder layers
        self.decoder_fc = nn.Linear(image_size, feature_dims * 18)
        self.decoder_deconv1 = nn.ConvTranspose2d(feature_dims * 2, feature_dims, kernel_size=5, stride=2, padding=1)
        self.decoder_deconv2 = nn.ConvTranspose2d(feature_dims, input_channels,  kernel_size=4, stride=2, padding=1)

    def encode(self, input_tensor):
        """Pass the input through the encoder layers and return the mean and log variance."""
        x = F.relu(self.encoder_conv1(input_tensor))
        x = F.relu(self.encoder_conv2(x))
        x = x.view(x.size(0), -1)
        mean = self.encoder_fc1(x)
        log_variance = self.encoder_fc2(x)
        return mean, log_variance

    def reparameterize(self, mean, log_variance):
        """Apply reparameterization trick to sample from a normal distribution."""
        std = torch.exp(log_variance / 2)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std if self.training else mean

    def decode(self, z):
        """Reconstruct the image from the latent space representation."""
        x = F.relu(self.decoder_fc(z))
        x = x.view(x.size(0), 64 * 2, 3, 3)  # Reshape to fit the deconvolutional layers
        x = F.relu(self.decoder_deconv1(x))
        x = torch.sigmoid(self.decoder_deconv2(x))
        return x

    def forward(self, input_tensor):
        """Defines the forward pass of the model."""
        mean, log_variance = self.encode(input_tensor)
        z = self.reparameterize(mean, log_variance)
        reconstructed_img = self.decode(z)
        return reconstructed_img, mean, log_variance

class MNISTEvenDataset:
    """
    A custom dataset class for MNIST even numbers.
    Processes the input data for training with PyTorch models, normalizing images
    and preparing labels for the Variational Auto-Encoder (VAE).
    """
    def __init__(self, dataset):
        """
        Initializes the dataset with MNIST data.

        Parameters:
        - dataset: A NumPy array with images as rows, the last column being the label.
        """
        self.dataset = dataset
        self.images = dataset[:, :-1]
        self.normalized_images = torch.from_numpy(self.images).float() / 255.0  # Normalize the images to [0, 1]
        self.labels = dataset[:, -1]  # The labels are in the last column

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, index):
        """
        Retrieves an item at the specified index.

        Parameters:
        - index: Index of the item to retrieve.

        Returns:
        - A tuple (image, label) where `image` is a normalized image tensor and
          `label` is the corresponding label as a floating-point number.
        """
        label = self.labels[index]
        # Reshape the image to 2D (14x14) and add a channel dimension, converting to a PyTorch tensor
        image = torch.from_numpy(np.expand_dims(self.images[index].reshape(14, 14), 0)).float()
        return image, label

if __name__ == "__main__":

    # Check for help command
    if sys.argv[1] == "--help":
        print("Usage: python main.py -o results_dir -n 100")
        print("Options:")
        print("  -o results_dir     Output directory for the generated images.")
        print("                     Default: 'results'")
        print("  -n 100             Number of images to generate for testing.")
        print("                     Default: 100")
    else:
        # Parse command line arguments
        if len(sys.argv) == 5: 
            output_dir = sys.argv[2]
            num_images = int(sys.argv[4])
        else:
            output_dir = "results"
            num_images = 100

        # Path to the parameters JSON file
        params_file = "param/param_file_name.json" 
        dataset_path = "data/even_mnist.csv"
        
        # Load model parameters from JSON
        with open(params_file) as file:
            params = json.load(file)
            learning_rate = params['learning_rate']
            batch_size = params['batch_size']
            num_epochs = params['num_epochs']
            latent_dim = params['latent_dim']
            image_size = params['image_size']
            verbose = params['verbose_mode']
            input_channels = 1  # Assuming black and white images

        # Create the output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load and preprocess the dataset
        data = np.genfromtxt(dataset_path, delimiter=" ")
        np.random.shuffle(data)  # Shuffle for randomness
        
        # Split data into training and testing
        test_data = data[-3000:, :]
        train_data = data[:-len(test_data), :] 
        training_dataset = MNISTEvenDataset(train_data)
        testing_dataset = MNISTEvenDataset(test_data)
        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
       
        # Initialize the model and optimizer
        vae_model = EnhancedVAE(input_channels, latent_dim, image_size).to(device)
        optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)

        # Training loop
        print(f"Verbose Mode: {verbose}")
        training_losses = []
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}...")
            vae_model.train()
            total_loss = 0
            for _, batch in enumerate(train_loader):
                images, _ = batch
                images = images.to(device)

                # Forward pass through the model
                reconstructed, mean, log_var = vae_model(images)
                kl_divergence = 0.5 * torch.sum(-1 - log_var + mean.pow(2) + log_var.exp())
                loss = F.binary_cross_entropy(reconstructed, images, reduction='sum') + kl_divergence 
                total_loss += loss.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(train_loader.dataset)
            training_losses.append(avg_loss)
            if verbose: print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")

        # Plot and save training loss
        plt.figure(figsize=(10, 6))
        plt.plot(training_losses, label="Training Loss")
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{output_dir}/training_loss.png")
        if verbose:
            plt.show()

        # Generate and save images
        print("Generating sample images...")
        vae_model.eval()
        with torch.no_grad():
            for i in range(num_images):
                sample = random.choice(test_loader.dataset)
                image, _ = sample
                image = image.unsqueeze(0).to(device)
                output, _, _ = vae_model(image)
                output_image = output.squeeze().cpu().numpy()
                plt.imsave(f"{output_dir}/{i+1}.pdf", output_image, cmap='gray')

        print("********* Process Completed *********")
