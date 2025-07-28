#!/usr/bin/env python3
"""
Simple PyTorch Toy Example

This script demonstrates basic PyTorch usage with a simple neural network
that learns to classify random data.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNet(nn.Module):
    """A simple neural network for demonstration."""

    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_toy_data(n_samples=1000, input_size=10):
    """Generate simple toy data for classification."""
    # Generate random features
    X = torch.randn(n_samples, input_size)

    # Create simple classification rule: if sum > 0, class 1, else class 0
    y = (X.sum(dim=1) > 0).long()

    return X, y


def train_model(model, X, y, epochs=100, lr=0.01):
    """Train the model and return loss history."""
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = cross_entropy(outputs, y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return losses


def plot_training(losses):
    """Plot the training loss."""
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def evaluate_model(model, X, y):
    """Evaluate the model and print accuracy."""
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y).sum().item() / y.size(0)
        print(f"Accuracy: {accuracy:.2%}")
    model.train()


def main():
    """Main function to run the toy example."""
    print("PyTorch Toy Example")
    print("=" * 30)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Prefer CUDA, then Apple Silicon. Use CPU as fallback.
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.device_count() > 0:
        device = torch.device("mps")
    print(f"Using device: {device}")

    # Generate toy data
    print("\nGenerating toy data...")
    X, y = generate_toy_data(n_samples=1000, input_size=128)
    X = X.to(device)
    y = y.to(device)

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {torch.bincount(y).tolist()}")

    # Create model
    print("\nCreating model...")
    model = SimpleNet(input_size=128, hidden_size=2048, output_size=2)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print("\nTraining model...")
    start_time = time.time()
    losses = train_model(model, X, y, epochs=100, lr=0.01)
    elapsed_time = time.time() - start_time
    print(f"Training time: {elapsed_time:.2f} seconds on device {device}")

    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, X, y)

    # Plot training progress
    print("\nPlotting training progress...")
    plot_training(losses)

    print("\nToy example completed!")


if __name__ == "__main__":
    main()
