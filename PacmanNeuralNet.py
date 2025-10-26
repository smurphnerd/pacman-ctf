"""
PacmanNeuralNet.py - Wrapper for SmurphCNN implementing alpha-zero-general's NeuralNet API.

This adapter allows our CNN to work with the AlphaZero training pipeline.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from NeuralNet import NeuralNet
from pacman_args import TrainingArgs
from myTeam import SmurphCNN

try:
    profile
except NameError:

    def profile(func):
        return func


class PacmanDataset(Dataset):
    """
    Dataset for training from self-play examples.
    """

    def __init__(self, examples):
        """
        Args:
            examples: List of (board, pi, v) tuples
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        board, pi, v = self.examples[idx]

        # Extract spatial and scalar features from board dict
        spatial = torch.FloatTensor(board["spatial"])
        scalar = torch.FloatTensor(board["scalar"])
        pi = torch.FloatTensor(pi)
        v = torch.FloatTensor([v])

        return spatial, scalar, pi, v


class PacmanNeuralNet(NeuralNet):
    """
    Wrapper for SmurphCNN implementing NeuralNet interface.
    """

    def __init__(self, game, args: TrainingArgs):
        """
        Args:
            game: PacmanGame instance
            args: Training arguments (lr, epochs, batch_size, etc.)
        """
        self.game = game
        self.args = args

        # Initialize SmurphCNN
        self.nnet = SmurphCNN(num_actions=game.getActionSize())

        # Setup device (force CPU to avoid MPS segfault on Mac)
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.nnet.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.nnet.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    @profile
    def train(self, examples):
        """
        Train the neural network on examples from self-play.

        Args:
            examples: List of (board, pi, v) tuples where:
                - board: Board dict with spatial/scalar features
                - pi: MCTS policy (action probabilities) [5]
                - v: Value from current board's perspective [-1, 1]
        """
        self.nnet.train()

        # Create dataset and dataloader
        dataset = PacmanDataset(examples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for debugging, increase for speed
        )

        # Training parameters
        epochs = self.args.epochs

        for epoch in range(epochs):
            total_loss = 0
            policy_loss_sum = 0
            value_loss_sum = 0
            batches = 0

            for spatial, scalar, target_pi, target_v in dataloader:
                # Move to device
                spatial = spatial.to(self.device)
                scalar = scalar.to(self.device)
                target_pi = target_pi.to(self.device)
                target_v = target_v.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                policy_logits, value = self.nnet(spatial, scalar)

                # Losses
                policy_loss = -torch.sum(
                    target_pi * torch.log_softmax(policy_logits, dim=1)
                ) / target_pi.size(0)
                value_loss = torch.sum(
                    (value.squeeze() - target_v.squeeze()) ** 2
                ) / target_v.size(0)

                loss = policy_loss + value_loss

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Track losses
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                batches += 1

            # Print epoch summary
            avg_loss = total_loss / batches
            avg_policy_loss = policy_loss_sum / batches
            avg_value_loss = value_loss_sum / batches

            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Loss={avg_loss:.4f}, "
                f"Policy={avg_policy_loss:.4f}, "
                f"Value={avg_value_loss:.4f}"
            )

    @profile
    def predict(self, board):
        """
        Predict policy and value for a given board.

        Args:
            board: Board dict with spatial/scalar features

        Returns:
            pi: Policy vector (action probabilities) [5]
            v: Value estimate [-1, 1]
        """
        self.nnet.eval()

        with torch.no_grad():
            # Extract features
            spatial = torch.FloatTensor(board["spatial"]).unsqueeze(0)  # Add batch dim
            scalar = torch.FloatTensor(board["scalar"]).unsqueeze(0)  # Add batch dim

            spatial = spatial.to(self.device)
            scalar = scalar.to(self.device)

            # Forward pass
            policy_logits, value = self.nnet(spatial, scalar)

            # Convert to probabilities
            pi = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            v = value.cpu().numpy()[0]

        return pi, v

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        """
        Save model checkpoint.

        Args:
            folder: Directory to save checkpoint
            filename: Checkpoint filename
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(
            {
                "state_dict": self.nnet.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filepath,
        )

        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        """
        Load model checkpoint.

        Args:
            folder: Directory containing checkpoint
            filename: Checkpoint filename
        """
        filepath = os.path.join(folder, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"Checkpoint loaded from {filepath}")
