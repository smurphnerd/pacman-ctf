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

    def __init__(self, game, args):
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

        # Bootstrap mode: use expert policy instead of neural network
        self.bootstrap_mode = args.bootstrap_iterations > 0
        if self.bootstrap_mode:
            self._init_bootstrap_agents()

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

        # Filter to only winning examples (v = 1.0)
        if hasattr(self.args, 'filter_winning_only') and self.args.filter_winning_only:
            original_count = len(examples)
            examples = [(board, pi, v) for board, pi, v in examples if v == 1.0]
            filtered_count = len(examples)
            print(f"Filtered dataset: {filtered_count}/{original_count} examples (winning only)")

            if filtered_count == 0:
                print("Warning: No winning examples found! Skipping training.")
                return

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
        # Use bootstrap (expert) policy if enabled
        if self.bootstrap_mode:
            return self._predict_bootstrap(board)

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

    def _init_bootstrap_agents(self):
        """
        Initialize staffTeam agents for bootstrap mode.
        Creates expert agents that will be queried during MCTS instead of the neural network.
        """
        from staffTeam import createTeam

        # Get initial game state for agent initialization
        init_board = self.game.getInitBoard()
        game_state = init_board["env"].game_state

        # Create two expert agents for each team (red and blue)
        # Red team: agents 0, 2 (player 1)
        # Blue team: agents 1, 3 (player -1)
        red_agents = createTeam(0, 2, True)
        blue_agents = createTeam(1, 3, False)

        self.bootstrap_agents = {
            0: red_agents[0],  # Red agent 0
            1: blue_agents[0],  # Blue agent 1
            2: red_agents[1],  # Red agent 2
            3: blue_agents[1],  # Blue agent 3
        }

        # Initialize all agents with the initial game state
        for agent in self.bootstrap_agents.values():
            agent.registerInitialState(game_state)

        print("Bootstrap mode enabled: Using staffTeam expert policy for MCTS guidance")

    def _predict_bootstrap(self, board):
        """
        Query staffTeam expert policy instead of neural network.

        Args:
            board: Board dict with spatial/scalar features and environment state

        Returns:
            pi: Expert policy vector (one-hot encoding of expert action) [5]
            v: Value estimate (always 0.0 for bootstrap mode)
        """
        # Extract game state and current agent from board
        game_state = board["env"].game_state
        current_agent = board["current_agent"]

        # Get the expert agent for current player
        agent = self.bootstrap_agents[current_agent]

        # Query expert for action
        expert_action = agent.chooseAction(game_state)

        # Convert action string to index
        action_map = {
            "North": 0,
            "South": 1,
            "East": 2,
            "West": 3,
            "Stop": 4,
        }

        action_idx = action_map[expert_action]

        # Create one-hot policy vector
        pi = np.zeros(5)
        pi[action_idx] = 1.0

        # Return policy and dummy value estimate
        # We use 0.0 for value since the expert doesn't provide value estimates
        return pi, np.array([0.0])
