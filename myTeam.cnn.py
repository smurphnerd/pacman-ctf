# myTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# myTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from typing import List, Tuple, Dict, Literal, Optional
from dataclasses import dataclass
from enum import IntEnum

from capture import COLLISION_TOLERANCE
from captureAgents import CaptureAgent
import random, math, uuid, time
from capture import GameState
from game import Directions, Actions
from util import manhattanDistance

import torch
import torch.nn as nn
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import belief_tracking


# torch.set_num_threads(1)


class SmurphCNN(nn.Module):
    def __init__(
        self,
        num_actions: int = 5,
        num_spatial_features: int = 11,
        num_scalar_features: int = 14,
    ):
        super(SmurphCNN, self).__init__()

        self.num_scalar_features = num_scalar_features

        # Convolutional layers (process spatial input)
        self.conv1 = nn.Conv2d(num_spatial_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Combined features: 256 (spatial) + num_scalar_features (non-spatial)
        combined_dim = 256 + num_scalar_features

        # Policy head
        self.policy_fc1 = nn.Linear(combined_dim, 128)
        self.policy_output = nn.Linear(128, num_actions)

        # Value head
        self.value_fc1 = nn.Linear(combined_dim, 128)
        self.value_output = nn.Linear(128, 1)

    def forward(
        self,
        spatial_input: Tensor,  # "batch", "num_channels", "width", "height"
        scalar_input: Tensor,  # "batch", "num_scalar_features"
    ) -> Tuple[Tensor, Tensor]:  # "batch", "num_actions"
        # Process spatial features through CNN
        x = F.relu(self.conv1(spatial_input))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Global pooling → [batch, 256]
        spatial_features = self.global_max_pool(x).squeeze()

        # Handle batch dimension edge cases
        if spatial_features.dim() == 1:
            spatial_features = spatial_features.unsqueeze(0)

        # Concatenate spatial and scalar features
        scalar_input = scalar_input.reshape(-1, self.num_scalar_features)
        combined = torch.cat([spatial_features, scalar_input], dim=1)

        # Policy head
        p = F.relu(self.policy_fc1(combined))
        policy_logits = self.policy_output(p)

        # Value head
        v = F.relu(self.value_fc1(combined))
        value_estimate = torch.tanh(self.value_output(v))

        return policy_logits, value_estimate


class Transition:
    """Experience tuple for replay buffer - plain class for better pickling"""

    __slots__ = ("state", "action_probs", "value")

    def __init__(
        self,
        state: Tensor,  # "num_channels", "width", "height"
        action_probs: Tensor,  # "num_actions"
        value: Tensor,  # "1"
    ):
        self.state = state
        self.action_probs = action_probs
        self.value = value

    def __iter__(self):
        """Allow unpacking like a tuple"""
        return iter((self.state, self.action_probs, self.value))


class CellType(IntEnum):
    WALL = 0
    EMPTY_ALLY_TERRITORY = 1  # Empty space on our side
    ALLY_FOOD = 2  # Food we need to defend
    ALLY_CAPSULE = 3  # Capsules we need to defend
    EMPTY_ENEMY_TERRITORY = 4  # Empty space on enemy side
    ENEMY_FOOD = 5  # Food we can eat
    ENEMY_CAPSULE = 6  # Capsules we can eat


def createTeam(firstIndex, secondIndex, isRed, **kwargs):
    shared_opponent_pos_dist = {}
    return [
        SmurphAgent(firstIndex, shared_opponent_pos_dist, **kwargs),
        SmurphAgent(secondIndex, shared_opponent_pos_dist, **kwargs),
    ]


class SmurphAgent(CaptureAgent):
    IDX_TO_DIR = {
        0: Directions.NORTH,
        1: Directions.SOUTH,
        2: Directions.EAST,
        3: Directions.WEST,
        4: Directions.STOP,
    }

    DIR_TO_IDX = {v: k for k, v in IDX_TO_DIR.items()}

    def __init__(self, index, shared_opponent_pos_dist, **kwargs):
        super().__init__(index)

        self.mode: Literal["inference", "training"] = kwargs.get("mode", "inference")
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.shared_opponent_pos_dist = shared_opponent_pos_dist

        self.network = SmurphCNN(5)
        self.network.to(self.device)
        checkpoint = torch.load(
            "temp/best.pth.tar", map_location=self.device
        )
        self.network.load_state_dict(checkpoint["state_dict"])

    def registerInitialState(self, gameState: GameState):
        """Called once at the start of the game to perform expensive precomputation."""
        CaptureAgent.registerInitialState(self, gameState)

        self.max_food = len(self.getFood(gameState).asList())
        self.max_food_defending = len(self.getFoodYouAreDefending(gameState).asList())
        self.max_score = max(self.max_food, self.max_food_defending)

        self.width, self.height = (
            gameState.data.layout.width,
            gameState.data.layout.height,
        )
        self.time_limit = gameState.data.timeleft

        # Initialize shared beliefs using centralized function
        # Only initialize once per team (both teammates share the same dict)
        if not self.shared_opponent_pos_dist:
            beliefs = belief_tracking.initialize_beliefs(gameState)
            # Only store beliefs for our opponents
            opponents = self.getOpponents(gameState)
            for opp_idx in opponents:
                self.shared_opponent_pos_dist[opp_idx] = beliefs[opp_idx]

    def chooseAction(self, gameState: GameState):
        """
        Example implementation showing how to use belief tracking.
        """
        # Update beliefs using centralized function
        updated_beliefs = belief_tracking.update_all_beliefs(
            self.shared_opponent_pos_dist,
            gameState,
            self.index,
        )

        # Update the shared dict (teammates will see updated beliefs)
        opponents = self.getOpponents(gameState)
        for opp_idx in opponents:
            self.shared_opponent_pos_dist[opp_idx] = updated_beliefs[opp_idx]

        state = self._get_state(gameState)
        policy_logits, _ = self.network(state[0], state[1])
        max_action = torch.argmax(policy_logits)
        legal_actions = gameState.getLegalActions(self.index)

        if max_action in legal_actions:
            return max_action

        return random.choice(legal_actions)

    def final(self, gameState: GameState):
        """This is called at the end of a game to trigger training."""
        pass

    def _get_state(self, gameState: GameState) -> Tuple[Tensor, Tensor]:
        """
        Creates tensor representations of the game state for CNN input.

        Returns:
            spatial_state: [11, width, height] tensor with spatial features
                - Channels 0-6: One-hot encoding for CellType
                - Channel 7: Opponent 1 belief distribution
                - Channel 8: Opponent 2 belief distribution
                - Channel 9: Teammate's position
                - Channel 10: My agent's position
            scalar_state: [14] tensor with non-spatial features
                - Scared timers (4 values)
                - Food carrying (4 values)
                - Pacman status (4 values)
                - Time remaining (1 value)
                - Score from my perspective (1 value)
        """
        width, height = gameState.data.layout.width, gameState.data.layout.height

        # Initialize 11-channel tensor (all zeros)
        num_channels = 11
        state_tensor = np.zeros((num_channels, width, height), dtype=np.float32)

        # Determine if we need to flip (Blue team's perspective)
        # For AlphaZero self-play, Blue agents see flipped board as if they were Red
        is_blue = self.index in [1, 3]

        # Get agent and opponent indices
        my_idx = self.index
        teammate_idx = self._get_teammate_idx()
        opponents = self.getOpponents(gameState)
        opp1_idx, opp2_idx = opponents[0], opponents[1]

        # Get opponent belief distributions
        opp1_belief = self.shared_opponent_pos_dist.get(opp1_idx)
        assert (
            opp1_belief is not None
        ), "_get_state: Opponent 1 belief distribution should not be None"
        opp2_belief = self.shared_opponent_pos_dist.get(opp2_idx)
        assert (
            opp2_belief is not None
        ), "_get_state: Opponent 2 belief distribution should not be None"

        # Get agent positions
        my_pos = gameState.getAgentPosition(my_idx)
        teammate_pos = gameState.getAgentPosition(teammate_idx)
        assert (
            teammate_pos is not None
        ), "_get_state: Teammate position should not be None"

        # Cache food/capsules/walls once before the loops (MAJOR OPTIMIZATION)
        walls = gameState.getWalls()
        enemy_food = self.getFood(gameState)
        ally_food = self.getFoodYouAreDefending(gameState)
        enemy_capsules = self.getCapsules(gameState)
        ally_capsules = self.getCapsulesYouAreDefending(gameState)

        # Fill in the tensor for each position
        # Iterate over observation coordinates, map to actual board positions
        for x in range(width):
            for y in range(height):
                # Get actual position (flip if Blue)
                actual_x = self.transform_pos((x, y))[0]

                # --- Channels 0-6: One-hot encoding for CellType ---
                cell_type = self._get_cell_type(
                    (actual_x, y),
                    walls,
                    enemy_food,
                    ally_food,
                    enemy_capsules,
                    ally_capsules,
                )
                state_tensor[cell_type.value, x, y] = 1.0

                # --- Channel 7: Opponent 1 belief distribution ---
                if opp1_belief:
                    state_tensor[7, x, y] = opp1_belief[actual_x][y]

                # --- Channel 8: Opponent 2 belief distribution ---
                if opp2_belief:
                    state_tensor[8, x, y] = opp2_belief[actual_x][y]

        # --- Channel 9: Teammate's position ---
        teammate_x, teammate_y = self.transform_pos(teammate_pos)
        state_tensor[9, teammate_x, teammate_y] = 1.0

        # --- Channel 10: My agent's position ---
        my_x, my_y = self.transform_pos(my_pos)
        state_tensor[10, my_x, my_y] = 1.0

        # ============ Scalar Features ============

        # Get agent states
        my_state = gameState.getAgentState(my_idx)
        teammate_state = gameState.getAgentState(teammate_idx)
        opp1_state = gameState.getAgentState(opp1_idx)
        opp2_state = gameState.getAgentState(opp2_idx)

        # Build scalar features (normalized to [0, 1] or [-1, 1])
        scalar_features = [
            # Scared timers (0-40) → normalize to [0, 1]
            my_state.scaredTimer / 40.0,
            teammate_state.scaredTimer / 40.0,
            opp1_state.scaredTimer / 40.0,
            opp2_state.scaredTimer / 40.0,
            # Food carrying (0 to max_food) → normalize to [0, 1]
            my_state.numCarrying / float(self.max_food),
            teammate_state.numCarrying / float(self.max_food),
            opp1_state.numCarrying / float(self.max_food_defending),
            opp2_state.numCarrying / float(self.max_food_defending),
            # Pacman status (0 or 1)
            float(my_state.isPacman),
            float(teammate_state.isPacman),
            float(opp1_state.isPacman),
            float(opp2_state.isPacman),
            # Time remaining (0-1200) → normalize to [0, 1]
            gameState.data.timeleft / self.time_limit,
            # Score (from my perspective, -max_score to +max_score) → normalize to [-1, 1]
            gameState.getScore() / float(self.max_score) * (1.0 if self.red else -1.0),
        ]

        spatial_tensor = torch.tensor(
            state_tensor, dtype=torch.float32, device=self.device
        )
        scalar_tensor = torch.tensor(
            scalar_features, dtype=torch.float32, device=self.device
        )

        return spatial_tensor, scalar_tensor

    def _get_cell_type(
        self,
        pos: Tuple[int, int],
        walls,
        enemy_food,
        ally_food,
        enemy_capsules,
        ally_capsules,
    ):
        x, y = pos

        if x < 0 or y < 0 or x >= walls.width or y >= walls.height or walls[x][y]:
            return CellType.WALL

        # Check capsules (territory-specific)
        if pos in ally_capsules:
            return CellType.ALLY_CAPSULE
        elif pos in enemy_capsules:
            return CellType.ENEMY_CAPSULE

        # Check food (territory-specific)
        elif ally_food[x][y]:
            return CellType.ALLY_FOOD
        elif enemy_food[x][y]:
            return CellType.ENEMY_FOOD

        # Empty space - determine territory
        else:
            width = walls.width
            midpoint = width // 2

            # Red team controls left half (x < midpoint)
            if self.red:
                if x < midpoint:
                    return CellType.EMPTY_ALLY_TERRITORY
                else:
                    return CellType.EMPTY_ENEMY_TERRITORY

            # Blue team controls right half (x >= midpoint)
            else:
                if x >= midpoint:
                    return CellType.EMPTY_ALLY_TERRITORY
                else:
                    return CellType.EMPTY_ENEMY_TERRITORY

    def _get_teammate_idx(self):
        return (self.index + 2) % 4

    def transform_pos(self, pos) -> Tuple[int, int]:
        if self.red:
            return pos

        return (self.width - 1 - pos[0], pos[1])
