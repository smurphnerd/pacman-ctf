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


class SmurphCNN(nn.Module):
    def __init__(self, num_input_channels: int, num_actions: int):
        super(SmurphCNN, self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        feature_vector_size = 256

        # --- Policy Head (Decides which action to take) ---
        self.policy_fc1 = nn.Linear(feature_vector_size, 128)
        self.policy_output = nn.Linear(128, num_actions)

        # --- Value Head (Estimates the chance of winning) ---
        self.value_fc1 = nn.Linear(feature_vector_size, 128)
        self.value_output = nn.Linear(128, 1)

    def forward(
        self,
        x: Tensor,  # "batch", "num_channels", "width", "height"
        pos: Tensor,  # "batch", 2
    ) -> Tuple[Tensor, Tensor]:  # "batch", "num_actions"
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)  # x shape: (batch, 128, H/2, W/2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)  # x shape: (batch, 256, H/4, W/4)

        # 2. Extract location-specific features for each agent in the batch
        # We need to scale the agent's coordinates to match the pooled feature map size.
        # Note: PyTorch expects (y, x) for grid indexing.
        scaled_pos = pos // 4

        # Create a list of feature vectors for each item in the batch
        batch_indices = torch.arange(x.size(0), device=x.device)
        agent_features = x[batch_indices, :, scaled_pos[:, 1], scaled_pos[:, 0]]
        # agent_features shape: (batch_size, 256)

        # 3. Pass the extracted features through the policy head
        p = F.relu(self.policy_fc1(agent_features))
        policy_logits = self.policy_output(p)  # Raw scores, apply softmax later

        # 4. Pass the extracted features through the value head
        v = F.relu(self.value_fc1(agent_features))
        value_estimate = torch.tanh(self.value_output(v))  # Output between -1 and 1

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
    return [
        SmurphAgent(firstIndex, **kwargs),
        SmurphAgent(secondIndex, **kwargs),
    ]


class SmurphAgent(CaptureAgent):
    # Shared belief distributions for opponent positions (per team)
    # Dict mapping (team_color, opponent_idx) -> 2D probability array
    # team_color is "red" or "blue"
    _opponent_beliefs: Dict[Tuple[str, int], List[List[float]]] = {}

    # Track the last timestep when beliefs were updated per team (to avoid double-updating per timestep)
    # Dict mapping team_color -> last_timestep
    _last_belief_update_timestep: Dict[str, int] = {}

    IDX_TO_DIR = {
        0: Directions.NORTH,
        1: Directions.SOUTH,
        2: Directions.EAST,
        3: Directions.WEST,
        4: Directions.STOP,
    }

    DIR_TO_IDX = {v: k for k, v in IDX_TO_DIR.items()}

    def __init__(self, index, **kwargs):
        super().__init__(index)

        self.mode: Literal["inference", "training"] = kwargs.get("mode", "inference")
        # self.device = torch.device(
        #     "cuda"
        #     if torch.cuda.is_available()
        #     else "mps"
        #     if torch.backends.mps.is_available()
        #     else "cpu"
        # )
        self.device = torch.device("cpu")

        self.nn = SmurphCNN(11, 5)
        self.nn.to(self.device)

        self.medChooseActionTimes = []
        self.jumboChooseActionTimes = []

        if self.mode == "training":
            self.episode_memory: List[Transition] = []
            self.previous_action_idx = None
            self.previous_state_graph = None

    def registerInitialState(self, gameState: GameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.graph_nodes = []
        self.graph_edge_index = []
        self.node_map = {}  # Maps (x,y) to node index

        walls = gameState.getWalls()
        width, height = walls.width, walls.height

        # Create a node for each non-wall cell
        node_idx = 0
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    self.graph_nodes.append((x, y))
                    self.node_map[(x, y)] = node_idx
                    node_idx += 1

        for current_node_idx, (x, y) in enumerate(self.graph_nodes):
            # Check the four cardinal directions for neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor_pos = (x + dx, y + dy)

                # If the neighbor is a valid, walkable node, add an edge
                if neighbor_pos in self.node_map:
                    neighbor_node_idx = self.node_map[neighbor_pos]
                    self.graph_edge_index.append([current_node_idx, neighbor_node_idx])

    def chooseAction(self, gameState: GameState):
        """
        Example implementation showing how to use belief tracking.
        """
        randXMed = torch.rand(1, 11, 25, 25, device=self.device)
        randXLarge = torch.rand(1, 11, 50, 50, device=self.device)
        randPos = torch.tensor([[1, 1]], device=self.device)

        medStart = time.time()
        self.nn(randXMed, randPos)
        medEnd = time.time()
        jumboStart = time.time()
        self.nn(randXLarge, randPos)
        jumboEnd = time.time()
        self.medChooseActionTimes.append(medEnd - medStart)
        self.jumboChooseActionTimes.append(jumboEnd - jumboStart)

        legal_actions = gameState.getLegalActions(self.index)
        return random.choice(legal_actions)

    def final(self, gameState: GameState):
        """This is called at the end of a game to trigger training."""
        raise Exception(
            f"Med: {np.array(self.medChooseActionTimes).mean()}, Jumbo: {np.array(self.jumboChooseActionTimes).mean()}"
        )

    def _get_cell_type(self, x, y, gameState):
        pos = (x, y)

        walls = gameState.getWalls()
        if x < 0 or y < 0 or x >= walls.width or y >= walls.height or walls[x][y]:
            return CellType.WALL

        # Get food for both teams
        enemy_food = self.getFood(gameState)  # Food we can eat
        ally_food = self.getFoodYouAreDefending(gameState)  # Food we defend

        # Get capsules for both teams
        enemy_capsules = self.getCapsules(gameState)  # Capsules we can eat
        ally_capsules = self.getCapsulesYouAreDefending(gameState)  # Capsules we defend

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
            if gameState.isOnRedTeam(self.index):
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

    def _get_team_key(self):
        """Returns 'red' or 'blue' for this agent's team."""
        return "red" if self.red else "blue"

    def _initialize_beliefs_if_needed(self, gameState: GameState):
        """
        Initialize opponent beliefs on first call with exact starting positions.
        """
        team_key = self._get_team_key()
        opponents = self.getOpponents(gameState)

        # Check if this team's beliefs are already initialized
        initialized = True
        for opp_idx in opponents:
            if (team_key, opp_idx) not in SmurphAgent._opponent_beliefs:
                initialized = False
                break

        if initialized:
            return

        walls = gameState.getWalls()
        width, height = walls.width, walls.height

        for opp_idx in opponents:
            # Create empty probability array
            prob_array = [[0.0 for _ in range(height)] for _ in range(width)]

            # Get opponent's exact starting position
            opp_pos = gameState.getAgentState(opp_idx).start.pos
            assert opp_pos is not None, "Start position is None"

            x, y = int(opp_pos[0]), int(opp_pos[1])
            prob_array[x][y] = 1.0  # Certain they're at starting position

            SmurphAgent._opponent_beliefs[(team_key, opp_idx)] = prob_array

        # Initialize timestep tracker for this team
        SmurphAgent._last_belief_update_timestep[team_key] = -1

    def _update_opponent_belief(self, gameState: GameState, opponent_idx: int):
        """
        Updates belief distribution for one opponent based on current observations.
        This should be called each turn to incrementally update beliefs.

        Steps:
        1. If opponent is visible, reset belief to exact position
        2. Otherwise, propagate belief forward (account for possible movement) - ONCE per timestep
        3. Apply Bayesian update using noisy distance observation - for each agent

        Args:
            gameState: Current game state
            opponent_idx: Index of opponent to update
        """
        team_key = self._get_team_key()
        walls = gameState.getWalls()
        width, height = walls.width, walls.height

        current_timestep = len(self.observationHistory) - 1
        last_update = SmurphAgent._last_belief_update_timestep.get(team_key, -1)
        need_propagation = last_update < current_timestep

        # Check if opponent is visible
        opp_pos = gameState.getAgentState(opponent_idx).getPosition()

        if opp_pos is not None:
            # EXACT OBSERVATION - reset belief to certain position
            prob_array = [[0.0 for _ in range(height)] for _ in range(width)]
            x, y = int(opp_pos[0]), int(opp_pos[1])
            prob_array[x][y] = 1.0
            SmurphAgent._opponent_beliefs[(team_key, opponent_idx)] = prob_array
            return

        # PARTIAL OBSERVATION - update using noisy distance
        old_beliefs = SmurphAgent._opponent_beliefs[(team_key, opponent_idx)]

        # Step 1: Propagate beliefs forward (account for movement) - ONLY ONCE PER TIMESTEP
        if need_propagation:
            new_beliefs = [[0.0 for _ in range(height)] for _ in range(width)]

            for x in range(width):
                for y in range(height):
                    if old_beliefs[x][y] > 0:
                        # From position (x,y), distribute probability to reachable neighbors
                        neighbors = Actions.getLegalNeighbors((x, y), walls)
                        prob_per_neighbor = old_beliefs[x][y] / len(neighbors)

                        for nx, ny in neighbors:
                            new_beliefs[nx][ny] += prob_per_neighbor

            # Update the class variable with propagated beliefs
            SmurphAgent._opponent_beliefs[(team_key, opponent_idx)] = new_beliefs
            old_beliefs = new_beliefs

        # Step 2: Bayesian update using noisy distance observation (each agent does this)
        my_pos = gameState.getAgentPosition(self.index)
        noisy_distances = gameState.getAgentDistances()
        assert noisy_distances is not None, "No noisy distances"
        noisy_dist = noisy_distances[opponent_idx]

        updated_beliefs = [[0.0 for _ in range(height)] for _ in range(width)]
        for x in range(width):
            for y in range(height):
                if old_beliefs[x][y] > 0:
                    # P(pos | observation) ∝ P(observation | pos) * P(pos)
                    prior = old_beliefs[x][y]
                    true_dist = self.getMazeDistance(my_pos, (x, y))
                    likelihood = gameState.getDistanceProb(true_dist, noisy_dist)
                    updated_beliefs[x][y] = prior * likelihood

        # Step 3: Normalize
        total_prob = sum(sum(row) for row in updated_beliefs)
        if total_prob > 0:
            for x in range(width):
                for y in range(height):
                    updated_beliefs[x][y] /= total_prob

        SmurphAgent._opponent_beliefs[(team_key, opponent_idx)] = updated_beliefs

    def _get_opponent_pos_dist(self, opponent_idx: int):
        """
        Returns the current belief distribution for an opponent.
        Call _update_opponent_belief() first to ensure beliefs are current.

        Args:
            opponent_idx: Index of opponent agent

        Returns:
            2D array where [x][y] = probability opponent is at (x, y)
        """
        team_key = self._get_team_key()
        return SmurphAgent._opponent_beliefs.get((team_key, opponent_idx))

    def _get_graph_state(self, gameState: GameState):
        """
        Creates the node feature matrix for the current gameState.
        """
        num_nodes = len(self.graph_nodes)

        # Define the size of your feature vector.
        # One-hot CellType (7) + my_agent (1) + teammate (1) + opp1_prob (1) + opp2_prob (1) = 11 features
        num_features = 7 + 4
        node_features = [[0.0] * num_features for _ in range(num_nodes)]

        # Get agent and opponent indices
        my_idx = self.index
        teammate_idx = self._get_teammate_idx()
        opponents = self.getOpponents(gameState)
        opp1_idx, opp2_idx = opponents[0], opponents[1]

        # Get opponent belief distributions
        opp1_belief = self._get_opponent_pos_dist(opp1_idx)
        opp2_belief = self._get_opponent_pos_dist(opp2_idx)

        # Get agent positions
        my_pos = gameState.getAgentPosition(my_idx)
        teammate_pos = gameState.getAgentPosition(teammate_idx)

        for i, (x, y) in enumerate(self.graph_nodes):
            # --- Static Features (One-Hot Encoded) ---
            cell_type = self._get_cell_type(x, y, gameState)
            node_features[i][cell_type.value] = 1.0

            # --- Dynamic Agent Features ---

            # My agent's position
            if (x, y) == my_pos:
                node_features[i][7] = 1.0

            # Teammate's position
            if (x, y) == teammate_pos:
                node_features[i][8] = 1.0

            # --- Dynamic Opponent Belief Features ---
            if opp1_belief:
                node_features[i][9] = opp1_belief[x][y]

            if opp2_belief:
                node_features[i][10] = opp2_belief[x][y]

        return node_features
