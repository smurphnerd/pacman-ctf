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

from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import IntEnum

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os
from capture import GameState, noisyDistance
from game import Directions, Actions, AgentState, Agent, Grid
from util import nearestPoint
import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool


class SmurphGNN(nn.Module):
    def __init__(self, num_node_features: int, num_actions: int, hidden_channels=32):
        super(SmurphGNN, self).__init__()
        # Use a small MLP to encode the initial node features
        self.node_encoder = nn.Linear(num_node_features, hidden_channels)

        # GraphSAGE layers for message passing
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # --- Output Heads ---
        # 1. Policy head: Decides which action to take
        self.policy_head = nn.Linear(hidden_channels, num_actions)

        # 2. Value head: Estimates the quality of the current state
        self.value_head = nn.Linear(hidden_channels, 1)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Encode initial node features
        x = self.node_encoder(x)
        x = F.relu(x)

        # 2. Perform message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # 3. Get the embedding for the current agent's node
        # We need to know which node in the graph corresponds to our agent
        agent_node_embedding = x[data.node_idx]

        # 4. Calculate policy and value
        action_logits = self.policy_head(agent_node_embedding)

        # For the state value, we can use a global representation
        # by pooling all node embeddings
        global_graph_embedding = global_mean_pool(x, batch)
        state_value = self.value_head(global_graph_embedding)

        # Return action probabilities and the state value
        return F.softmax(action_logits, dim=-1), torch.tanh(state_value)


class CellType(IntEnum):
    WALL = 0
    EMPTY_ALLY_TERRITORY = 1  # Empty space on our side
    ALLY_FOOD = 2  # Food we need to defend
    ALLY_CAPSULE = 3  # Capsules we need to defend
    EMPTY_ENEMY_TERRITORY = 4  # Empty space on enemy side
    ENEMY_FOOD = 5  # Food we can eat
    ENEMY_CAPSULE = 6  # Capsules we can eat


def createTeam(firstIndex, secondIndex, isRed, **kwargs):
    first_config = kwargs.get("first")
    second_config = kwargs.get("second")

    return [
        SmurphAgent(firstIndex, first_config),
        SmurphAgent(secondIndex, second_config),
    ]


@dataclass
class SmurphAgentConfig:
    name: str
    model: SmurphGNN
    alpha: float  # learning rate
    discount_rate: float
    epsilon: float  # exploration prob


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

    def final(self, gameState: GameState):
        pass

    def chooseAction(self, gameState: GameState):
        """
        Example implementation showing how to use belief tracking.
        """
        if len(self.observationHistory) <= 1:
            # Initialize beliefs on first call
            self._initialize_beliefs_if_needed(gameState)

        # Update beliefs for all opponents
        for opponent_idx in self.getOpponents(gameState):
            self._update_opponent_belief(gameState, opponent_idx)

        # Mark this timestep as updated for this team (after all opponents processed)
        current_timestep = len(self.observationHistory) - 1
        team_key = self._get_team_key()
        self._last_belief_update_timestep[team_key] = current_timestep

        node_features = self._get_graph_state(gameState)
        my_pos = gameState.getAgentPosition(self.index)
        assert my_pos in self.node_map

        # Get the node index of my agent
        my_idx = self.node_map[my_pos]

        x = torch.tensor(node_features, dtype=torch.float, device=self.device)
        edge_index = torch.tensor(
            self.graph_edge_index, dtype=torch.long, device=self.device
        ).T.contiguous()

        data = Data(x=x, edge_index=edge_index, node_idx=my_idx)

        self.model.eval()
        with torch.no_grad():
            action_probs, state_value = self.model(data)
            action_probs = action_probs.squeeze(0)

        legal_actions = gameState.getLegalActions(self.index)

        best_action = None
        max_prob = -float("inf")

        for action in legal_actions:
            action_idx = self.DIR_TO_IDX.get(action)
            prob = action_probs[action_idx].item()
            if prob > max_prob:
                max_prob = prob
                best_action = action

        assert best_action is not None

        return best_action

    def __init__(self, index, config_path):
        super().__init__(index)

        if config_path:
            config = torch.load(config_path)
            assert isinstance(config, SmurphAgentConfig)
            self.config = config
        else:
            model = SmurphGNN(num_node_features=11, num_actions=5)

            self.config = SmurphAgentConfig(
                name="SmurphAgent",
                model=model,
                alpha=0.1,
                discount_rate=0.99,
                epsilon=0.1,
            )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model = self.config.model
        self.model.to(self.device)

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
            if (team_key, opp_idx) not in self._opponent_beliefs:
                initialized = False
                break

        if initialized:
            return

        walls = gameState.getWalls()
        width, height = walls.width, walls.height

        for opp_idx in opponents:
            # Create empty probability array
            prob_array = [[0.0 for y in range(height)] for x in range(width)]

            # Get opponent's exact starting position
            opp_pos = gameState.getAgentState(opp_idx).start.pos
            assert opp_pos is not None, "Start position is None"

            x, y = int(opp_pos[0]), int(opp_pos[1])
            prob_array[x][y] = 1.0  # Certain they're at starting position

            self._opponent_beliefs[(team_key, opp_idx)] = prob_array

        # Initialize timestep tracker for this team
        self._last_belief_update_timestep[team_key] = -1

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
        last_update = self._last_belief_update_timestep.get(team_key, -1)
        need_propagation = last_update < current_timestep

        # Check if opponent is visible
        opp_pos = gameState.getAgentState(opponent_idx).getPosition()

        if opp_pos is not None:
            # EXACT OBSERVATION - reset belief to certain position
            prob_array = [[0.0 for y in range(height)] for x in range(width)]
            x, y = int(opp_pos[0]), int(opp_pos[1])
            prob_array[x][y] = 1.0
            self._opponent_beliefs[(team_key, opponent_idx)] = prob_array
            return

        # PARTIAL OBSERVATION - update using noisy distance
        old_beliefs = self._opponent_beliefs[(team_key, opponent_idx)]

        # Step 1: Propagate beliefs forward (account for movement) - ONLY ONCE PER TIMESTEP
        if need_propagation:
            new_beliefs = [[0.0 for y in range(height)] for x in range(width)]

            for x in range(width):
                for y in range(height):
                    if old_beliefs[x][y] > 0:
                        # From position (x,y), distribute probability to reachable neighbors
                        neighbors = Actions.getLegalNeighbors((x, y), walls)
                        prob_per_neighbor = old_beliefs[x][y] / len(neighbors)

                        for nx, ny in neighbors:
                            new_beliefs[nx][ny] += prob_per_neighbor

            # Update the class variable with propagated beliefs
            self._opponent_beliefs[(team_key, opponent_idx)] = new_beliefs
            old_beliefs = new_beliefs

        # Step 2: Bayesian update using noisy distance observation (each agent does this)
        my_pos = gameState.getAgentPosition(self.index)
        noisy_distances = gameState.getAgentDistances()
        assert noisy_distances is not None, "No noisy distances"
        noisy_dist = noisy_distances[opponent_idx]

        updated_beliefs = [[0.0 for y in range(height)] for x in range(width)]
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

        self._opponent_beliefs[(team_key, opponent_idx)] = updated_beliefs

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
        return self._opponent_beliefs.get((team_key, opponent_idx))

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
