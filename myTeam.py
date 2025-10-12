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
import random, math, uuid
from capture import GameState
from game import Directions, Actions
from util import manhattanDistance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class SmurphGNN(nn.Module):
    def __init__(self, num_node_features: int, num_actions: int, hidden_channels=32):
        super(SmurphGNN, self).__init__()
        # Use a small MLP to encode the initial node features
        self.node_encoder = nn.Linear(num_node_features, hidden_channels)

        # GraphSAGE layers for message passing
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.q_head = nn.Linear(hidden_channels, num_actions)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = self.node_encoder(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        agent_node_embedding = x[data.node_idx]

        q_values = self.q_head(agent_node_embedding)
        return q_values


class Transition:
    """Experience tuple for replay buffer - plain class for better pickling"""

    __slots__ = ("state", "action", "reward", "next_state", "done")

    def __init__(
        self,
        state: Data,
        action: int,
        reward: float,
        next_state: Optional[Data],
        done: bool,
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def __iter__(self):
        """Allow unpacking like a tuple"""
        return iter((self.state, self.action, self.reward, self.next_state, self.done))


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


@dataclass
class RewardWeights:
    score_change: float
    scored_points: float
    lost_points: float
    eaten_as_scared_ghost: float
    ate_food: float
    ate_capsule: float
    teammate_scored_points: float
    teammate_ate_food: float
    saved_points: float
    ate_scared_ghost: float
    food_eaten_by_opponent: float
    capsule_eaten_by_opponent: float
    time_penalty: float


@dataclass
class SmurphAgentConfig:
    reward_weights: RewardWeights
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_decay_rate: float
    epsilon_min: float
    games_played: int


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

        self.id = kwargs.get("agentId")

        self.mode: Literal["inference", "training"] = kwargs.get("mode", "inference")
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.config: SmurphAgentConfig = torch.load(self._get_config_path())
        assert type(self.config).__name__ == "SmurphAgentConfig"

        self.policy_net = SmurphGNN(11, 5, hidden_channels=32)
        self.policy_net.load_state_dict(torch.load(self._get_weights_path()))
        self.policy_net.to(self.device)

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
        # --- 1. Belief Tracking ---
        if len(self.observationHistory) <= 1:
            # Initialize beliefs on first call
            self._initialize_beliefs_if_needed(gameState)

        # Update beliefs for all opponents
        for opponent_idx in self.getOpponents(gameState):
            self._update_opponent_belief(gameState, opponent_idx)

        # Mark this timestep as updated for this team (after all opponents processed)
        current_timestep = len(self.observationHistory) - 1
        team_key = self._get_team_key()
        SmurphAgent._last_belief_update_timestep[team_key] = current_timestep

        # --- 2. Get the current state graph ---
        node_features = self._get_graph_state(gameState)
        my_pos = gameState.getAgentPosition(self.index)
        assert my_pos in self.node_map

        # Get the node index of my agent
        my_idx = self.node_map[my_pos]

        x = torch.tensor(node_features, dtype=torch.float, device=self.device)
        edge_index = torch.tensor(
            self.graph_edge_index, dtype=torch.long, device=self.device
        ).T.contiguous()

        current_state_graph = Data(x=x, edge_index=edge_index, node_idx=my_idx)

        # --- 3. Calculate the reward for the previous state ---
        if self.mode == "training":
            previous_state = self.getPreviousObservation()
            if previous_state:
                assert (
                    self.previous_state_graph is not None
                    and self.previous_action_idx is not None
                )
                reward = self._calculate_reward(previous_state, gameState)
                self.episode_memory.append(
                    Transition(
                        self.previous_state_graph,
                        self.previous_action_idx,
                        reward,
                        current_state_graph,
                        False,
                    )
                )
            self._upload_experiences()

        # --- 4. Choose an action ---
        legal_actions = gameState.getLegalActions(self.index)

        epsilon = self.config.epsilon_min + (
            self.config.epsilon_start - self.config.epsilon_min
        ) * math.exp(-1.0 * self.config.games_played / self.config.epsilon_decay_rate)
        if random.random() < epsilon:
            action = random.choice(legal_actions)
        else:
            self.policy_net.eval()
            with torch.no_grad():
                q_vals = self.policy_net(current_state_graph)

            action = None
            max_q = -float("inf")

            for legal_action in legal_actions:
                q_val = q_vals[self.DIR_TO_IDX[legal_action]].item()
                if q_val > max_q:
                    max_q = q_val
                    action = legal_action

        assert action is not None

        # --- 5. Save state for next action ---
        if self.mode == "training":
            self.previous_action_idx = self.DIR_TO_IDX[action]
            self.previous_state_graph = current_state_graph

        return action

    def final(self, gameState: GameState):
        """This is called at the end of a game to trigger training."""
        if self.mode == "training":
            previous_state = self.getPreviousObservation()
            if previous_state:
                reward = self._calculate_reward(previous_state, gameState)
                self.episode_memory.append(
                    Transition(
                        self.previous_state_graph,
                        self.previous_action_idx,
                        reward,
                        None,
                        True,
                    )
                )

            self._upload_experiences(force=True)

    def _calculate_reward(
        self, prev_gameState: GameState, current_gameState: GameState
    ):
        """
        Event-based reward with robust disambiguation and partial-observability safeguards.
        """
        reward = 0.0
        w = self.config.reward_weights

        # --- Shortcuts
        my_prev = prev_gameState.getAgentState(self.index)
        my_curr = current_gameState.getAgentState(self.index)
        my_prev_pos = my_prev.getPosition()
        my_curr_pos = my_curr.getPosition()
        teammate_prev = prev_gameState.getAgentState(self._get_teammate_idx())
        teammate_curr = current_gameState.getAgentState(self._get_teammate_idx())

        def manh(a, b):
            return None if (a is None or b is None) else manhattanDistance(a, b)

        def near(a, b, tol):
            d = manh(a, b)
            return (d is not None) and (d <= tol)

        # ----- 1) Global signals

        # Score delta from *our team's* perspective (Red-Blue from env)
        score_delta = current_gameState.getScore() - prev_gameState.getScore()
        if not self.red:  # flip for blue
            score_delta *= -1
        reward += w.score_change * score_delta

        # Opponent ate our food (defending side food decreased)
        # Prefer robust counting via asList() if available
        prev_def_cnt = len(self.getFoodYouAreDefending(prev_gameState).asList())
        curr_def_cnt = len(self.getFoodYouAreDefending(current_gameState).asList())
        food_lost = prev_def_cnt - curr_def_cnt
        if food_lost > 0:
            reward += w.food_eaten_by_opponent * food_lost

        # Opponent ate our capsule (defending side capsule decreased)
        prev_def_cnt = len(self.getCapsulesYouAreDefending(prev_gameState))
        curr_def_cnt = len(self.getCapsulesYouAreDefending(current_gameState))
        capsule_lost = prev_def_cnt - curr_def_cnt
        if capsule_lost > 0:
            reward += w.capsule_eaten_by_opponent * capsule_lost

        # Time step penalty
        reward += w.time_penalty

        # ----- Helper: opponent iteration
        opp_indices = self.getOpponents(current_gameState)

        # ----- 2) When we were an invader at t-1 (Pacman last turn)
        if my_prev.isPacman:
            # Ate food: carrying increased
            if my_curr.numCarrying > my_prev.numCarrying:
                reward += w.ate_food

            # Ate a capsule: we stepped onto a capsule that disappeared
            prev_caps = set(self.getCapsules(prev_gameState))
            curr_caps = set(self.getCapsules(current_gameState))
            eaten_capsules = prev_caps - curr_caps
            if my_curr_pos in eaten_capsules:
                reward += w.ate_capsule

            # Ate scared ghost: credit only if (i) opp was scared, (ii) now unscared & respawned,
            # and (iii) we were colliding with their previous position.
            for oi in opp_indices:
                opp_prev = prev_gameState.getAgentState(oi)
                opp_curr = current_gameState.getAgentState(oi)
                opp_prev_pos = opp_prev.getPosition()
                opp_curr_pos = opp_curr.getPosition()

                # must have been scared at t-1, and now not scared
                if opp_prev.scaredTimer > 0 and opp_curr.scaredTimer == 0:
                    # respawned now (at start). If out of FOV (None), still ok if we had collision evidence.
                    respawned = (
                        opp_curr_pos is not None and opp_curr_pos == opp_curr.start.pos
                    ) or (opp_curr_pos is None)

                    # collision evidence: we moved onto them OR were already on them
                    collided = near(
                        my_curr_pos, opp_prev_pos, COLLISION_TOLERANCE
                    ) or near(my_prev_pos, opp_prev_pos, COLLISION_TOLERANCE)

                    if respawned and collided:
                        reward += w.ate_scared_ghost

            # Transitioned out of Pacman at t (we are home or we died)
            if not my_curr.isPacman:
                # If we are at start now -> we died as invader
                if my_curr_pos == my_curr.start.pos:
                    lost_points = my_prev.numCarrying
                    reward += w.lost_points * lost_points
                else:
                    # We crossed home and banked points (carrying dropped to 0 but no teleport)
                    if my_prev.numCarrying > 0 and my_curr.numCarrying == 0:
                        reward += w.scored_points * my_prev.numCarrying

        # ----- 3) When we are a ghost at t (defending last turn or now)
        else:
            for oi in opp_indices:
                opp_prev = prev_gameState.getAgentState(oi)
                opp_curr = current_gameState.getAgentState(oi)
                opp_prev_pos = opp_prev.getPosition()
                opp_curr_pos = opp_curr.getPosition()

                # Ate an invader: opp was Pacman, now not Pacman and respawned,
                # and we had collision evidence.
                if opp_prev.isPacman and not opp_curr.isPacman:
                    respawned = (
                        opp_curr_pos is not None and opp_curr_pos == opp_curr.start.pos
                    ) or (opp_curr_pos is None)
                    collided = near(
                        my_curr_pos, opp_prev_pos, COLLISION_TOLERANCE
                    ) or near(my_prev_pos, opp_prev_pos, COLLISION_TOLERANCE)
                    if respawned and collided:
                        saved_points = opp_prev.numCarrying
                        reward += w.saved_points * saved_points

            # Eaten as a scared ghost: we were scared at t-1, now unscared at t (reset),
            # and teleported to start.
            if (
                my_prev.scaredTimer > 0
                and my_curr.scaredTimer == 0
                and my_curr_pos == my_curr.start.pos
            ):
                reward += w.eaten_as_scared_ghost

        # --- 4. Teammate reward ---
        if teammate_prev.isPacman:
            # Ate food: carrying increased
            if teammate_curr.numCarrying > teammate_prev.numCarrying:
                reward += w.teammate_ate_food

            # Teammate scored points
            if (
                not teammate_curr.isPacman
                and teammate_curr.numCarrying == 0
                and teammate_prev.numCarrying > 0
                and teammate_prev.getPosition() != teammate_curr.start.pos
            ):
                reward += w.teammate_scored_points * teammate_prev.numCarrying

        return reward

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

    def _get_config_path(self):
        return f"configs/{self.id}.pt"

    def _get_weights_path(self):
        return f"weights/{self.id}.pt"

    def _upload_experiences(self, force=False):
        if len(self.episode_memory) < 32 and not force:
            return

        file_path = f"experiences/{self.id}/{uuid.uuid4()}.pt"
        print(
            f"[Agent {self.id}] Uploading {len(self.episode_memory)} experiences to {file_path}"
        )

        # Convert Transitions to plain tuples to avoid pickle issues with dynamic module names
        experiences_as_tuples = [
            (t.state, t.action, t.reward, t.next_state, t.done)
            for t in self.episode_memory
        ]
        torch.save(experiences_as_tuples, file_path)
        self.episode_memory = []

        # Try to update weights
        try:
            self.policy_net.load_state_dict(torch.load(self._get_weights_path()))
            print(f"[Agent {self.id}] Reloaded updated weights")
        except Exception as e:
            print(f"[Agent {self.id}] Failed to reload weights: {e}")
