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

from ast import Raise
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import IntEnum

from numpy import true_divide
import torch
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os
from capture import GameState, noisyDistance
from game import Directions, Actions, AgentState, Agent, Grid
from util import nearestPoint
import sys, os


HIDDEN_LAYERS = [300, 200, 100]
RECEPTIVE_RANGE = 9


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
    weights: torch.Tensor
    alpha: float  # learning rate
    discount_rate: float
    epsilon: float  # exploration prob


class SmurphAgent(CaptureAgent):
    # Class variable to store pre-generated relative offsets (shared across all agents)
    _relative_offsets: List[Tuple[int, int]] = []

    # Shared belief distributions for opponent positions (per team)
    # Dict mapping (team_color, opponent_idx) -> 2D probability array
    # team_color is "red" or "blue"
    _opponent_beliefs: Dict[Tuple[str, int], List[List[float]]] = {}

    # Track the last timestep when beliefs were updated per team (to avoid double-updating per timestep)
    # Dict mapping team_color -> last_timestep
    _last_belief_update_timestep: Dict[str, int] = {}

    def registerInitialState(self, gameState: GameState):
        CaptureAgent.registerInitialState(self, gameState)

        # Generate relative offsets once for all agents
        self.receptive_state_size = self.get_expected_state_size(RECEPTIVE_RANGE)
        if len(SmurphAgent._relative_offsets) != self.receptive_state_size:
            SmurphAgent._relative_offsets = self._generate_relative_offsets(
                RECEPTIVE_RANGE
            )
        assert len(SmurphAgent._relative_offsets) == self.receptive_state_size

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
        SmurphAgent._last_belief_update_timestep[team_key] = current_timestep

        # Now you can access the belief distributions
        # Example: Get belief for first opponent
        opponents = self.getOpponents(gameState)
        opp_belief = self._get_opponent_pos_dist(opponents[0])

        # Optional: Visualize beliefs for debugging
        # belief_counters = []
        # for i in range(gameState.getNumAgents()):
        #     if i in opponents:
        #         belief = self._get_opponent_pos_dist(i)
        #         counter = util.Counter()
        #         for x in range(len(belief)):
        #             for y in range(len(belief[x])):
        #                 if belief[x][y] > 0:
        #                     counter[(x, y)] = belief[x][y]
        #         belief_counters.append(counter)
        #     else:
        #         belief_counters.append(None)
        # self.displayDistributionsOverPositions(belief_counters)

        # TODO: Implement your actual action selection logic here
        time.sleep(1)
        actions = gameState.getLegalActions(self.index)
        from capture import CaptureRules

        return random.choice(actions)

    def __init__(self, index, config_path):
        super().__init__(index)

        # config = torch.load(config_path)
        # assert isinstance(config, SmurphAgentConfig)
        # self.config = config

    def _get_processed_state(self, gameState: GameState):
        receptive_field_state = self._get_receptive_field_state(gameState)
        # TODO: Add self and teammates agent states
        # TODO: Add opponents agent states with best guess of positions from previous turn

    def _get_receptive_field_state(self, gameState: GameState):
        agent_pos = gameState.getAgentPosition(self.index)

        # Calculate expected array size and initialize with walls (0)
        receptive_state_array = [CellType.WALL] * self.receptive_state_size

        # Use pre-generated relative offsets for consistent ordering
        for array_index, (dx, dy) in enumerate(SmurphAgent._relative_offsets):
            # Calculate absolute position
            x, y = agent_pos[0] + dx, agent_pos[1] + dy

            receptive_state_array[array_index] = self._get_cell_type(
                x,
                y,
                gameState,
            )

        # Normalize to [0, 1] range
        return (
            torch.tensor(receptive_state_array, dtype=torch.float32)
            / max(CellType).value
        )

    @staticmethod
    def get_expected_state_size(receptive_range):
        return 2 * receptive_range * (receptive_range + 1)

    def _generate_relative_offsets(self, receptive_range):
        offsets = []

        for manhattan_dist in range(
            1, receptive_range + 1
        ):  # Start from 1, exclude center
            dist_offsets = []

            # Generate all (dx, dy) pairs where |dx| + |dy| = manhattan_dist
            for dx in range(-manhattan_dist, manhattan_dist + 1):
                remaining_dist = manhattan_dist - abs(dx)

                if remaining_dist == 0:
                    # Only one position at this dx
                    dist_offsets.append((dx, 0))
                else:
                    # Two positions: one above, one below
                    dist_offsets.append((dx, -remaining_dist))
                    dist_offsets.append((dx, remaining_dist))

            # Sort positions at this distance for consistent ordering
            # Order by dx first, then by dy
            dist_offsets.sort(key=lambda pos: (pos[0], pos[1]))
            offsets.extend(dist_offsets)

        return offsets

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
            prob_array = [[0.0 for y in range(height)] for x in range(width)]

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
            prob_array = [[0.0 for y in range(height)] for x in range(width)]
            x, y = int(opp_pos[0]), int(opp_pos[1])
            prob_array[x][y] = 1.0
            SmurphAgent._opponent_beliefs[(team_key, opponent_idx)] = prob_array
            return

        # PARTIAL OBSERVATION - update using noisy distance
        old_beliefs = SmurphAgent._opponent_beliefs[(team_key, opponent_idx)]

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
            SmurphAgent._opponent_beliefs[(team_key, opponent_idx)] = new_beliefs
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
