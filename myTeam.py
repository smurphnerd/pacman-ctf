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
from typing import List, Tuple
from dataclasses import dataclass
from enum import IntEnum

from numpy import true_divide
import torch
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os
from capture import GameState, noisyDistance
from game import Directions, Actions, AgentState, Agent
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
        pass

    def __init__(self, index, config_path):
        super().__init__(index)
        self._load_config(config_path)

    def _load_config(self, config_path):
        config = torch.load(config_path)
        assert isinstance(config, SmurphAgentConfig)
        self.config = config

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
