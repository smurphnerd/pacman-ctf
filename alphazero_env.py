"""
AlphaZero-compatible environment for Pacman Capture the Flag.

This implements a 4-player sequential turn-based game where:
- All 4 agents use the same neural network policy
- State is transformed to each agent's perspective (board flip for Blue team)
- Only terminal rewards (+1 win, -1 loss)
- MCTS controls all agents during tree search
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, Tuple, Dict, Any
import copy

from capture import GameState, SIGHT_RANGE
from game import Directions, Actions
import layout
from util import manhattanDistance
import belief_tracking

try:
    profile
except NameError:

    def profile(func):
        return func


class CellType:
    """Cell type enumeration."""

    WALL = 0
    EMPTY_ALLY_TERRITORY = 1
    ALLY_FOOD = 2
    ALLY_CAPSULE = 3
    EMPTY_ENEMY_TERRITORY = 4
    ENEMY_FOOD = 5
    ENEMY_CAPSULE = 6


class AlphaZeroPacmanEnv(gym.Env):
    """
    AlphaZero environment for Pacman Capture the Flag.

    Features:
    - Sequential turn-based: agents move in order 0→1→2→3→0...
    - Perspective transformation: Blue agents see flipped board
    - Terminal rewards only: +1 for win, -1 for loss
    - All agents use same policy (symmetrical self-play)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        layout_name: str = "mediumCapture",
        render_mode: Optional[str] = None,
        time_limit: int = 1200,
    ):
        """
        Args:
            layout_name: Name of the layout file (without .lay extension)
            render_mode: "human" for GUI, None for headless
            time_limit: Maximum steps per episode (in total agent moves, not rounds)
                       Default 1200 = 300 moves per agent
        """
        super().__init__()

        self.layout_name = layout_name
        self.render_mode = render_mode
        self.time_limit = time_limit

        # Load layout to get dimensions
        layout_obj = layout.getLayout(layout_name)
        assert isinstance(layout_obj, layout.Layout)
        self.width = layout_obj.width
        self.height = layout_obj.height

        # Action space: 5 actions per agent
        self.action_space = spaces.Discrete(5)

        # Observation space: Dictionary with spatial and scalar features
        # Spatial: 11-channel image
        #   Channels 0-6: One-hot cell type
        #   Channel 7: Opponent 1 position/belief
        #   Channel 8: Opponent 2 position/belief
        #   Channel 9: Teammate position
        #   Channel 10: My position
        # Scalar: 14 non-spatial features
        #   0-3: Scared timers (me, teammate, opp1, opp2)
        #   4-7: Food carrying (me, teammate, opp1, opp2)
        #   8-11: Pacman status (me, teammate, opp1, opp2)
        #   12: Time remaining
        #   13: Score
        self.observation_space = spaces.Dict(
            {
                "spatial": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(11, self.width, self.height),
                    dtype=np.float32,
                ),
                "scalar": spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32),
            }
        )

        # Action mapping
        self.IDX_TO_DIR = {
            0: Directions.NORTH,
            1: Directions.SOUTH,
            2: Directions.EAST,
            3: Directions.WEST,
            4: Directions.STOP,
        }
        self.DIR_TO_IDX = {v: k for k, v in self.IDX_TO_DIR.items()}

        # Game state
        self.current_agent: int = 0  # Whose turn it is (0, 1, 2, or 3)
        self.steps: int = 0  # Total number of agent moves taken

        # Shared belief distributions: beliefs[agent_idx] = 2D probability array
        # Teammates share beliefs about opponents (Red team shares, Blue team shares)
        self.beliefs: Dict[int, list] = {}

        # Initialize display if rendering
        self.display = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Load layout and initialize game state
        layout_obj = layout.getLayout(self.layout_name)
        self.game_state = GameState()
        self.game_state.initialize(layout_obj, 4)
        self.game_state.data.timeleft = self.time_limit

        # Randomly choose starting agent (Red 0 or Blue 1)
        self.current_agent = 0
        self.steps = 0

        # Calculate layout-specific bounds for normalization
        self.red_food = len(self.game_state.getRedFood().asList())
        self.blue_food = len(self.game_state.getBlueFood().asList())
        self.max_score = max(self.red_food, self.blue_food)

        # Initialize belief distributions to exact starting positions
        self.beliefs = belief_tracking.initialize_beliefs(self.game_state)

        # Get initial observation for current agent
        observation = self._get_observation()
        info = self._get_info()

        # Initialize display if rendering
        if self.render_mode == "human":
            import captureGraphicsDisplay

            self.display = captureGraphicsDisplay.PacmanGraphics(
                "Red", "Blue", zoom=1.0, frameTime=0.1
            )
            self.display.initialize(self.game_state.data)

        return observation, info

    @profile
    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one agent's move.

        Args:
            action: Action index (0-4) for the current agent

        Returns:
            observation: Dict with "spatial" and "scalar" keys, from NEXT agent's perspective
            reward: 0 during play, +1/-1 at termination (from next agent's perspective)
            terminated: Whether episode ended (win/loss)
            truncated: Whether episode was cut off (time limit)
            info: Additional information
        """
        # Get legal actions for current agent
        legal_actions = self.game_state.getLegalActions(self.current_agent)
        chosen_direction = self.IDX_TO_DIR[action]

        # If action is illegal, fall back to STOP (or could raise error)
        assert chosen_direction in legal_actions, "Illegal action"

        # Apply action
        self.game_state = self.game_state.generateSuccessor(
            self.current_agent, chosen_direction
        )

        # Increment step counter (counts total agent moves)
        self.steps += 1

        # Move to next agent
        self.current_agent = (self.current_agent + 1) % 4

        # Update beliefs from current agent's perspective
        self.beliefs = belief_tracking.update_all_beliefs(
            self.beliefs,
            self.game_state.makeObservation(self.current_agent),
            self.current_agent,
        )

        # Check termination
        truncated = self.steps >= self.time_limit
        if truncated:
            self.game_state.data._win = True
        terminated = self.game_state.isOver()

        # Calculate reward (0 during play, +1/-1 at end from NEXT agent's perspective)
        reward = 0.0
        if terminated or truncated:
            reward = self._get_terminal_reward()
            if reward == 0:
                # Tie game
                reward = 1e-8

        # Render if needed
        if self.render_mode == "human":
            self.render()

        # Get observation from NEXT agent's perspective
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    @profile
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get observation from current agent's perspective.

        For Blue agents (1, 3), the board is flipped horizontally so they
        see the game as if they were Red (attacking from left to right).

        Returns:
            observation: Dictionary with keys:
                - "spatial": [11, width, height] array
                - "scalar": [14] array
        """
        assert self.game_state is not None

        width, height = self.width, self.height
        state_tensor = np.zeros((11, width, height), dtype=np.float32)

        is_red = self.current_agent in [0, 2]

        # Cache food/capsules/walls once before the loops (MAJOR OPTIMIZATION)
        walls = self.game_state.getWalls()
        red_food = self.game_state.getRedFood()
        blue_food = self.game_state.getBlueFood()
        red_capsules = self.game_state.getRedCapsules()
        blue_capsules = self.game_state.getBlueCapsules()

        # Fill in cell types (channels 0-6)
        for x in range(width):
            for y in range(height):
                # Get actual position (flip if Blue)
                pos = self.transform_pos((x, y), is_red)

                cell_type = self._get_cell_type(
                    pos, is_red, walls, red_food, blue_food, red_capsules, blue_capsules
                )
                state_tensor[cell_type, x, y] = 1.0

        # Agent positions (channels 7-10)
        # We need to place agents in the perspective-transformed coordinates
        my_team = [self.current_agent, (self.current_agent + 2) % 4]
        opponent_team = [idx for idx in range(4) if idx not in my_team]

        # Place teammate and self (always exact positions)
        for idx in my_team:
            pos = self.game_state.getAgentPosition(idx)
            if pos is not None:
                # Transform position if Blue
                pos = self.transform_pos(pos, is_red)
                x, y = int(pos[0]), int(pos[1])

                # Place in appropriate channel
                if idx == self.current_agent:
                    state_tensor[10, x, y] = 1.0  # My position
                else:
                    state_tensor[9, x, y] = 1.0  # Teammate

        # Place opponents (use beliefs for distant, exact for visible)
        for i, opp_idx in enumerate(opponent_team):
            opp_channel = 7 + i  # Channel 7 for first opponent, 8 for second

            # Check if opponent is visible
            if belief_tracking.is_opponent_visible(
                self.game_state, self.current_agent, opp_idx
            ):
                # Visible - use exact position
                pos = self.game_state.getAgentPosition(opp_idx)
                if pos is not None:
                    pos = self.transform_pos(pos, is_red)
                    x, y = int(pos[0]), int(pos[1])
                    state_tensor[opp_channel, x, y] = 1.0
            else:
                # Not visible - use belief distribution
                belief = self.beliefs[opp_idx]
                for x in range(width):
                    for y in range(height):
                        actual_x = self.transform_pos((x, y), is_red)[0]
                        state_tensor[opp_channel, x, y] = belief[actual_x][y]

        # ============ Scalar Features ============

        # Get agent states
        my_idx = self.current_agent
        teammate_idx = (self.current_agent + 2) % 4
        opp1_idx, opp2_idx = opponent_team[0], opponent_team[1]

        my_state = self.game_state.getAgentState(my_idx)
        teammate_state = self.game_state.getAgentState(teammate_idx)
        opp1_state = self.game_state.getAgentState(opp1_idx)
        opp2_state = self.game_state.getAgentState(opp2_idx)

        is_red = True if my_idx in [0, 2] else False
        max_food = self.blue_food if is_red else self.red_food
        max_food_defending = self.red_food if is_red else self.blue_food

        # Build scalar features (normalized to [0, 1] or [-1, 1])
        scalar_features = np.array(
            [
                # Scared timers (0-40) → normalize to [0, 1]
                my_state.scaredTimer / 40.0,
                teammate_state.scaredTimer / 40.0,
                opp1_state.scaredTimer / 40.0,
                opp2_state.scaredTimer / 40.0,
                # Food carrying (0 to max_food) → normalize to [0, 1]
                my_state.numCarrying / float(max_food),
                teammate_state.numCarrying / float(max_food),
                opp1_state.numCarrying / float(max_food_defending),
                opp2_state.numCarrying / float(max_food_defending),
                # Pacman status (0 or 1)
                float(my_state.isPacman),
                float(teammate_state.isPacman),
                float(opp1_state.isPacman),
                float(opp2_state.isPacman),
                # Time remaining → normalize to [0, 1]
                self.game_state.data.timeleft / self.time_limit,
                # Score (from my perspective) → normalize to [-1, 1]
                self.game_state.getScore()
                / float(self.max_score)
                * (1.0 if is_red else -1.0),
            ],
            dtype=np.float32,
        )

        return {"spatial": state_tensor, "scalar": scalar_features}

    @profile
    def _get_cell_type(
        self,
        pos: Tuple[int, int],
        is_red: bool,
        walls,
        red_food,
        blue_food,
        red_capsules,
        blue_capsules,
    ) -> int:
        """
        Get cell type at position (x, y).

        Args:
            pos: Board coordinates (x, y)
            is_red: If True, use Red team perspective
            walls: Cached walls grid
            red_food: Cached red food grid
            blue_food: Cached blue food grid
            red_capsules: Cached red capsules list
            blue_capsules: Cached blue capsules list

        Returns:
            CellType value
        """
        x, y = pos

        if x < 0 or y < 0 or x >= walls.width or y >= walls.height or walls[x][y]:
            return CellType.WALL

        # From current perspective, determine ally vs enemy
        if is_red:
            ally_food = red_food
            enemy_food = blue_food
            ally_capsules = red_capsules
            enemy_capsules = blue_capsules
        else:
            ally_food = blue_food
            enemy_food = red_food
            ally_capsules = blue_capsules
            enemy_capsules = red_capsules

        # Check capsules
        if pos in ally_capsules:
            return CellType.ALLY_CAPSULE
        elif pos in enemy_capsules:
            return CellType.ENEMY_CAPSULE

        # Check food
        elif ally_food[x][y]:
            return CellType.ALLY_FOOD
        elif enemy_food[x][y]:
            return CellType.ENEMY_FOOD

        # Empty space - determine territory
        else:
            midpoint = walls.width // 2

            if is_red:
                # Red team: left half is ally, right half is enemy
                if x < midpoint:
                    return CellType.EMPTY_ALLY_TERRITORY
                else:
                    return CellType.EMPTY_ENEMY_TERRITORY
            else:
                # Blue team: right half is ally, left half is enemy
                if x >= midpoint:
                    return CellType.EMPTY_ALLY_TERRITORY
                else:
                    return CellType.EMPTY_ENEMY_TERRITORY

    def _get_terminal_reward(self) -> float:
        """
        Get terminal reward from current agent's team perspective.

        Returns:
            +1.0 if current team won
            -1.0 if current team lost
        """
        if not self.game_state.isOver():
            return 0.0

        score = self.game_state.getScore()  # Positive = Red winning
        sign = 1.0 if score > 0 else -1.0 if score < 0 else 0.0
        current_team_is_red = self.current_agent in [0, 2]

        return sign * (1.0 if current_team_is_red else -1.0)

    def _get_info(self) -> Dict[str, Any]:
        """Return additional information."""
        return {
            "current_agent": self.current_agent,
            "steps": self.steps,
            "score": self.game_state.getScore() if self.game_state else 0,
        }

    def get_legal_actions_mask(self) -> np.ndarray:
        """
        Get binary mask of legal actions for current agent.

        Returns:
            mask: [5] binary array where 1 = legal, 0 = illegal
        """
        if self.game_state is None:
            return np.ones(5, dtype=np.float32)

        legal_actions = self.game_state.getLegalActions(self.current_agent)
        mask = np.zeros(5, dtype=np.float32)

        for action in legal_actions:
            action_idx = self.DIR_TO_IDX[action]
            mask[action_idx] = 1.0

        return mask

    def transform_pos(self, pos, is_red: bool) -> Tuple[int, int]:
        if is_red:
            return pos

        return (self.width - 1 - pos[0], pos[1])

    @profile
    def clone(self) -> "AlphaZeroPacmanEnv":
        """
        Create a deep copy of the environment for MCTS simulation.

        Returns:
            Cloned environment
        """
        cloned = AlphaZeroPacmanEnv(
            layout_name=self.layout_name,
            render_mode=None,  # Don't clone display
            time_limit=self.time_limit,
        )

        cloned.width = self.width
        cloned.height = self.height
        cloned.game_state = self.game_state.deepCopy() if self.game_state else None
        cloned.current_agent = self.current_agent
        cloned.steps = self.steps

        # Copy normalization bounds
        if hasattr(self, "red_food"):
            cloned.red_food = self.red_food
        if hasattr(self, "blue_food"):
            cloned.blue_food = self.blue_food
        if hasattr(self, "max_score"):
            cloned.max_score = self.max_score

        # Deep copy belief distributions
        if hasattr(self, "beliefs"):
            cloned.beliefs = {}
            for agent_idx, belief in self.beliefs.items():
                # Deep copy the 2D list
                cloned.beliefs[agent_idx] = [row[:] for row in belief]

        return cloned

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.display and self.game_state:
                self.display.update(self.game_state.data)

    def close(self):
        """Clean up resources."""
        if self.display:
            self.display.finish()
            self.display = None
