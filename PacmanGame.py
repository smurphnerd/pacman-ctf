"""
PacmanGame.py - Adapter between AlphaZero Pacman Environment and alpha-zero-general's Game API.

This bridges our Gymnasium environment (alphazero_env.py) with the alpha-zero-general framework.

Key Design:
- 4 agents (0, 1, 2, 3) form 2 teams:
  - Red Team: agents 0, 2 → player  1
  - Blue Team: agents 1, 3 → player -1
- Board is a dict containing: {spatial, scalar, current_agent, env_state}
- Environment already handles perspective flipping
- Prevents oscillation by blocking opposite moves (N↔S, E↔W)
"""

import numpy as np
from AZGame import Game
from alphazero_env import AlphaZeroPacmanEnv

# Action indices
ACTION_NORTH = 0
ACTION_SOUTH = 1
ACTION_EAST = 2
ACTION_WEST = 3
ACTION_STOP = 4

# Opposite action mapping (to prevent oscillation)
OPPOSITE_ACTIONS = {
    ACTION_NORTH: ACTION_SOUTH,
    ACTION_SOUTH: ACTION_NORTH,
    ACTION_EAST: ACTION_WEST,
    ACTION_WEST: ACTION_EAST,
    ACTION_STOP: ACTION_STOP,  # Prevent consecutive STOPs
}

try:
    profile
except NameError:
    def profile(func):
        return func

class PacmanGame(Game):
    """
    Adapter for 4-player Pacman CTF compatible with alpha-zero-general.
    """

    def __init__(self, layout_name="mediumCapture", time_limit=1200):
        self.layout_name = layout_name
        self.time_limit = time_limit

        # Create template environment to get dimensions (no rendering)
        self.env = AlphaZeroPacmanEnv(
            layout_name=layout_name, time_limit=time_limit, render_mode=None
        )
        obs, _ = self.env.reset()

        self.width = self.env.width
        self.height = self.env.height
        self.action_size = 5  # N, S, E, W, STOP

        # Store spatial and scalar shapes
        self.spatial_shape = obs["spatial"].shape  # (11, width, height)
        self.scalar_shape = obs["scalar"].shape  # (14,)

    @profile
    def getInitBoard(self):
        """
        Returns:
            Initial board state as dict containing observation + metadata
        """
        obs, info = self.env.reset()

        # Create board representation
        board = {
            "spatial": obs["spatial"].copy(),
            "scalar": obs["scalar"].copy(),
            "current_agent": info["current_agent"],
            "env": self.env,  # Keep reference to environment
            "reward": 0,
            "last_actions": {
                0: None,
                1: None,
                2: None,
                3: None,
            },  # Track last action per agent
        }

        return board

    def getBoardSize(self):
        """
        Returns:
            (width, height) of the board
        """
        return (self.width, self.height)

    def getActionSize(self):
        """
        Returns:
            Total number of possible actions (5: N, S, E, W, STOP)
        """
        return self.action_size

    @profile
    def getNextState(self, board, player, action):
        """
        Execute action and return next state.

        IMPORTANT: This clones the environment to avoid corrupting MCTS tree search.

        Args:
            board: Current board dict
            player: current player (1 or -1)
            action: Action index (0-4)

        Returns:
            nextBoard: Board after action (contains current_agent for next player)
        """
        # Clone environment to avoid modifying original during MCTS simulation
        env = board["env"].clone()
        current_agent = board["current_agent"]

        # Execute action
        obs, reward, _, _, info = env.step(action)

        next_agent = info["current_agent"]

        # Update last actions - copy from previous board and update current agent
        last_actions = board.get(
            "last_actions", {0: None, 1: None, 2: None, 3: None}
        ).copy()
        last_actions[current_agent] = action

        # Create next board
        next_board = {
            "spatial": obs["spatial"].copy(),
            "scalar": obs["scalar"].copy(),
            "current_agent": next_agent,  # Board tracks whose turn it is
            "env": env,
            "reward": reward,
            "last_actions": last_actions,
        }

        return next_board, -player

    def getValidMoves(self, board, player):
        """
        Returns:
            Binary mask of legal actions [5], with opposite of last action filtered out
        """
        env = board["env"]
        current_agent = board["current_agent"]

        # Get base legal actions from environment
        legal_mask = env.get_legal_actions_mask().copy()

        # Filter out opposite action to prevent oscillation
        last_action = board.get("last_actions", {}).get(current_agent)
        if last_action is not None:
            opposite_action = OPPOSITE_ACTIONS.get(last_action)
            if opposite_action is not None and legal_mask[opposite_action] == 1:
                legal_mask[opposite_action] = 0

        # Ensure at least one action is legal (shouldn't happen, but safety check)
        if np.sum(legal_mask) == 0:
            legal_mask = env.get_legal_actions_mask()  # Fallback to original

        return legal_mask.astype(np.int8)

    def getGameEnded(self, board, player):
        """
        Check if game has ended.

        Args:
            board: Current board
            player: Current player (1 or -1)

        Returns:
            0 if ongoing
            +1 if player won
            -1 if player lost
            small non-zero for draw
        """
        return board.get("reward")

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        # Use spatial tensor and current agent for hash
        spatial_str = board["spatial"].tobytes().hex()
        scalar_str = board["scalar"].tobytes().hex()
        agent_str = str(board["current_agent"])

        return f"{spatial_str}_{scalar_str}_{agent_str}"

    def clone_board(self, board):
        env = board["env"]
        cloned_env = env.clone()

        cloned_board = {
            "spatial": board["spatial"].copy(),
            "scalar": board["scalar"].copy(),
            "current_agent": board["current_agent"],
            "env": cloned_env,
        }

        if "reward" in board:
            cloned_board["reward"] = board["reward"]

        # Copy last actions tracking
        if "last_actions" in board:
            cloned_board["last_actions"] = board["last_actions"].copy()

        return cloned_board
