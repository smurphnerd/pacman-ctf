import torch
import torch.nn as nn
from capture import GameState
from enum import IntEnum

from captureAgents import CaptureAgent
from myTeam import SmurphAgent


class MCTSNode:
    def __init__(self, parent=None, prior_p=0.0):
        self.parent = parent
        self.children = {}  # A map from action to MCTSNode
        self.visit_count = 0  # N
        self.total_action_value = 0.0  # W
        self.mean_action_value = 0.0  # Q
        self.prior_probability = prior_p  # P


class MCTS:
    def __init__(
        self, agent: SmurphAgent, agent_index: int, top_k: int = 1, batch_size: int = 32
    ):
        self.agent = agent
        self.network = agent.network
        self.index = agent_index
        self.top_k = top_k
        self.batch_size = batch_size

    def run(self, root_state: GameState) -> dict:
        # 1. Creates a temporary root node.
        # 2. Runs the simulation loop (Selection, Expansion, Evaluation, Backpropagation) for num_simulations.
        # 3. Returns the final improved policy (a dictionary of {action: visit_count_probability}).
        pass


# This could be a class that writes to files in a shared directory.
class ReplayBuffer:
    def add_game_experience(self, game_history: list):
        # game_history is a list of (state_tensor, mcts_policy, final_winner_z) tuples.
        # Writes this data to a file with a unique ID.
        pass

    def sample_batch(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor]:
        # Randomly loads data files, samples from them, and returns a training batch.
        pass
