"""
belief_tracking.py - Shared belief tracking utilities for Pacman CTF.

These functions are used by both:
- SmurphAgent (myTeam.py) during inference
- AlphaZeroPacmanEnv (alphazero_env.py) during training

This ensures consistent opponent modeling between training and deployment.
"""

from typing import List, Dict
from capture import GameState, SIGHT_RANGE
from game import Actions
from util import manhattanDistance

try:
    profile
except NameError:
    def profile(func):
        return func

@profile
def initialize_beliefs(game_state: GameState) -> Dict[int, List[List[float]]]:
    """
    Initialize belief distributions for all 4 agents to their exact starting positions.

    Args:
        game_state: Initial game state

    Returns:
        beliefs: Dict mapping agent_idx -> 2D probability array
                 Each agent gets a belief distribution initialized to their start position
    """
    walls = game_state.getWalls()
    width, height = walls.width, walls.height

    beliefs = {}

    for agent_idx in range(4):
        # Create empty probability array
        prob_array = [[0.0 for _ in range(height)] for _ in range(width)]

        # Get agent's exact starting position
        agent_pos = game_state.getAgentState(agent_idx).start.pos
        x, y = int(agent_pos[0]), int(agent_pos[1])
        prob_array[x][y] = 1.0  # Certain they're at starting position

        beliefs[agent_idx] = prob_array

    return beliefs


def is_opponent_visible(
    game_state: GameState, observer_idx: int, opponent_idx: int
) -> bool:
    """
    Check if an opponent is visible to any teammate of the observer.

    An opponent is visible if within SIGHT_RANGE (5 Manhattan distance) of any teammate.

    Args:
        game_state: Current game state
        observer_idx: Index of observing agent
        opponent_idx: Index of opponent to check

    Returns:
        True if opponent is visible to observer's team
    """
    opponent_pos = game_state.getAgentPosition(opponent_idx)

    if opponent_pos is None:
        return False

    # Get observer's team
    if observer_idx in [0, 2]:
        team = [0, 2]  # Red team
    else:
        team = [1, 3]  # Blue team

    # Check if any teammate can see the opponent
    for teammate_idx in team:
        teammate_pos = game_state.getAgentPosition(teammate_idx)
        if teammate_pos is not None:
            if manhattanDistance(opponent_pos, teammate_pos) <= SIGHT_RANGE:
                return True

    return False


@profile
def update_belief(
    prev_belief: List[List[float]],
    game_state: GameState,
    opponent_idx: int,
    observer_idx: int,
) -> List[List[float]]:
    """
    Update belief distribution for one opponent.

    Steps:
    1. If opponent is visible, reset belief to exact position
    2. If opponent just moved, propagate belief forward
    3. Apply Bayesian update using noisy distance observation

    Args:
        prev_belief: Previous belief distribution [[prob]]
        game_state: Current game state
        opponent_idx: Index of opponent to update
        observer_idx: Index of agent doing the observing

    Returns:
        Updated belief distribution
    """
    walls = game_state.getWalls()
    width, height = walls.width, walls.height

    # Step 1: Check if opponent is visible
    if is_opponent_visible(game_state, observer_idx, opponent_idx):
        # EXACT OBSERVATION - reset belief to certain position
        opponent_pos = game_state.getAgentPosition(opponent_idx)
        new_belief = [[0.0 for _ in range(height)] for _ in range(width)]
        x, y = int(opponent_pos[0]), int(opponent_pos[1])
        new_belief[x][y] = 1.0
        return new_belief

    # Step 2: Propagate beliefs forward ONLY if this opponent just moved
    # The agent who just moved before observer is (observer_idx - 1) % 4
    prev_agent_idx = (observer_idx - 1) % 4

    if opponent_idx == prev_agent_idx:
        new_beliefs = [[0.0 for _ in range(height)] for _ in range(width)]

        for x in range(width):
            for y in range(height):
                if prev_belief[x][y] > 0:
                    # From position (x,y), distribute probability to reachable neighbors
                    neighbors = Actions.getLegalNeighbors((x, y), walls)
                    prob_per_neighbor = prev_belief[x][y] / len(neighbors)

                    for nx, ny in neighbors:
                        new_beliefs[nx][ny] += prob_per_neighbor

        prev_belief = new_beliefs

    # Step 3: Bayesian update using noisy distance observation
    observer_pos = game_state.getAgentPosition(observer_idx)
    noisy_distances = game_state.getAgentDistances()
    assert noisy_distances, "No distances provided"
    noisy_dist = noisy_distances[opponent_idx]

    updated_belief = [[0.0 for _ in range(height)] for _ in range(width)]
    for x in range(width):
        for y in range(height):
            if prev_belief[x][y] > 0:
                # P(pos | observation) ∝ P(observation | pos) * P(pos)
                prior = prev_belief[x][y]
                true_dist = manhattanDistance(observer_pos, (x, y))
                likelihood = game_state.getDistanceProb(true_dist, noisy_dist)
                updated_belief[x][y] = prior * likelihood

    # Step 4: Normalize
    total_prob = sum(sum(row) for row in updated_belief)
    if total_prob > 0:
        for x in range(width):
            for y in range(height):
                updated_belief[x][y] /= total_prob

    return updated_belief


@profile
def update_all_beliefs(
    prev_beliefs: Dict[int, List[List[float]]],
    game_state: GameState,
    observer_idx: int,
) -> Dict[int, List[List[float]]]:
    """
    Update belief distributions for all opponents from observer's perspective.

    Args:
        prev_beliefs: Previous beliefs for all agents
        game_state: Current game state
        observer_idx: Index of agent doing the observing

    Returns:
        Updated beliefs for all agents
    """
    # Determine observer's opponents
    if observer_idx in [0, 2]:
        opponents = [1, 3]  # Red team observes Blue team
    else:
        opponents = [0, 2]  # Blue team observes Red team

    updated_beliefs = prev_beliefs.copy()

    for opponent_idx in opponents:
        updated_beliefs[opponent_idx] = update_belief(
            prev_beliefs[opponent_idx],
            game_state,
            opponent_idx,
            observer_idx,
        )

    return updated_beliefs


def get_belief_as_array(
    belief: List[List[float]], width: int, height: int
) -> List[List[float]]:
    """
    Helper to ensure belief is proper 2D array format.

    Args:
        belief: Belief distribution (may be None or malformed)
        width: Board width
        height: Board height

    Returns:
        Valid 2D probability array
    """
    if belief is None:
        # Return uniform distribution over non-wall positions
        # (This shouldn't happen with proper initialization)
        return [[1.0 / (width * height) for _ in range(height)] for _ in range(width)]

    return belief
