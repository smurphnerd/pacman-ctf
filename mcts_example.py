"""
Example MCTS integration with AlphaZero Pacman environment.

This shows how to use the environment for AlphaZero-style self-play
where a single neural network plays all 4 agents.
"""

import numpy as np
import torch
from typing import List, Tuple
from alphazero_env import AlphaZeroPacmanEnv


class SimpleMCTS:
    """
    Simplified MCTS for demonstration.

    In practice, you'd use a full AlphaZero implementation with:
    - Neural network for policy and value
    - Tree search with UCB selection
    - Self-play data generation
    - Network training
    """

    def __init__(self, env: AlphaZeroPacmanEnv, network, num_simulations: int = 100):
        self.env = env
        self.network = network  # Your SmurphCNN
        self.num_simulations = num_simulations

    def search(self, env_state: AlphaZeroPacmanEnv) -> Tuple[int, float]:
        """
        Run MCTS search from current state.

        Args:
            env_state: Current environment state

        Returns:
            best_action: Action index to take
            value: Estimated value of current state
        """
        # Get legal actions
        legal_mask = env_state.get_legal_actions_mask()
        legal_actions = np.where(legal_mask > 0)[0]

        if len(legal_actions) == 0:
            return 4, 0.0  # STOP action

        # Get neural network predictions
        obs = env_state._get_observation()
        spatial_tensor = torch.from_numpy(obs["spatial"]).unsqueeze(0)  # Add batch dim
        scalar_tensor = torch.from_numpy(obs["scalar"]).unsqueeze(0)  # Add batch dim

        with torch.no_grad():
            policy_logits, value = self.network(spatial_tensor, scalar_tensor)

        # Mask illegal actions
        policy_logits = policy_logits.squeeze(0)  # Remove batch dim
        policy_logits = policy_logits.numpy()
        policy_logits[legal_mask == 0] = -np.inf

        # Get action probabilities
        action_probs = np.exp(policy_logits - np.max(policy_logits))
        action_probs = action_probs / np.sum(action_probs)

        # Run simulations (simplified - real MCTS would build tree)
        action_values = np.zeros(5)
        action_counts = np.zeros(5)

        for _ in range(self.num_simulations):
            # Clone environment for simulation
            sim_env = env_state.clone()

            # Sample action proportional to policy
            action = np.random.choice(5, p=action_probs)

            # Simulate to end (or depth limit)
            terminal_value = self._simulate(sim_env, action)

            action_values[action] += terminal_value
            action_counts[action] += 1

        # Select best action
        avg_values = np.where(
            action_counts > 0, action_values / action_counts, -np.inf
        )
        best_action = int(np.argmax(avg_values))

        return best_action, value.item()

    def _simulate(
        self, env: AlphaZeroPacmanEnv, first_action: int, max_depth: int = 50
    ) -> float:
        """
        Simulate a game from current state using network policy.

        Args:
            env: Cloned environment to simulate in
            first_action: First action to take
            max_depth: Maximum simulation depth

        Returns:
            terminal_reward: Final reward from perspective of agent who started simulation
        """
        starting_agent = env.current_agent

        # Take first action
        obs, reward, terminated, truncated, info = env.step(first_action)

        # If game ended, return reward
        if terminated or truncated:
            # Adjust reward to starting agent's perspective
            return self._adjust_reward_perspective(
                reward, starting_agent, env.current_agent
            )

        # Continue simulation with network policy
        for depth in range(max_depth):
            # Get network prediction
            spatial_tensor = torch.from_numpy(obs["spatial"]).unsqueeze(0)
            scalar_tensor = torch.from_numpy(obs["scalar"]).unsqueeze(0)
            with torch.no_grad():
                policy_logits, value = self.network(spatial_tensor, scalar_tensor)

            # Get legal actions
            legal_mask = env.get_legal_actions_mask()
            policy_logits = policy_logits.squeeze(0).numpy()
            policy_logits[legal_mask == 0] = -np.inf

            # Sample action
            action_probs = np.exp(policy_logits - np.max(policy_logits))
            action_probs = action_probs / np.sum(action_probs)
            action = np.random.choice(5, p=action_probs)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                # Adjust reward to starting agent's perspective
                return self._adjust_reward_perspective(
                    reward, starting_agent, env.current_agent
                )

        # If we hit depth limit, use value estimate
        return value.item()

    def _adjust_reward_perspective(
        self, reward: float, from_agent: int, to_agent: int
    ) -> float:
        """
        Adjust reward from one agent's perspective to another's.

        Args:
            reward: Reward from from_agent's perspective
            from_agent: Agent index reward is from perspective of
            to_agent: Agent index to convert to

        Returns:
            Adjusted reward
        """
        # Check if agents are on same team
        from_team = 0 if from_agent in [0, 2] else 1  # 0=Red, 1=Blue
        to_team = 0 if to_agent in [0, 2] else 1

        if from_team == to_team:
            return reward
        else:
            return -reward  # Opposite teams, flip reward


def self_play_episode(env: AlphaZeroPacmanEnv, network) -> List[Tuple]:
    """
    Generate one self-play episode for training.

    Args:
        env: AlphaZero environment
        network: Neural network (SmurphCNN)

    Returns:
        episode_data: List of (state, policy, value, agent) tuples
    """
    episode_data = []
    mcts = SimpleMCTS(env, network, num_simulations=50)

    obs, info = env.reset()

    while True:
        current_agent = env.current_agent

        # Run MCTS search
        action, value = mcts.search(env)

        # Store state and policy (simplified - should store MCTS visit counts)
        policy = np.zeros(5)
        policy[action] = 1.0  # In real implementation, use MCTS visit counts

        episode_data.append((obs.copy(), policy, value, current_agent))

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            # Backpropagate final reward to all states
            final_reward = reward
            for i in range(len(episode_data)):
                state, policy, _, agent = episode_data[i]
                # Adjust reward to each agent's perspective
                agent_team = 0 if agent in [0, 2] else 1
                current_team = 0 if env.current_agent in [0, 2] else 1
                adjusted_reward = (
                    final_reward if agent_team == current_team else -final_reward
                )
                episode_data[i] = (state, policy, adjusted_reward, agent)
            break

    return episode_data


# Example usage
if __name__ == "__main__":
    from myTeam import SmurphCNN

    # Create environment
    env = AlphaZeroPacmanEnv(layout_name="mediumCapture")

    # Create network (all agents share same network)
    network = SmurphCNN(num_actions=5)
    network.eval()

    # Generate one self-play episode
    print("Running self-play episode...")
    episode_data = self_play_episode(env, network)

    print(f"Episode length: {len(episode_data)} transitions")
    print(f"Final reward: {episode_data[-1][2]}")

    # Show perspective transformation
    print("\nTesting perspective transformation:")
    obs, info = env.reset()
    print(f"Agent 0 (Red) spatial shape: {obs['spatial'].shape}")
    print(f"Agent 0 (Red) scalar shape: {obs['scalar'].shape}")
    print(f"Scalar features: {obs['scalar']}")

    # Step to agent 1 (Blue)
    env.step(4)  # STOP
    obs, _, _, _, _ = env.step(4)  # STOP
    print(f"\nAgent 1 (Blue) spatial shape: {obs['spatial'].shape}")
    print(f"Agent 1 (Blue) scalar shape: {obs['scalar'].shape}")
    print("(Board should be flipped for Blue)")
    print(f"Scalar features: {obs['scalar']}")

    env.close()
