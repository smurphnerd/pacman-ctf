"""
Convert recorded Pacman games to AlphaZero training examples.

Usage:
    1. Record games with keyboard:
       python capture.py --keys0 --keys1 --record -n 5

    2. Convert replays to training examples:
       python convert_replay_to_examples.py replay-0 replay-1 replay-2

    3. This creates checkpoint/expert_data.pth.tar.examples

    4. Load in training:
       args.load_folder_file = ('checkpoint', 'expert_data.pth.tar')
       coach.loadTrainExamples()
"""

import pickle
import numpy as np
from pickle import Pickler
import os
import sys
from typing import List, Tuple, Dict, Any

from game import Directions
from alphazero_env import AlphaZeroPacmanEnv

def convert_replay_to_examples(replay_file: str) -> List[Tuple]:
    """
    Convert a single replay file to AlphaZero training examples.

    Args:
        replay_file: Path to replay file (e.g., 'replay-0')

    Returns:
        List of (observation, policy, value) tuples for one game
    """
    print(f"\nConverting {replay_file}...")

    # Load recorded game
    if not os.path.exists(replay_file):
        print(f"Error: File {replay_file} not found!")
        return []

    with open(replay_file, 'rb') as f:
        recorded = pickle.load(f)

    game_layout = recorded['layout']
    actions = recorded['actions']  # List of (agent_idx, action) tuples

    print(f"  Layout: {game_layout.layoutText[0] if hasattr(game_layout, 'layoutText') else 'unknown'}")
    print(f"  Total moves: {len(actions)}")

    # Create environment with the same layout
    # We need to save the layout temporarily if it doesn't have a name
    layout_name = None
    temp_layout_file = None

    # Try to find the layout name
    if hasattr(game_layout, 'name') and game_layout.name:
        layout_name = game_layout.name
    else:
        # Create temporary layout file
        temp_layout_file = 'temp_replay_layout.lay'
        layout_text = '\n'.join(game_layout.layoutText)
        with open(f'layouts/{temp_layout_file}', 'w') as f:
            f.write(layout_text)
        layout_name = 'temp_replay_layout'

    # Create environment
    env = AlphaZeroPacmanEnv(layout_name=layout_name, render_mode=None)
    obs, info = env.reset()

    examples = []

    # Replay actions and collect (state, policy, placeholder_value)
    for step_num, (agent_idx, action) in enumerate(actions):
        # Verify it's the correct agent's turn
        if agent_idx != env.current_agent:
            print(f"  Warning: Agent mismatch at step {step_num}: expected {env.current_agent}, got {agent_idx}")
            # This might happen if the replay format differs - try to continue anyway

        # Get current observation from current agent's perspective
        obs = env._get_observation()

        # Convert action to policy vector (one-hot for expert moves)
        policy = np.zeros(5, dtype=np.float32)

        if action in env.DIR_TO_IDX:
            action_idx = env.DIR_TO_IDX[action]
        else:
            print(f"  Warning: Unknown action {action} at step {step_num}, using STOP")
            action_idx = env.DIR_TO_IDX[Directions.STOP]

        policy[action_idx] = 1.0

        # Store (state, policy, None) - value will be assigned later
        examples.append([obs, policy, None])

        # Take action in environment
        try:
            obs, reward, terminated, truncated, info = env.step(action_idx)
        except AssertionError as e:
            print(f"  Warning: Illegal action at step {step_num}: {e}")
            # Try STOP instead
            action_idx = env.DIR_TO_IDX[Directions.STOP]
            obs, reward, terminated, truncated, info = env.step(action_idx)

        # Check if game ended
        if terminated or truncated:
            final_score = info['score']
            print(f"  Game ended at step {step_num + 1}/{len(actions)}")
            print(f"  Final score: {final_score} ({'Red wins' if final_score > 0 else 'Blue wins' if final_score < 0 else 'Tie'})")
            break

    # Clean up temp layout
    if temp_layout_file:
        try:
            os.remove(f'layouts/{temp_layout_file}')
        except:
            pass

    # Assign values to all examples based on final outcome
    final_score = info['score']

    for i in range(len(examples)):
        agent_idx = i % 4  # Which agent made this move
        is_red = agent_idx in [0, 2]

        # Value from this agent's perspective
        if final_score > 0:  # Red won
            value = 1.0 if is_red else -1.0
        elif final_score < 0:  # Blue won
            value = -1.0 if is_red else 1.0
        else:  # Tie
            value = 0.0

        # Update the example with the value
        examples[i][2] = value

    print(f"  Collected {len(examples)} training examples")

    # Convert to tuple format expected by Coach
    # Each example should be (spatial_state, policy, value)
    # But we need to handle the dict observation format
    final_examples = []
    for obs_dict, policy, value in examples:
        # For Coach, we need a format that can be used by the neural network
        # The obs_dict has 'spatial' and 'scalar' keys
        # We'll store the full dict along with policy and value
        final_examples.append((obs_dict, policy, value))

    return final_examples


def convert_multiple_replays(replay_files: List[str], output_file: str = None) -> None:
    """
    Convert multiple replay files to a training examples history.

    Args:
        replay_files: List of replay file paths
        output_file: Output path (default: checkpoint/expert_data.pth.tar.examples)
    """
    if output_file is None:
        output_file = 'checkpoint/expert_data.pth.tar.examples'

    # Ensure checkpoint directory exists
    os.makedirs('checkpoint', exist_ok=True)

    # Convert each replay
    all_examples = []
    successful_conversions = 0

    for replay_file in replay_files:
        examples = convert_replay_to_examples(replay_file)
        if examples:
            all_examples.append(examples)
            successful_conversions += 1

    if not all_examples:
        print("\nNo examples collected! Check your replay files.")
        return

    # trainExamplesHistory format: list of lists
    # Each inner list contains examples from one iteration/game
    trainExamplesHistory = all_examples

    # Save in pickle format
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Converted {successful_conversions}/{len(replay_files)} replay files")
    print(f"  Total games: {len(trainExamplesHistory)}")
    print(f"  Total examples: {sum(len(game) for game in trainExamplesHistory)}")
    print(f"  Saving to: {output_file}")

    with open(output_file, 'wb+') as f:
        Pickler(f).dump(trainExamplesHistory)

    print(f"  ✓ Saved successfully!")
    print(f"\nTo use in training:")
    print(f"  args.load_folder_file = ('checkpoint', 'expert_data.pth.tar')")
    print(f"  coach.loadTrainExamples()")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: No replay files specified!")
        print("\nExample:")
        print("  python convert_replay_to_examples.py replay-0 replay-1 replay-2")
        sys.exit(1)

    replay_files = sys.argv[1:]

    # Check if output file is specified with --output
    output_file = None
    if '--output' in replay_files:
        idx = replay_files.index('--output')
        output_file = replay_files[idx + 1]
        replay_files = replay_files[:idx] + replay_files[idx+2:]

    convert_multiple_replays(replay_files, output_file)


if __name__ == '__main__':
    main()
