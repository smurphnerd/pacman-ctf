"""
main_pacman.py - AlphaZero training script for Pacman Capture the Flag.

This configures and runs the AlphaZero training pipeline for 4-player Pacman CTF.
"""

from Coach import Coach
from PacmanGame import PacmanGame
from PacmanNeuralNet import PacmanNeuralNet
import numpy as np
from pacman_args import TrainingArgs


def main():
    """
    Configure and run AlphaZero training for Pacman CTF.
    """

    # ============ Game Configuration ============
    game_args = {
        "layout_name": "mediumCapture",  # or 'defaultCapture', 'strategicCapture', etc.
        "time_limit": 1200,  # Total agent moves before truncation
    }

    # ============ AlphaZero Training Configuration ============
    train_args = TrainingArgs(
        numIters=100,
        numEps=25,
        numMCTSSims=50,  # Reduced for faster testing
    )

    # ============ Initialize Components ============

    print("=" * 60)
    print("AlphaZero Training for Pacman Capture the Flag")
    print("=" * 60)
    print(f"Layout: {game_args['layout_name']}")
    print(f"Time Limit: {game_args['time_limit']} agent moves")
    print(f"MCTS Simulations: {train_args.numMCTSSims}")
    print(f"Episodes per Iteration: {train_args.numEps}")
    print(f"Total Iterations: {train_args.numIters}")
    print("=" * 60)

    # Create game
    print("\nInitializing game...")
    game = PacmanGame(**game_args)
    print(f"Board size: {game.getBoardSize()}")
    print(f"Action size: {game.getActionSize()}")

    # Create neural network
    print("\nInitializing neural network...")
    nnet = PacmanNeuralNet(game, train_args)
    print(f"Device: {nnet.device}")
    print(f"Model parameters: {sum(p.numel() for p in nnet.nnet.parameters()):,}")

    # Load checkpoint if requested
    if train_args.load_model:
        print(f"\nLoading checkpoint from {train_args.load_folder_file}...")
        nnet.load_checkpoint(*train_args.load_folder_file)

    # Create coach (training orchestrator)
    print("\nInitializing coach...")
    coach = Coach(game, nnet, train_args)

    # ============ Start Training ============

    print("\n" + "=" * 60)
    print("Starting AlphaZero Training Loop")
    print("=" * 60 + "\n")

    try:
        coach.learn()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        nnet.save_checkpoint(
            folder=train_args.checkpoint, filename="interrupted.pth.tar"
        )
        print("Checkpoint saved. You can resume training by setting 'load_model': True")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final model saved in: {train_args.checkpoint}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)

    main()
