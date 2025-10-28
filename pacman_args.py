from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainingArgs:
    """AlphaZero training configuration"""

    # Neural network args
    lr: float = 1e-3
    weight_decay: float = 0
    epochs: int = 10
    batch_size: int = 64

    # Training iterations
    numIters: int = 1
    numEps: int = 1
    tempThreshold: int = 15
    updateThreshold: float = 0.55

    # MCTS
    numMCTSSims: int = 1
    cpuct: float = 1.0

    # Memory
    maxlenOfQueue: int = 200000
    numItersForTrainExamplesHistory: int = 20

    # Evaluation
    arenaCompare: int = 20
    checkpoint: str = "./checkpoint/"
    load_model: bool = True
    load_folder_file: Tuple[str, str] = ("./checkpoint/", "temp.pth.tar")
    numItersForCheckpoint: int = 5

    # Bootstrap mode (use expert policy for initial iterations)
    bootstrap_iterations: int = 1  # Number of iterations to use staffTeam expert policy

    # Data filtering
    filter_winning_only: bool = False  # If True, only train on examples where v=1.0 (winning positions)
