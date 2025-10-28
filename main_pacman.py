import logging

import coloredlogs

from Coach import Coach
from PacmanGame import PacmanGame as Game
from PacmanNeuralNet import PacmanNeuralNet as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.

args = dotdict(
    {
        "numIters": 1,
        "numEps": 0,  # Number of complete self-play games to simulate during a new iteration.
        "tempThreshold": 15,  #
        "updateThreshold": 0.55,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlenOfQueue": 200000,  # Number of game examples to train the neural networks.
        "numMCTSSims": 5,  # Number of games moves for MCTS to simulate.
        "arenaCompare": 20,  # Number of games to play during arena play to determine if new net will be accepted.
        "cpuct": 1,
        "checkpoint": "./temp/",
        "load_model": True,
        "load_folder_file": ("./checkpoint", "expert_data.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
        "lr": 1e-3,
        "weight_decay": 0,
        "epochs": 10,
        "batch_size": 64,
        "bootstrap_iterations": 0,
        "filter_winning_only": False,  # Set to True to only train on winning examples
    }
)


def main():
    log.info("Loading %s...", Game.__name__)
    g = Game(layout_name="mediumCapture")

    log.info("Loading %s...", nn.__name__)
    nnet = nn(g, args)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning("Not loading a checkpoint!")

    log.info("Loading the Coach...")
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info("Starting the learning process 🎉")
    c.learn()


if __name__ == "__main__":
    main()
