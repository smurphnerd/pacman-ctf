"""
Quick integration test for AlphaZero components.
"""

import numpy as np
import torch
from PacmanGame import PacmanGame
from PacmanNeuralNet import PacmanNeuralNet

print("=" * 60)
print("AlphaZero Integration Test")
print("=" * 60)

# Test 1: Create game
print("\n[Test 1] Creating PacmanGame...")
game = PacmanGame(layout_name='tinyCapture', time_limit=1200)
print(f"✓ Game created")
print(f"  Board size: {game.getBoardSize()}")
print(f"  Action size: {game.getActionSize()}")

# Test 2: Get initial board
print("\n[Test 2] Getting initial board...")
board = game.getInitBoard()
print(f"✓ Initial board created")
print(f"  Spatial shape: {board['spatial'].shape}")
print(f"  Scalar shape: {board['scalar'].shape}")
print(f"  Current agent: {board['current_agent']}")

# Test 3: Get legal actions
print("\n[Test 3] Getting legal actions...")
player = 1 if board['current_agent'] in [0, 2] else -1
legal_moves = game.getValidMoves(board, player)
print(f"✓ Legal moves: {legal_moves}")

# Test 4: Test getNextState
print("\n[Test 4] Testing getNextState...")
action = np.where(legal_moves == 1)[0][0]  # Pick first legal action
next_board, next_player = game.getNextState(board, player, action)
print(f"✓ Next state computed")
print(f"  Next agent: {next_board['current_agent']}")
print(f"  Next player: {next_player}")

# Test 5: Test game termination
print("\n[Test 5] Testing game termination...")
result = game.getGameEnded(board, player)
print(f"✓ Game ended result: {result} (0 = ongoing)")

# Test 6: Test canonical form
print("\n[Test 6] Testing canonical form...")
canonical = game.getCanonicalForm(board, player)
print(f"✓ Canonical form computed")

# Test 7: Test symmetries
print("\n[Test 7] Testing symmetries...")
pi = np.zeros(5)
pi[action] = 1.0  # One-hot policy
symmetries = game.getSymmetries(board, pi)
print(f"✓ Symmetries generated: {len(symmetries)} forms")

# Test 8: Test string representation
print("\n[Test 8] Testing string representation...")
string_rep = game.stringRepresentation(board)
print(f"✓ String representation: {len(string_rep)} chars")

# Test 9: Create neural network
print("\n[Test 9] Creating neural network...")
nn_args = {
    'lr': 0.001,
    'weight_decay': 1e-4,
    'epochs': 1,
    'batch_size': 2,
}
nnet = PacmanNeuralNet(game, nn_args)
print(f"✓ Neural network created")
print(f"  Device: {nnet.device}")
print(f"  Parameters: {sum(p.numel() for p in nnet.nnet.parameters()):,}")

# Test 10: Test prediction
print("\n[Test 10] Testing neural network prediction...")
pi, v = nnet.predict(board)
print(f"✓ Prediction computed")
print(f"  Policy shape: {pi.shape}")
print(f"  Policy: {pi}")
print(f"  Value: {v}")

# Test 11: Test training
print("\n[Test 11] Testing neural network training...")
# Create dummy examples
examples = []
for i in range(10):
    dummy_board = game.getInitBoard()
    dummy_pi = np.ones(5) / 5  # Uniform policy
    dummy_v = 0.5
    examples.append((dummy_board, dummy_pi, dummy_v))

nnet.train(examples)
print(f"✓ Training completed")

# Test 12: Test checkpoint save/load
print("\n[Test 12] Testing checkpoint save/load...")
import os
import shutil

test_folder = './test_checkpoint'
if os.path.exists(test_folder):
    shutil.rmtree(test_folder)

nnet.save_checkpoint(folder=test_folder, filename='test.pth.tar')
nnet.load_checkpoint(folder=test_folder, filename='test.pth.tar')
print(f"✓ Checkpoint save/load successful")

# Cleanup
shutil.rmtree(test_folder)
board['env'].close()

print("\n" + "=" * 60)
print("All Tests Passed! ✓")
print("=" * 60)
print("\nYou can now run: python main_pacman.py")
