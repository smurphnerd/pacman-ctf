# Expert Data Collection Guide

This guide shows you how to collect expert gameplay data for training your AlphaZero Pacman agent.

## Quick Start

### 1. Play and Record Games

```bash
# Record games with keyboard control (both players)
python capture.py --keys0 --keys1 --record -n 5 --frameTime 0.3 -l mediumCapture

# Or play alone against baseline AI
python capture.py --keys0 --record -n 5 --frameTime 0.3 -l mediumCapture
```

### 2. Convert Replays to Training Data

First, install required dependencies if you haven't:
```bash
pip install gymnasium numpy
```

Then convert the replay files:
```bash
python convert_replay_to_examples.py replay-0 replay-1 replay-2 replay-3 replay-4
```

This creates `checkpoint/expert_data.pth.tar.examples`

### 3. Load in Training

```python
# In your training script (e.g., main_pacman.py)
args.load_folder_file = ('checkpoint', 'expert_data.pth.tar')
coach.loadTrainExamples()
coach.learn()
```

---

## Keyboard Controls

### Player 1 (--keys0)
- **W** or **↑**: Move North
- **S** or **↓**: Move South
- **A** or **←**: Move West
- **D** or **→**: Move East
- **Q**: Stop

### Player 2 (--keys1)
- **I**: Move North
- **K**: Move South
- **J**: Move West
- **L**: Move East
- **U**: Stop

---

## Command Options

### Recording Options

```bash
# Basic recording
python capture.py --keys0 --record

# Multiple games
python capture.py --keys0 --record -n 10

# Slower gameplay (more time to react)
python capture.py --keys0 --record --frameTime 0.5

# Different layouts
python capture.py --keys0 --record -l tinyCapture      # Small map
python capture.py --keys0 --record -l mediumCapture    # Medium map (recommended)
python capture.py --keys0 --record -l defaultCapture   # Default map

# Shorter games (for testing)
python capture.py --keys0 --record -i 300  # 300 moves instead of 1200
```

### Useful Combinations

```bash
# Beginner-friendly: slow, small map
python capture.py --keys0 --keys1 --record -n 3 --frameTime 0.5 -l tinyCapture

# Intermediate: moderate speed, medium map
python capture.py --keys0 --keys1 --record -n 5 --frameTime 0.3 -l mediumCapture

# Expert: normal speed, full map
python capture.py --keys0 --keys1 --record -n 10 --frameTime 0.1 -l defaultCapture
```

---

## Data Format

The converter creates training examples in the format expected by Coach.py:

```python
# Each example is a tuple:
(observation, policy, value)

# observation: Dict with keys 'spatial' and 'scalar'
#   - spatial: [11, width, height] numpy array
#   - scalar: [14] numpy array

# policy: [5] numpy array (one-hot for expert moves)
#   - Index 0: North
#   - Index 1: South
#   - Index 2: East
#   - Index 3: West
#   - Index 4: Stop

# value: +1.0 (win), -1.0 (loss), 0.0 (tie)
#   - From the perspective of the agent who made this move
```

---

## Advanced Usage

### Custom Output Location

```bash
python convert_replay_to_examples.py replay-* --output my_expert_data.pth.tar.examples
```

### Batch Convert All Replays

```bash
# On Unix/Mac
python convert_replay_to_examples.py replay-*

# On Windows PowerShell
python convert_replay_to_examples.py (Get-ChildItem replay-*).Name
```

### Check Replay Contents

```python
import pickle

with open('replay-0', 'rb') as f:
    data = pickle.load(f)

print(f"Layout: {data['layout'].name}")
print(f"Total moves: {len(data['actions'])}")
print(f"Red team: {data['redTeamName']}")
print(f"Blue team: {data['blueTeamName']}")
```

---

## Tips for Good Expert Data

1. **Play strategically** - The AI will learn from your moves
2. **Record winning games** - Data from wins is most valuable
3. **Vary your strategy** - Don't play the same way every time
4. **Use different layouts** - Helps with generalization
5. **Record 10-50 games** - More data = better learning
6. **Delete bad replays** - Remove games where you played poorly

---

## Troubleshooting

### "No module named 'gymnasium'"
```bash
pip install gymnasium
```

### "File replay-0 not found"
Make sure you ran `python capture.py --record` first.

### "Illegal action at step X"
This can happen if the replay file is corrupted. Delete and re-record that game.

### Game is too fast
Increase `--frameTime`:
```bash
python capture.py --keys0 --record --frameTime 1.0  # Very slow
```

### Can't control the agent
- Make sure you're using the correct keyboard controls
- Check that you used `--keys0` or `--keys1` in the command
- Verify the graphics window has focus (click on it)

---

## Example Workflow

```bash
# Step 1: Record 10 expert games
python capture.py --keys0 --keys1 --record -n 10 --frameTime 0.3 -l mediumCapture

# Step 2: Check that replay files were created
ls replay-*

# Step 3: Convert to training data
python convert_replay_to_examples.py replay-*

# Step 4: Use in training (in your Python script)
# args.load_folder_file = ('checkpoint', 'expert_data.pth.tar')
# coach.loadTrainExamples()
# coach.learn()
```

---

## Understanding the Output

When you run the converter, you'll see output like:

```
Converting replay-0...
  Layout: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  Total moves: 87
  Game ended at step 87/87
  Final score: 5 (Red wins)
  Collected 87 training examples

============================================================
Summary:
  Converted 1/1 replay files
  Total games: 1
  Total examples: 87
  Saving to: checkpoint/expert_data.pth.tar.examples
  ✓ Saved successfully!

To use in training:
  args.load_folder_file = ('checkpoint', 'expert_data.pth.tar')
  coach.loadTrainExamples()
============================================================
```

This means you successfully created 87 training examples from one game!
