# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Basic Game Execution
```bash
# Run a basic game with default teams
python capture.py

# Show all available options
python capture.py --help

# Run specific teams against each other
python capture.py -r berkeleyTeam -b myTeam

# Run with different layouts
python capture.py -l mediumCapture
python capture.py -l layouts/strategicCapture.lay

# Interactive play (keyboard control)
python capture.py --keys0  # Control red agent 0
python capture.py --keys1  # Control red agent 1
python capture.py --keys2  # Control blue agent 0
python capture.py --keys3  # Control blue agent 1
```

### Testing and Development
```bash
# Run multiple games for testing
python capture.py -n 10 -q  # 10 games, quiet mode

# Run with fixed random seed for reproducible results
python capture.py -f

# Run in training mode (suppresses output)
python capture.py -x 100  # 100 training games

# Text-only mode (no graphics)
python capture.py -t

# Super quiet mode (minimal output)
python capture.py -Q
```

### Team Development
```bash
# Test your team against baseline
python capture.py -r myTeam -b berkeleyTeam

# Test different agent configurations
python capture.py -r myTeam -b myTeam --redOpts first=OffensiveAgent --blueOpts first=DefensiveAgent
```

## Architecture Overview

### Core Components

**Game Engine (`capture.py`)**
- Main orchestrator for Pacman Capture the Flag games
- Implements game rules, scoring, and win conditions
- Manages team creation and agent coordination
- Handles time limits (1 second per move, 15 seconds initialization)

**Agent Framework (`captureAgents.py`)**
- `CaptureAgent` base class for all team agents
- Provides utilities for team coordination and game state analysis
- Methods like `getFood()`, `getOpponents()`, `getTeam()` for game information
- Distance calculation utilities and maze navigation

**Team Structure**
- Teams consist of exactly 2 agents (indices 0,2 for red; 1,3 for blue)
- Each team file must implement `createTeam(firstIndex, secondIndex, isRed, **kwargs)`
- Agents alternate between Ghost (defensive) and Pacman (offensive) roles based on map position

### Key Development Files

**`myTeam.py`** - Primary file for agent development
- Contains custom agent implementations
- Uses PDDL planning via `lib_piglet` library
- Imports from `lib_piglet.utils.pddl_solver`, `lib_piglet.domains.pddl`
- Defines distance constants: CLOSE_DISTANCE=4, MEDIUM_DISTANCE=15, LONG_DISTANCE=25

**Game State Management**
- `GameState` objects provide game information via methods:
  - `getRedFood()`, `getBlueFood()` - food locations
  - `getRedTeamIndices()`, `getBlueTeamIndices()` - team member indices
  - `getAgentDistances()` - noisy distance readings (±6 noise range)
  - `isOnRedTeam()`, `isOnBlueTeam()` - team membership

**Layout System**
- Map layouts stored in `/layouts/*.lay` files
- Available layouts: defaultCapture, mediumCapture, strategicCapture, tinyCapture, etc.
- Layouts define maze structure, food placement, and starting positions

### External Dependencies

**PDDL Planning Library (`lib_piglet`)**
- Used for automated planning in myTeam.py and staffTeam.py
- Provides `pddl_solver`, `pddl_state`, and `Action` classes
- PDDL files: `myTeam.pddl`, `staffTeam.pddl` define planning domains

**Graphics and Display**
- `graphicsDisplay.py`, `captureGraphicsDisplay.py` - visual game rendering
- `textDisplay.py` - ASCII-based display for headless execution
- Graphics can be disabled with `-t` or `-q` flags

### Game Rules Summary

**Objective**: Capture opponent's food while defending your own
**Scoring**: +1 point per food pellet returned to home side
**Agents**: Switch between Ghost (defensive) and Pacman (offensive) based on map position
**Observation**: Limited to 5-square Manhattan distance, plus noisy distance readings
**Time Limits**: 1 second per move, 3-second forfeit threshold, 1200 total moves per game
**Power Capsules**: Make opposing ghosts vulnerable for 40 moves

### Development Notes

- Agents must inherit from `CaptureAgent` class
- Use `registerInitialState()` for 15-second initialization period
- Implement `chooseAction()` method for move selection
- Team coordination via shared game state observation
- Consider probabilistic tracking for hidden opponent positions
- Balance offensive and defensive strategies based on game state