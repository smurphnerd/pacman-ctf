# Pacman Capture the Flag: AI Agent Challenge

## Problem Overview

Design and implement AI agents to compete in a team-based variant of Pacman called "Capture the Flag." Your task is to create a team of 2 autonomous agents that work together to outscore an opposing team in a competitive multi-agent environment with partial observability and real-time constraints.

## Game Environment

### Map Structure
- **Symmetric Layout**: The game board is divided into two halves - red territory (left) and blue territory (right)
- **Variable Dimensions**: Board sizes range from 20×7 (tiny) to 46×25 (jumbo), with typical competition size around 32×15
- **Elements**:
  - Walls (`%`) block movement
  - Food dots (`.`) are scattered throughout both territories
  - Power capsules (`o`) provide temporary advantages
  - Open spaces allow free movement

### Team Composition
- **Red Team**: Agents with indices 0 and 2 (start on left side)
- **Blue Team**: Agents with indices 1 and 3 (start on right side)
- Each team defends their territory while attempting to invade the opponent's territory

### Agent Behavior
- **On Home Territory**: Agents function as **Ghosts** (defensive mode)
  - Can capture invading Pacman agents
  - Move at full speed
  - Protect food and capsules

- **On Enemy Territory**: Agents function as **Pacman** (offensive mode)
  - Can collect enemy food dots
  - Vulnerable to enemy ghosts
  - Must return home to score collected food

## Scoring System

### Point Values
- **+1 point** per food dot successfully returned to home territory
- **No points** for eating opponents (they simply respawn)
- **Food carrying**: Dots are stored in the agent until they return home

### Win Conditions
- **Primary**: First team to return all but 2 of opponent's food dots
- **Secondary**: Highest score when 1200 total moves are reached (300 moves per agent)
- **Tie**: Score of 0 (equal food returned)

## Game Mechanics

### Power Capsules
- Consuming a power capsule makes opposing ghosts "scared" for 40 moves
- Scared ghosts can be eaten by Pacman agents
- Eaten scared ghosts respawn at starting position (no longer scared)

### Death and Respawn
- Pacman caught by ghost: respawns at starting position, carried food scatters back to original locations
- Ghost eaten by powered Pacman: respawns at starting position immediately

### Observation Model
- **Perfect Information**: Agent positions within 5 Manhattan distance
- **Noisy Information**: Distance readings to ALL agents with ±6 uniform random noise
- **Hidden Information**: Exact positions of distant opponents must be inferred

## Technical Constraints

### Real-Time Performance
- **Action Deadline**: 1 second per move (automatic random action if exceeded)
- **Initialization Time**: 15 seconds allowed for setup
- **Forfeit Conditions**: 3 warnings or single 3+ second delay

### Information Access

#### At Initialization (15-second setup period):
- Complete map layout (walls, food positions, capsule locations)
- Team assignments and starting positions
- Maze distance calculations can be precomputed

#### Each Turn (1-second action deadline):
- **Observable**: Exact state of agents within 5 Manhattan distance
- **Noisy**: Distance readings to all agents (true distance ± random noise in [-6, +6])
- **Game State**: Current score, remaining food/capsules, legal actions
- **History**: Access to previous observations for tracking

## Implementation Interface

### Required Team Structure
```python
def createTeam(firstIndex, secondIndex, isRed, **kwargs):
    """Create team of 2 agents"""
    return [Agent1(firstIndex), Agent2(secondIndex)]

class YourAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        """Optional 15-second initialization"""
        pass

    def chooseAction(self, gameState):
        """Required: Return action within 1 second"""
        return action  # One of: North, South, East, West, Stop
```

### Key Information Methods
- **Team Context**: `getTeam()`, `getOpponents()`, `isOnRedTeam()`
- **Spatial**: `getFood()`, `getCapsules()`, `getAgentPosition()`, `getMazeDistance()`
- **Game State**: `getScore()`, `getLegalActions()`, `getAgentDistances()`
- **History**: `getCurrentObservation()`, `getPreviousObservation()`

## Strategic Challenges

### Multi-Agent Coordination
- Coordinate 2 agents with shared objectives but independent decision-making
- Balance offensive and defensive roles dynamically
- Communicate implicitly through observable actions

### Partial Observability
- Track hidden opponent positions using noisy distance readings
- Maintain probabilistic beliefs about enemy locations and intentions
- Plan under uncertainty with limited information

### Real-Time Decision Making
- Process complex game state and make decisions within 1-second deadline
- Balance computation time between perception, planning, and execution
- Handle time pressure while maintaining strategic coherence

### Dynamic Strategy Adaptation
- Adapt tactics based on score differential and remaining time
- Switch between offensive and defensive priorities
- Respond to opponent strategies and counter-adaptations

## Success Metrics

### Primary Objectives
1. **Survival**: Avoid timeout penalties and maintain consistent performance
2. **Scoring**: Successfully return food dots to home territory
3. **Defense**: Prevent opponents from scoring on your territory
4. **Efficiency**: Maximize points per risk taken

### Evaluation Scenarios
- **Balanced opponents**: Even skill level requiring strategic depth
- **Aggressive opponents**: Heavy offensive pressure requiring strong defense
- **Defensive opponents**: Requiring efficient penetration strategies
- **Adaptive opponents**: Changing strategies mid-game

## Development Considerations

### Algorithm Design Space
- **Planning**: Search algorithms, hierarchical planning, real-time planning
- **Learning**: Reinforcement learning, opponent modeling, adaptation
- **Probabilistic Reasoning**: Belief tracking, particle filters, Bayesian inference
- **Game Theory**: Nash equilibria, minimax variants, cooperative strategies

### Performance Trade-offs
- **Computation vs. Accuracy**: More sophisticated models vs. real-time constraints
- **Exploration vs. Exploitation**: Information gathering vs. immediate scoring
- **Risk vs. Reward**: Aggressive play vs. conservative safety
- **Individual vs. Team**: Agent autonomy vs. coordination overhead

## Technical Notes

- Programming language: Python 3
- Action space: Discrete (5 directional actions)
- State space: Large but structured (grid-based with agent states)
- Environment: Deterministic except for distance noise
- Multi-agent: Competitive team-based setting
- Evaluation: Tournament-style competition against diverse opponents

This environment provides a rich testbed for multi-agent AI research, combining elements of planning under uncertainty, real-time decision making, partial observability, and cooperative strategy in a competitive setting.