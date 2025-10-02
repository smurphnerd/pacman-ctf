## AgentState

```python
class AgentState:
    """
    AgentStates hold the state of an agent (configuration, speed, scared, etc).
    """

    def __init__( self, startConfiguration, isPacman ):
        self.start = startConfiguration
        self.configuration = startConfiguration
        self.isPacman = isPacman
        self.scaredTimer = 0
        self.numCarrying = 0
        self.numReturned = 0
```

## What is contained in an individual agent's state observation?

- Current AgentState
- Teammate's AgentState
- Opponent's AgentState within 5 Manhattan distance of either teammate
- Noisy AgentState for opponents outside of 5 Manhattan distance
- Full game board (walls, food, capsules, etc)

### Noisy AgentState

```python
class AgentState:
    """
    AgentStates hold the state of an agent (configuration, speed, scared, etc).
    """

    def __init__( self, startConfiguration, isPacman ):
        self.start = startConfiguration
        self.configuration = None
        self.isPacman = isPacman
        self.scaredTimer = 0
        self.numCarrying = 0
        self.numReturned = 0
        self.distanceToAgent = noisyDistance(agentPos, self.configuration.pos)
```

## How to encode?

- Relative position of current agent (first person view) e.g. (1,0) is to the right of the agent
- Enum for what can be seen e.g. 0=wall, 1=food, 2=capsule, 3=teammate, 4=opponent
- FOV should be a fixed size (maybe within 10 manhatten distance) and centered on the agent

- Midpoint of map

