# Topological Map Analysis Guide

This guide explains how to use the preprocessed topological map data for strategic Pacman AI planning.

## Overview

The map is automatically analyzed during `registerInitialState()` and stored in the class variable `MixedAgent.MAP_TOPOLOGY`. This provides a strategic graph-based view of the map instead of a tile-by-tile view, enabling smarter pathfinding and tactical decisions.

## Accessing the Topology

In any agent method, access the topology via:
```python
topology = MixedAgent.MAP_TOPOLOGY
```

## Data Structures

### MapTopology Object

The `MAP_TOPOLOGY` object contains:

```python
@dataclass
class MapTopology:
    junctions: Dict[Tuple[int, int], Junction]
    corridors: Dict[int, Corridor]
    tile_to_corridor: Dict[Tuple[int, int], int]
    articulation_points: set
    dead_end_zones: Dict[Tuple[int, int], Tuple[int, int]]
```

---

## 1. Junctions (Decision Points)

**What**: Strategic nodes in the map - places where the agent must make a decision.

**Types**:
- `"junction"`: Tiles with 3+ non-wall neighbors (decision points)
- `"dead_end"`: Tiles with only 1 non-wall neighbor (dead ends)

**Note**: Tiles with exactly 2 neighbors are corridor tiles, NOT junctions.

### Junction Object

```python
@dataclass
class Junction:
    pos: Tuple[int, int]  # (x, y) position
    neighbors: List[Tuple[int, int]]  # Adjacent non-wall tiles
    junction_type: str  # "junction" or "dead_end"
    connected_junctions: Dict[Tuple[int, int], int]  # {junction_pos: corridor_length}
```

### Usage Examples

```python
# Get all junctions
junctions = topology.junctions

# Check if a position is a junction
if my_pos in junctions:
    junction = junctions[my_pos]
    print(f"Type: {junction.junction_type}")
    print(f"Num neighbors: {len(junction.neighbors)}")

# Find all junctions within distance
for junction_pos, junction in junctions.items():
    dist = self.getMazeDistance(my_pos, junction_pos)
    if dist <= 5:
        print(f"Nearby junction at {junction_pos}")

# Get connected junctions (direct graph neighbors)
if my_pos in junctions:
    for connected_pos, corridor_length in junctions[my_pos].connected_junctions.items():
        print(f"Junction {connected_pos} is {corridor_length} tiles away")
```

---

## 2. Corridors (Edges)

**What**: Paths connecting two junctions. All tiles along a corridor have exactly 2 neighbors (forward and backward).

### Corridor Object

```python
@dataclass
class Corridor:
    corridor_id: int
    junction_a: Tuple[int, int]  # First endpoint
    junction_b: Tuple[int, int]  # Second endpoint
    length: int  # Number of tiles in corridor
    path: List[Tuple[int, int]]  # All tiles including endpoints
```

### Usage Examples

```python
# Get corridor ID for current position
if my_pos in topology.tile_to_corridor:
    corridor_id = topology.tile_to_corridor[my_pos]
    corridor = topology.corridors[corridor_id]

    # Get both endpoints
    endpoint_a = corridor.junction_a
    endpoint_b = corridor.junction_b

    # Calculate distance to each endpoint
    dist_a = self.getMazeDistance(my_pos, endpoint_a)
    dist_b = self.getMazeDistance(my_pos, endpoint_b)

    print(f"In corridor {corridor_id}, length {corridor.length}")
    print(f"Exit A: {endpoint_a} ({dist_a} tiles away)")
    print(f"Exit B: {endpoint_b} ({dist_b} tiles away)")

# Check if enemy can block corridor exit
def can_escape_corridor(self, my_pos, enemy_positions, walls):
    """Check if we can reach a corridor exit before enemies"""
    if my_pos not in topology.tile_to_corridor:
        return True  # Not in a corridor

    corridor_id = topology.tile_to_corridor[my_pos]
    corridor = topology.corridors[corridor_id]

    # Check both exits
    for exit_junction in [corridor.junction_a, corridor.junction_b]:
        my_dist = self.getMazeDistance(my_pos, exit_junction)

        # Check if any enemy can reach this exit first
        enemy_can_block = False
        for enemy_pos in enemy_positions:
            enemy_dist = self.getMazeDistance(enemy_pos, exit_junction)
            if enemy_dist <= my_dist:
                enemy_can_block = True
                break

        # If at least one exit is safe, we can escape
        if not enemy_can_block:
            return True

    return False  # All exits are blocked
```

---

## 3. Articulation Points (Critical Choke Points)

**What**: Junctions that, if blocked/controlled, would split the map into disconnected regions. These are **critical strategic positions**.

### Data Structure

```python
articulation_points: set  # Set of (x, y) positions
```

### Strategic Importance

**For Attackers**:
- ⚠️ **Avoid when enemies nearby** - you can get trapped!
- These are natural ambush points
- If an enemy controls an articulation point, entire regions become dangerous

**For Defenders**:
- ✅ **Camp here to control large areas**
- One defender at an articulation point can monitor/block access to entire regions
- Force enemies into unfavorable positions

### Usage Examples

```python
# Check if a position is an articulation point
if my_pos in topology.articulation_points:
    print("WARNING: Critical choke point!")

# Find nearest articulation point
def nearest_articulation_point(self, pos):
    """Find the closest articulation point to a position"""
    min_dist = float('inf')
    nearest = None

    for art_point in topology.articulation_points:
        dist = self.getMazeDistance(pos, art_point)
        if dist < min_dist:
            min_dist = dist
            nearest = art_point

    return nearest, min_dist

# Check if path crosses an articulation point
def path_crosses_choke_point(self, start, goal):
    """Check if path from start to goal crosses an articulation point"""
    # Use A* or BFS to find path
    path = self.aStarSearch(start, goal)  # Your pathfinding implementation

    for pos in path:
        if pos in topology.articulation_points:
            return True, pos

    return False, None
```

---

## 4. Dead-End Zones (Trap Regions)

**What**: Regions of the map with **exactly ONE exit**. If an enemy controls the exit, you're trapped!

### Data Structure

```python
dead_end_zones: Dict[Tuple[int, int], Tuple[int, int]]
# Maps: tile_position -> exit_junction_position
```

**Important**: Only TRUE dead-end zones (one exit) are marked. Regions with multiple exits are NOT marked because they're safer.

### Strategic Importance

**For Attackers**:
- ⚠️ **NEVER enter if enemy can reach exit first**
- Calculate: `my_distance_to_exit` vs `enemy_distance_to_exit`
- If `enemy_distance <= my_distance`, **DO NOT ENTER** - it's a trap!

**For Defenders**:
- ✅ **If enemy is in a dead-end zone, go to the exit to trap them**
- Easy captures - enemy has nowhere else to go

### Usage Examples

```python
# Check if position is in a dead-end zone
if my_pos in topology.dead_end_zones:
    exit_junction = topology.dead_end_zones[my_pos]
    print(f"In dead-end zone! Exit at {exit_junction}")

# Safe to enter dead-end zone?
def is_safe_dead_end_zone(self, next_pos, enemy_virtual_states, walls):
    """
    Check if it's safe to move to next_pos (might be in dead-end zone).
    Returns True if safe, False if it's a trap.
    """
    # Not in a dead-end zone? Always safe
    if next_pos not in topology.dead_end_zones:
        return True

    # Get the exit junction for this zone
    exit_junction = topology.dead_end_zones[next_pos]

    # Calculate our distance to the exit
    my_dist_to_exit = self.getMazeDistance(next_pos, exit_junction)

    # Check each non-scared enemy ghost
    for enemy_idx, enemy_state in enemy_virtual_states.items():
        # Only worry about enemy ghosts (not Pacmen) that aren't scared
        if not enemy_state.isPacman and enemy_state.scaredTimer <= 2:
            enemy_pos = enemy_state.getPosition()
            if enemy_pos:
                enemy_dist_to_exit = self.getMazeDistance(enemy_pos, exit_junction)

                # Enemy can reach exit before us (or at same time)? TRAP!
                if enemy_dist_to_exit <= my_dist_to_exit:
                    return False  # NOT SAFE!

    return True  # Safe to enter

# Find all dead-end zones on the map
def analyze_dead_end_zones(self):
    """Get summary of all dead-end zones"""
    zone_to_tiles = {}

    for tile_pos, exit_junction in topology.dead_end_zones.items():
        if exit_junction not in zone_to_tiles:
            zone_to_tiles[exit_junction] = []
        zone_to_tiles[exit_junction].append(tile_pos)

    print(f"Found {len(zone_to_tiles)} dead-end zones:")
    for exit_junction, tiles in zone_to_tiles.items():
        print(f"  Exit {exit_junction}: {len(tiles)} tiles")

# Defender: Trap enemy in dead-end zone
def trap_enemy_in_dead_end(self, enemy_pos):
    """
    If enemy is in a dead-end zone, return the exit position to camp at.
    Returns None if enemy is not in a dead-end zone.
    """
    if enemy_pos in topology.dead_end_zones:
        exit_junction = topology.dead_end_zones[enemy_pos]
        print(f"Enemy trapped! Go to exit at {exit_junction}")
        return exit_junction
    return None
```

---

## Complete Usage Example: Safe Food Collection

```python
def choose_safe_food_target(self, gameState):
    """
    Choose a food pellet that is safe to reach, considering:
    - Corridor safety
    - Dead-end zone safety
    - Articulation point danger
    """
    my_pos = gameState.getAgentPosition(self.index)
    food_list = self.getFood(gameState).asList()
    walls = gameState.getWalls()
    topology = MixedAgent.MAP_TOPOLOGY

    # Get enemy positions
    enemy_virtual_states = {}
    for enemy_idx in self.getOpponents(gameState):
        enemy_state = gameState.getAgentState(enemy_idx)
        enemy_virtual_states[enemy_idx] = enemy_state

    safe_food = []

    for food_pos in food_list:
        # Check 1: Is food in a dead-end zone?
        if food_pos in topology.dead_end_zones:
            exit_junction = topology.dead_end_zones[food_pos]
            food_dist_to_exit = self.getMazeDistance(food_pos, exit_junction)

            # Can we reach the exit from the food position?
            is_trapped = False
            for enemy_idx, enemy_state in enemy_virtual_states.items():
                if not enemy_state.isPacman and enemy_state.scaredTimer <= 2:
                    enemy_pos = enemy_state.getPosition()
                    if enemy_pos:
                        enemy_dist = self.getMazeDistance(enemy_pos, exit_junction)
                        # Add buffer for safety
                        if enemy_dist <= food_dist_to_exit + 3:
                            is_trapped = True
                            break

            if is_trapped:
                continue  # Skip this food - it's a trap!

        # Check 2: Does path cross an articulation point with nearby enemies?
        path = self.aStarSearch(my_pos, food_pos, gameState, walls)
        crosses_dangerous_choke = False

        for path_pos in path:
            if path_pos in topology.articulation_points:
                # Check if any enemy is near this articulation point
                for enemy_idx, enemy_state in enemy_virtual_states.items():
                    if not enemy_state.isPacman:
                        enemy_pos = enemy_state.getPosition()
                        if enemy_pos:
                            dist = self.getMazeDistance(enemy_pos, path_pos)
                            if dist <= 3:  # Enemy near choke point
                                crosses_dangerous_choke = True
                                break
            if crosses_dangerous_choke:
                break

        if crosses_dangerous_choke:
            continue  # Skip - path crosses guarded choke point

        # This food is safe!
        safe_food.append(food_pos)

    # Choose closest safe food
    if safe_food:
        return min(safe_food, key=lambda f: self.getMazeDistance(my_pos, f))
    else:
        # No safe food - return closest food anyway (risky!)
        return min(food_list, key=lambda f: self.getMazeDistance(my_pos, f))
```

---

## Performance Notes

- All topology data is **computed once** during `registerInitialState()`
- **O(1) lookups** for checking if a tile is a junction, in a corridor, or in a dead-end zone
- **No runtime overhead** - just dictionary/set lookups
- The graph is **much smaller** than the tile grid (e.g., 214 junctions vs 600+ tiles on default map)

---

## Visualization

When `debug = True`, the topology is automatically visualized at game start with 4 maps:
1. **Junctions** - Shows decision points (J) and dead ends (D)
2. **Corridors** - Shows corridor IDs and junction connections (+)
3. **Articulation Points** - Shows critical choke points (A) vs regular junctions (.)
4. **Dead-End Zones** - Shows trap regions by zone ID, with exits marked (X)

---

## Quick Reference

| What You Want | How To Get It |
|---------------|---------------|
| Is this a junction? | `pos in topology.junctions` |
| What corridor am I in? | `corridor_id = topology.tile_to_corridor[pos]` |
| Get corridor endpoints | `topology.corridors[corridor_id].junction_a/b` |
| Is this a choke point? | `pos in topology.articulation_points` |
| Am I in a dead-end zone? | `pos in topology.dead_end_zones` |
| Where's the exit? | `exit = topology.dead_end_zones[pos]` |
| All junctions | `topology.junctions` (dict) |
| All articulation points | `topology.articulation_points` (set) |

---

## Tips for Low-Level Planners

1. **Pre-compute safe zones** during initialization based on current game state
2. **Use junction graph** for high-level pathfinding (much faster than tile-by-tile)
3. **Avoid dead-end zones** unless you've verified enemy distances
4. **Control articulation points** as a defender for maximum map coverage
5. **Plan escape routes** before entering corridors or dead-end zones
6. **Combine with belief tracking** to estimate enemy positions in unseen areas

---

Good luck with your planner! 🎮
