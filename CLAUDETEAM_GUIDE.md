# ClaudeTeam Implementation Guide

## Overview

This document describes the implementation of `claudeTeam.py` and `claudeTeam.pddl` for the Pacman Capture the Flag assignment (FIT5222). The system uses a **hierarchical planning architecture** combining PDDL-based high-level strategic planning with Q-learning for low-level action execution.

## Architecture

### Two-Level Planning System

```
┌─────────────────────────────────────────┐
│     High-Level PDDL Planning            │
│  (Strategic goals & action selection)   │
└──────────────┬──────────────────────────┘
               │ Outputs: attack, defence,
               │          go_home, etc.
               ▼
┌─────────────────────────────────────────┐
│     Low-Level Q-Learning                │
│  (Tactical movement: N/S/E/W/Stop)      │
└─────────────────────────────────────────┘
```

**High-Level (PDDL)**: Decides strategic goals based on game state (score, time, enemy positions)
- Output: High-level actions like "attack", "defence", "go_home_with_food", "chase_invader"
- Plans sequences of actions to achieve goals
- Replans when preconditions are violated or goals change

**Low-Level (Q-Learning)**: Executes tactical movements to fulfill high-level actions
- Output: Direction commands (North, South, East, West, Stop)
- Uses feature-based approximate Q-learning with hand-tuned weights
- Different reward functions and features for offensive, defensive, and escape scenarios

---

## File Structure

### claudeTeam.py

**Key Classes:**
- `MixedAgent(CaptureAgent)`: Main agent class implementing both PDDL planning and Q-learning

**Key Class Variables (Shared Between Agents):**
- `QLWeights`: Dictionary of weights for offensive, defensive, and escape Q-learning
- `CURRENT_ACTION`: Tracks what each agent is doing (for coordination)
- `ESTIMATED_POSITIONS`: Cached estimated enemy positions from belief tracking
- `OPPONENT_BELIEFS`: Belief distributions for unobserved enemies

**Key Methods:**

#### High-Level Planning
- `chooseAction(gameState)`: Entry point - selects high-level action, then low-level movement
- `getHighLevelPlan()`: Uses PDDL solver to generate action sequence
- `get_pddl_state()`: Converts game state to PDDL objects and predicates
- `getGoals()`: Determines strategic goals based on game state
- `stateSatisfyCurrentPlan()`: Checks if current plan is still valid

#### Goal Selection (Priority Order)
- `getGoals()`: **CRITICAL** - Checks if agent is Pacman first, prevents defensive goals for offensive agents
- `goalSecureWin()`: Endgame when winning - defend and run out clock
- `goalDesperateAttack()`: Endgame when losing - all-out attack
- `goalDefensive()`: Strong lead (>10 points) - focus on defense
- `goalBalanced()`: Moderate lead (5-10 points) - balance offense/defense
- `goalTimeAggressive()`: Low time remaining - push for points
- `goalAggressive()`: Standard play - attack while defending

#### Low-Level Execution
- `getLowLevelPlanQL()`: Generates movement using Q-learning
- `getQValue()`: Computes Q-value from features and weights
- `updateWeights()`: Updates weights during training (when `self.trainning = True`)
- `getValue()`: Gets max Q-value for next state

#### Feature Engineering
- `getOffensiveFeatures()`: 13 features for attacking (food distance, ghost proximity, etc.)
- `getDefensiveFeatures()`: 15 features for defending (invader tracking, food protection, etc.)
- `getEscapeFeatures()`: 12 features for returning home safely

#### Reward Functions
- `getOffensiveReward()`: Rewards eating food, returning food; penalizes getting caught
- `getDefensiveReward()`: Rewards catching invaders, protecting food
- `getEscapeReward()`: Rewards safe return home, penalizes getting caught

#### Belief Tracking Integration
- `updateEstimatedPositions()`: Caches enemy positions (exact or estimated via beliefs)
- Uses `belief_tracking.py` module for Bayesian inference on hidden enemies

---

### claudeTeam.pddl

**Domain:** `pacman_ctf_advanced`

**Requirements:** `:strips :typing :negative-preconditions`
- **NOTE**: Piglet PDDL parser does NOT support `:disjunctive-preconditions` (no `or` clauses)

**Types:**
```
enemy team - object
enemy1 enemy2 - enemy
ally current_agent - team
```

**Key Predicates:**

*Agent State:*
- `(is_pacman ?x)` - Agent is on enemy territory
- `(is_scared ?x)` - Agent is scared (capsule effect)
- `(food_in_backpack ?a)` - Agent carrying food
- `(3/5/10/20_food_in_backpack ?a)` - Food thresholds

*Environment:*
- `(food_available)` - Enemy food still exists
- `(capsule_available)` - Power capsule available
- `(near_food ?a)` - Food within 4 distance
- `(near_capsule ?a)` - Capsule within 4 distance

*Enemy Tracking:*
- `(enemy_around ?e ?a)` - Enemy within 4 distance
- `(enemy_carrying_food ?e)` - Enemy has food
- `(enemy_nearby_food ?e)` - Enemy threatening our food
- `(enemy_short/medium/long_distance ?e ?a)` - Distance bands

*Strategic:*
- `(safe_to_attack ?a)` - No nearby non-scared ghosts
- `(should_retreat ?a)` - Carrying food and threatened
- `(at_chokepoint ?a)` - At defensive chokepoint
- `(winning/winning_gt3/gt5/gt10/gt20)` - Score predicates
- `(low_time_remaining)` - <300 moves left
- `(very_low_time_remaining)` - <100 moves left

*Team Coordination:*
- `(near_ally)` - Ally within 4 distance
- `(ally_defending/attacking ?a)` - Ally's role
- `(more_enemies_around_ally)` - Ally needs support

**Actions (15 total):**

*Offensive (5):*
1. `attack` - Invade when safe
2. `aggressive_attack` - Attack when enemies scared
3. `collect_food_cluster` - Collect nearby food (when already Pacman)
4. `eat_capsule` - Get power capsule
5. `desperate_attack` - Endgame attack when losing

*Retreat (3):*
6. `go_home_with_food` - Return with food
7. `go_home_retreat` - Return when threatened
8. `emergency_retreat` - Emergency return with 5+ food

*Defensive (5):*
9. `defence` - Basic defense against invaders
10. `chase_invader` - Priority chase for food-carrying invaders
11. `chase_any_invader` - Chase any nearby invader
12. `patrol` - Patrol when winning
13. `defend_vulnerable_food` - Protect threatened food

*Support (2):*
14. `guard_chokepoint` - Hold strategic position
15. `support_ally` - Help ally under pressure
16. `secure_win` - Endgame defense when winning

---

## Known Issues & Solutions

### Issue 1: PDDL `(not (food_available))` Effect Problem

**Problem:** Initially added `(not (food_available))` to attack actions, causing agents to think all food was consumed after one attack action. This made the PDDL planner generate invalid plans (attack once, then immediately go home).

**Current Status:** Still present in lines 90, 107, 296 of claudeTeam.pddl

**Solution:** Remove `(not (food_available))` from attack action effects:
```pddl
(:action attack
    :effect (and
        (is_pacman ?a)
        ; DO NOT include: (not (food_available))
    )
)
```

The `food_available` predicate should only be removed when truly no food remains (handled by game state updates, not action effects).

### Issue 2: Pacman Agents Choosing Defensive Goals

**Problem:** Agents that were already Pacman (on enemy territory) would sometimes choose defensive goals based on score (e.g., `winning_gt10`), leading to impossible plans.

**Solution:** Implemented in `getGoals()` at line 652-672. **Always check if agent is Pacman FIRST** before strategic goal selection:
```python
def getGoals(self, objects, initState):
    myObj = f"a{self.index}"
    is_pacman = ("is_pacman", myObj) in initState

    if is_pacman:
        # Pacman should ONLY have offensive/retreat goals
        if should_retreat or has_food:
            return [goal to go home]
        else:
            return goalAggressive()  # Continue attacking

    # Only ghosts use strategic goal selection
    # ... rest of goal prioritization
```

### Issue 3: Ghost Distance Dominating Q-Values

**Problem:** The `closest-ghost-distance` feature with weight 50 was dominating Q-value calculations, causing agents to prioritize maximizing distance from ghosts even when they were far away (20+ steps). This prevented agents from pursuing food.

**Solution:** Replaced continuous distance feature with **threshold-based proximity features** (lines 1264-1269):
```python
if min_ghost_dist <= 2:
    features["ghost-very-close"] = 1.0  # -100 weight
elif min_ghost_dist <= 4:
    features["ghost-close"] = 1.0       # -30 weight
elif min_ghost_dist <= 6:
    features["ghost-nearby"] = 1.0      # -10 weight
# No feature if ghost is 7+ steps away - focus on food!
```

**Current Weights:**
- `closest-food`: -10 (negative = closer is better)
- `ghost-very-close`: -100 (1-2 steps = critical danger)
- `ghost-close`: -30 (3-4 steps = moderate danger)
- `ghost-nearby`: -10 (5-6 steps = minor concern)
- `#-of-ghosts-1-step-away`: -200 (never move next to ghost)

### Issue 4: Oscillating Movement (Back and Forth)

**Symptoms:** Agents moving North, then South, then North repeatedly without making progress.

**Root Cause:** Q-values for opposite directions were too similar (difference < 0.5), often due to:
1. Ghost distance dominating (see Issue 3)
2. `closest-food` weight too small
3. `reverse` penalty too weak

**Solution:**
- Increased `closest-food` weight: -1 → -10
- Increased action penalties: `stop` -10 → -20, `reverse` -2 → -10
- Fixed ghost distance features (threshold-based)

**Debug Tool:** Added Q-value logging at lines 889-904 to diagnose:
```python
if self.index == 0 and len(values) > 0:
    print(f"\nAgent {self.index} ({highLevelAction}) - Q-values:")
    for qval, act, feats in values:
        feat_breakdown = {k: (val, weight, contribution)
                         for k in feats if feats[k] != 0}
        print(f"  {act}: Q={qval} | {feat_breakdown}")
```

---

## Belief Tracking Integration

**Module:** `belief_tracking.py` (external)

**Purpose:** Estimate positions of unobserved enemies using Bayesian inference from noisy distance readings.

**Integration Points:**

1. **Initialization** (lines 186-192):
```python
from belief_tracking import initialize_beliefs, update_all_beliefs
MixedAgent.OPPONENT_BELIEFS = initialize_beliefs(gameState)
```

2. **Update** (lines 269-277):
```python
MixedAgent.OPPONENT_BELIEFS = update_all_beliefs(
    MixedAgent.OPPONENT_BELIEFS, gameState, self.index
)
```

3. **Position Estimation** (lines 221-257):
```python
def updateEstimatedPositions(self, gameState):
    """Cache estimated positions - O(num_enemies * w * h)"""
    for enemy_idx in self.getOpponents(gameState):
        if exact_pos is not None:
            ESTIMATED_POSITIONS[enemy_idx] = exact_pos
        else:
            # Use belief distribution
            belief_array = np.array(belief_grid)
            estimated_pos = argmax(belief_array)
            # Validate not a wall
            ESTIMATED_POSITIONS[enemy_idx] = estimated_pos
```

**Key Points:**
- Belief grids are 2D Python lists: `belief_grid[x][y]`
- **NOT** Grid objects - use `np.array()` for operations
- Cached once per turn in `ESTIMATED_POSITIONS` dict
- Used throughout for: ghost proximity, safe_to_attack, defensive features

---

## Q-Learning System

### Training Configuration

**Location:** Lines 167-172

```python
self.trainning = False  # SET TO TRUE for training
self.epsilon = 0.1      # Exploration rate (10% random)
self.alpha = 0.02       # Learning rate
self.discountRate = 0.9 # Discount factor
```

**Weight Persistence:**
- Weights saved to: `QLWeightsClaudeTeam.pkl` (pickle format)
- Loaded at start: lines 175-183
- Saved at end: lines 197-219 (only from agent 0 to avoid race condition)

### Training Process

**Command:**
```bash
./train_claudeTeam.sh -n 500  # Train for 500 games
```

**During Training:**
1. Agents explore with `epsilon` probability (10% random actions)
2. Weights updated via TD-learning: `w = w + α * (reward + γ*V(s') - Q(s,a)) * f`
3. Weights saved after each game
4. Next game loads previous weights and continues learning

**Testing (No Training):**
```bash
python capture.py -r claudeTeam.py -b berkeleyTeam.py -n 10
```

### Weight Categories

**Offensive Weights (13 features):**
- Primary driver: `closest-food` (-10)
- Ghost avoidance: `ghost-very-close` (-100), `ghost-close` (-30), `ghost-nearby` (-10)
- Food collection: `eats-food` (50), `food-density` (20)
- Safety: `#-of-ghosts-1-step-away` (-200)
- Action penalties: `stop` (-20), `reverse` (-10), `in-dead-end` (-50)

**Defensive Weights (15 features):**
- Invader tracking: `numInvaders` (-1000), `invaderDistance` (-10)
- Catching: `about-to-catch` (200)
- Food protection: `intercepting-food-threat` (100), `distance-to-threatened-food` (-20)
- Capsule protection: `protecting-capsule` (150), `capsule-under-threat` (-100)
- Positioning: `at-chokepoint` (20), `blocking-escape` (50)
- Action penalties: `stop` (-100), `reverse` (-2)

**Escape Weights (12 features):**
- Home priority: `onDefense` (1000), `distanceToHome` (-50)
- Progress: `moving-toward-home` (50)
- Ghost avoidance: `imminent-danger` (-200), `close-danger` (-100)
- Blocking: `ghost-blocking-home` (-150)
- Action penalties: `stop` (-100), `reverse` (-5)

### Feature Engineering Guidelines

**Good Features:**
- Normalized (divide by map width+height)
- Binary flags for clear states (e.g., `eats-food`, `at-chokepoint`)
- Threshold-based for proximity (NOT continuous distance)
- Use estimated positions from belief tracking

**Bad Features:**
- Continuous distances without normalization
- Features that always have large values (dominate Q-values)
- Features that don't differentiate between good/bad actions
- Ignoring hidden enemies (must use `ESTIMATED_POSITIONS`)

---

## Action Classification (Low-Level)

**Location:** Lines 843-866

High-level actions map to feature/reward functions:

```python
if highLevelAction == "attack":
    rewardFunction = getOffensiveReward
    featureFunction = getOffensiveFeatures
    weights = offensiveWeights

elif "retreat" in highLevelAction or "escape" in highLevelAction or "go_home" in highLevelAction:
    rewardFunction = getEscapeReward
    featureFunction = getEscapeFeatures
    weights = escapeWeights

else:  # All defensive actions (defence, chase_*, patrol, etc.)
    rewardFunction = getDefensiveReward
    featureFunction = getDefensiveFeatures
    weights = defensiveWeights
```

**Actions Using Defensive Features:**
- `defence`, `chase_invader`, `chase_any_invader`
- `patrol`, `guard_chokepoint`, `defend_vulnerable_food`
- `support_ally`, `secure_win`

---

## PDDL Planning Details

### Precondition Evaluation

**Location:** `lib_piglet` library (external)

**Key Insight:** PDDL planner checks preconditions against current state predicates. If no action's preconditions are satisfied, it returns an empty plan.

**Common Empty Plan Causes:**
1. Goal is already satisfied (no work needed)
2. No action can progress toward goal from current state
3. Invalid goal (impossible to achieve)

**Debug Output:** Lines 292-317 shows:
```
Agent 0 replanning:
  Positive Goal: [...]
  Negative Goal: [...]
  Key Predicates: [...]
  Plan: [attack, go_home_with_food]
```

### Replanning Triggers

**Location:** Lines 290-317

Agent replans when `stateSatisfyCurrentPlan()` returns False:

1. **No current plan** or plan is empty
2. **Goals changed** (score changed, time threshold crossed)
3. **Current action precondition violated** (e.g., became scared, enemy moved away)
4. **Current action effect achieved** and next action not applicable

**Important:** Replanning happens EVERY turn unless plan explicitly valid. Keep plans short!

---

## Common Debugging Workflows

### Issue: Agents Not Attacking

**Check:**
1. Is `safe_to_attack` predicate set? (Line 558-559)
2. Are ghosts nearby? Check `ESTIMATED_POSITIONS` and ghost distances
3. Is goal including `(not (food_available))`? (Lines 770, 714, 755)
4. Is PDDL plan empty? Check debug output

**Fix:** Adjust `safe_to_attack` logic (lines 544-559) or MEDIUM_DISTANCE threshold

### Issue: Agents Not Returning Home

**Check:**
1. Is agent Pacman and carrying food? (Predicate: `food_in_backpack`)
2. Is `should_retreat` set? (Line 576-577)
3. Is goal `(not (is_pacman))` being generated? (Lines 665-669)
4. Does PDDL have `go_home_with_food` or `go_home_retreat` action?

**Fix:** Verify goal generation in `getGoals()` when Pacman (lines 658-672)

### Issue: Only Choosing Defense

**Check:**
1. Is agent a Pacman but getting defensive goal? (Bug in `getGoals`)
2. Is `safe_to_attack` never true? (Ghosts too close)
3. Is PDDL plan empty, falling back to default "defence"? (Line 319)

**Fix:** Ensure `getGoals()` checks `is_pacman` FIRST (line 654)

### Issue: Q-Values Too Similar

**Check:**
1. Enable debug logging (lines 889-904)
2. Look at feature contributions: `(value, weight, contribution)`
3. Which features dominate? Which are too small?

**Fix:** Adjust weights - dominant features need reduction, weak features need increase

---

## Performance Optimization

### Computational Costs

**Expensive Operations (per turn):**
1. Belief tracking update: O(num_enemies × width × height) ≈ 2 × 32 × 16 = 1024 ops
2. PDDL planning: O(num_actions × depth) ≈ 15 × 3 = 45 ops (cached)
3. Position estimation: O(num_enemies × w × h) ≈ 1024 ops (cached once)
4. Q-learning: O(num_actions × num_features) ≈ 5 × 13 = 65 ops

**Critical Caching:**
- `ESTIMATED_POSITIONS`: Computed ONCE per turn (line 280), reused everywhere
- PDDL plans: Only recompute when `stateSatisfyCurrentPlan()` fails
- Maze distances: Automatically cached by game engine

**DO NOT:**
- Call `getMazeDistance()` in loops without caching
- Recompute belief tracking multiple times per turn
- Recalculate estimated positions in feature functions

---

## Future Improvements

### High Priority

1. **Fix PDDL `(not (food_available))` bug**
   - Remove from attack action effects (lines 90, 107, 296)
   - Test that agents continue attacking after first food collection

2. **Improve Goal Generation**
   - Consider agent carrying amount in goal selection
   - Add "risky attack" goal when losing by a lot
   - Better team coordination (one attack, one defend)

3. **Enhance Defensive Features**
   - Add "invader escape path" feature (path to their home)
   - Add "teammate distance to invader" (avoid both chasing same invader)
   - Add "capsule grab timing" (when invader about to get capsule)

### Medium Priority

4. **Better Escape Planning**
   - Pathfinding that avoids ghost-controlled areas
   - Alternative routes when ghosts block main path
   - "Bait and switch" - fake one direction, go another

5. **Dynamic Weight Adjustment**
   - Reduce epsilon over training: start 0.3, decay to 0.05
   - Increase food-carrying urgency based on time remaining
   - Adjust aggression based on score differential

6. **PDDL Action Refinement**
   - Add "fake attack" (cross border, then return to bait defender)
   - Add "capsule timing" (wait near capsule until needed)
   - Add "pincer movement" (coordinate with ally to trap enemy)

### Low Priority

7. **Advanced Belief Tracking**
   - Use enemy action patterns (offensive vs defensive agents)
   - Predict enemy goals from their movement
   - Communicate beliefs between teammates

8. **Meta-Learning**
   - Detect opponent strategy (aggressive, defensive, balanced)
   - Adapt strategy to counter opponent
   - Learn opponent-specific weights

---

## Testing & Evaluation

### Quick Tests

**Functionality:**
```bash
# Test against baseline
python capture.py -r claudeTeam.py -b baselineTeam.py -n 5

# Test against Berkeley
python capture.py -r claudeTeam.py -b berkeleyTeam.py -n 5

# Test different layouts
python capture.py -r claudeTeam.py -b berkeleyTeam.py -l mediumCapture -n 3
```

**Debug Modes:**
```bash
# With Q-value logging (agent 0)
python capture.py -r claudeTeam.py -b berkeleyTeam.py -n 1 | grep "Q-values" -A 10

# With PDDL planning logging
python capture.py -r claudeTeam.py -b berkeleyTeam.py -n 1 | grep "replanning" -A 5

# Quiet mode (fast)
python capture.py -r claudeTeam.py -b berkeleyTeam.py -n 10 -q
```

### Training Evaluation

**Baseline (before training):**
```bash
# Save original weights
cp QLWeightsClaudeTeam.pkl QLWeightsClaudeTeam_backup.pkl

# Test win rate
python capture.py -r claudeTeam.py -b berkeleyTeam.py -n 20 -q | grep "wins"
```

**After Training:**
```bash
# Train
./train_claudeTeam.sh -n 500

# Test win rate
python capture.py -r claudeTeam.py -b berkeleyTeam.py -n 20 -q | grep "wins"

# Compare
echo "Improvement: [new wins - old wins] wins"
```

### Rubric Alignment (FIT5222)

**Key Evaluation Criteria:**
1. **PDDL Domain Design (20%)**: Comprehensive actions, predicates, goal structure
2. **Q-Learning Implementation (20%)**: Feature engineering, reward functions, convergence
3. **Performance (30%)**: Win rate against baseline/Berkeley teams
4. **Code Quality (15%)**: Documentation, architecture, maintainability
5. **Report (15%)**: Analysis, justification, experimentation results

**Target Performance:**
- HD (80-100%): Win >70% against Berkeley, sophisticated strategy, learning convergence
- D (70-79%): Win >50%, reasonable strategy, basic learning
- C (60-69%): Win >30%, functional but simple

---

## Key Constants

```python
# Distance thresholds
CLOSE_DISTANCE = 4      # "Near" predicates
MEDIUM_DISTANCE = 15    # Strategic planning range
LONG_DISTANCE = 25      # Far away, low priority

# Files
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
PDDL_FILE = BASE_FOLDER + '/claudeTeam.pddl'
WEIGHTS_FILE = BASE_FOLDER + '/QLWeightsClaudeTeam.pkl'
```

---

## Emergency Reset

**If agents are completely broken:**

```bash
# 1. Delete learned weights
rm QLWeightsClaudeTeam.pkl

# 2. Restore defaults in code (lines 89-140)
# Verify weights are hand-tuned values

# 3. Disable training
# Line 167: self.trainning = False

# 4. Test
python capture.py -r claudeTeam.py -b berkeleyTeam.py -n 3

# 5. If still broken, check:
# - PDDL syntax (no 'or' clauses)
# - getGoals() checks is_pacman first
# - Ghost features are threshold-based
```

---

## References

**Course Materials:**
- PDDL Planning: lib_piglet documentation
- Q-Learning: Lecture notes on approximate Q-learning
- Belief Tracking: belief_tracking.py module

**External Resources:**
- PDDL syntax: http://planning.wiki/
- Pacman CTF: UC Berkeley AI course materials
- Q-Learning: Sutton & Barto, Reinforcement Learning textbook

---

## Contact & Version

**Last Updated:** 2025-10-30
**Status:** Functional with known issues (see Issue 1)
**Next Steps:** Fix PDDL food_available bug, improve defensive coordination

**For Questions:**
- Check CLAUDE.md for basic command usage
- Check this guide for architecture understanding
- Check code comments for implementation details
