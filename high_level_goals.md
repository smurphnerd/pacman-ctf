# Attacking based actions

How to make the high level actions follow each other: go enemy land -> eat food -> go home

## Go enemy land

- Precondition:
  - Not winning
  - Teammate not attacking mode
  - not isPacman
  - Food available
- Effect:
  - isPacman

## Eat food

- Precondition:
  - isPacman
  - Not winning
  - Food available
- Effect:
  - Food in backpack

## Go home

- Precondition:
  - isPacman
  - isWinning or foodInBackpack is enough to win or enemyClose # We go back if we can safely take the lead, or if we are in danger of being eaten
- Effect:
  - Not isPacman

# Defending based actions

## Patrol (Should just try to prevent the opponent from coming to our land)

- Precondition:
  - not isPacman
  - isWinning or teammate is attacking
  - not enemies isPacman
- Effect:
  - Basically get closer to the enemy
  - not isPacman (do not cross the line)

## Defend (try to eat the attacker)

- Precondition:
  - Enemy isPacman
  - not isPacman
- Effect:
  - not enemy isPacman
  - distance to enemy shortens

# States

Neutral (both teams are ghosts) (high level goals: a1 decoy, a2 take lead)
Race (e1 e2 both attacking, a1 attacks, a2 defends) (high level goals: defend territory, take lead)
Default (e1 defends, e2 attacks, a1 attacks, a2 defends) (high level goals: take lead, defend territory/defend border)
Heavy attack (e1 e2 both defending, a1 attacks, a2 attacks) (high level goals: take lead, decoy)
Heavy defense (a1 defends, a2 defends) (high level goals: mirror, defend territory, defend border)

# Preconditions

Winning
near enemy (within 4 grid distance)

# High level goals

## Decoy

- Precon: not winning
- Cross the border or threaten to cross the border to draw enemies closer
- Threaten should be done by the teammate closer to an enemy
- The other teammate who is further from an enemy should do take lead

## Take lead

- Precon: not winning, not near enemy
- Eat just enough food to take the lead (food in backpack > -score)
- Move away from defending enemies (don't care about moving closer to attacking enemies as they can't eat us)
- Then get back home with the lead

## Defend territory

- Be in between the enemy and our side (try to position self in middle of nearest food/pellet and enemy, and minimize distance to enemy)

## Defend border

- Be in between the enemy and enemy side (try to position self in middle of border and enemy, and minimize distance to enemy)
