# myTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# myTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from ast import Raise
from typing import List, Tuple, Dict

from numpy import true_divide
import numpy as np
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os, pickle, base64
from capture import GameState, noisyDistance
from game import Configuration, Directions, Actions, AgentState, Agent
from util import nearestPoint, manhattanDistance
import sys, os
import pickle
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class StateInfo:
    gameState: GameState
    agentState: AgentState
    teammateState: AgentState
    enemyVirtualStates: Dict[int, AgentState]


@dataclass
class Junction:
    """Represents a strategic node (junction or dead-end) in the map"""

    pos: Tuple[int, int]
    neighbors: List[Tuple[int, int]]  # Number of non-wall neighbors
    junction_type: str  # "junction" (3-4 neighbors), "dead_end" (1 neighbor)
    connected_junctions: Dict[Tuple[int, int], int]  # {junction_pos: corridor_length}


@dataclass
class Corridor:
    """Represents a corridor (edge) connecting two junctions"""

    corridor_id: int
    junction_a: Tuple[int, int]
    junction_b: Tuple[int, int]
    length: int
    path: List[Tuple[int, int]]  # All tiles in the corridor


@dataclass
class MapTopology:
    """Preprocessed topological map data"""

    junctions: Dict[Tuple[int, int], Junction]  # {pos: Junction}
    corridors: Dict[int, Corridor]  # {corridor_id: Corridor}
    tile_to_corridor: Dict[Tuple[int, int], int]  # {pos: corridor_id}
    articulation_points: set  # Set of junction positions that are cut vertices
    dead_end_zones: Dict[Tuple[int, int], Tuple[int, int]]  # {tile_pos: exit_junction}


try:
    profile
except NameError:

    def profile(func):
        return func


# the folder of current file.
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

from lib_piglet.utils.pddl_solver import pddl_solver
from lib_piglet.domains.pddl import pddl_state
from lib_piglet.utils.pddl_parser import Action

DEATH_DISTANCE = 1
BREATHING_DISTANCE = 2
CLOSE_DISTANCE = 4
MEDIUM_DISTANCE = 15
LONG_DISTANCE = 25

LOW_SCARED_TIMER = 10

# Score threshold parameters (as percentage of starting food)
SMALL_LEAD_PERCENTAGE = 0.10  # 10% of food for "winning_gt3" equivalent
LARGE_LEAD_PERCENTAGE = 0.40  # 40% of food for "winning_gt10" equivalent

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed, first="MixedAgent", second="MixedAgent"):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########


class MixedAgent(CaptureAgent):
    """
    This is an agent that use pddl to guide the high level actions of Pacman
    """

    # Default weights for q learning, if no QLWeights.txt find, we use the following weights.
    # You should add your weights for new low level planner here as well.
    # weights are defined as class attribute here, so taht agents share same weights.
    QLWeights = {
        "attackWeights": {
            "got-eaten": -1000000,  # Priority -1 (less better) - CRITICAL: got eaten
            "breathing-distance-ghosts": -10000,  # Priority 1 (less better) - CRITICAL: avoid ghosts
            "in-enemy-territory": 1000,  # Priority 2 (more better) - main goal
            "close-distance-ghosts": -500,  # Priority 2 (less better) - CRITICAL: avoid ghosts
            "distance-to-enemy-territory": -100,  # Priority 3 (closer better) - guide to border
            "distance-to-teammate": 10,  # Priority 4 (further better) - spread out
            "stop-reverse": -5,
        },
        "eatFoodWeights": {
            "got-eaten": -1000000,  # Priority -1 (less better) - CRITICAL: got eaten
            "breathing-distance-ghosts": -100000,  # Priority 0 (less better) - CRITICAL: avoid ghosts
            # "in-enemy-territory": 10000,  # Priority 1 (more better) - MUST be in enemy territory
            "ate-food": 5000,  # Priority 2 - big reward for eating
            "distance-to-nearest-food": -100,  # Priority 3 (closer better) - guide to food
            "close-distance-ghosts": -10,  # Priority 4 (less better) - minor ghost avoidance
            "distance-to-attacking-teammate": 90,  # Priority 5 (closer better) - guide to attacking teammate
            "stop-reverse": -5,
        },
        "eatCapsuleWeights": {
            "got-eaten": -1000000,  # Priority -1 (less better) - CRITICAL: got eaten
            "breathing-distance-ghosts": -10000,  # Priority 1 (less better) - CRITICAL: avoid ghosts
            # "in-enemy-territory": 100000,  # Priority 0 (more better) - MUST be in enemy territory
            "ate-capsule": 100000,  # Priority 2 - big reward for eating capsule
            "distance-to-nearest-capsule": -100,  # Priority 3 (closer better) - guide to capsule
            "close-distance-ghosts": -10,  # Priority 4 (less better) - minor ghost avoidance
            "stop-reverse": -5,
        },
        "escapeWeights": {
            "got-eaten": -1000000,  # Priority -1 (less better) - CRITICAL: got eaten
            "in-home": 10000,  # Priority 1 - CRITICAL: get home safely
            "breathing-distance-ghosts": -1000,  # Priority 2 (less better) - avoid immediate danger
            "distance-to-home": -1000,  # Priority 2 - get home quickly
            "close-distance-ghosts": -100,  # Priority 3 (less better) - avoid nearby ghosts
            "stop-reverse": -5,
        },
        "chaseWeights": {
            "got-eaten": -1000000,  # Priority -1 (less better) - CRITICAL: got eaten
            "ate_enemy": 1000000,
            # "in-home": 100000,  # Priority 1 - MUST stay in home territory
            "distance-to-enemy": -1000,  # Priority 2 (closer better) - get to target
            "between-enemy-and-escape": 100,  # Priority 3 - intercept bonus
            "distance-to-teammate": 10,  # Priority 4 (further better) - spread out
            "stop-reverse": -5,
        },
        "defaultDefendWeights": {
            "got-eaten": -1000000,  # Priority -1 (less better) - CRITICAL: got eaten
            "in-home": 100000,  # Priority 1 - MUST stay in home territory
            "distance-to-nearest-enemy": -1000,  # Priority 2 (closer better) - position near threats
            "distance-to-teammate": 10,  # Priority 3 (further better) - spread out
            "distance-to-teammate-both-defending": 900,  # Priority 4 (further better) - spread out
            # "stop-reverse": -5,
        },
    }

    # Also can use class variable to exchange information between agents.
    CURRENT_ACTION = {}
    ESTIMATED_POSITIONS = {}  # Cache for estimated enemy positions using beliefs
    CONSECUTIVE_STOP_REVERSE = {}  # Tracks consecutive stop/reverse moves per agent
    DEFENSIVE_ASSIGNMENTS = defaultdict(int)
    NUM_GAMES = 0
    MAP_TOPOLOGY: MapTopology = None  # Cached topological map analysis
    CURRENT_LAYOUT_STR = (
        None  # String representation of current layout for cache invalidation
    )

    def registerInitialState(self, gameState: GameState):
        self.pddl_solver = pddl_solver(BASE_FOLDER + "/myTeam.pddl")
        self.highLevelPlan: List[
            Tuple[Action, pddl_state]
        ] = None  # Plan is a list Action and pddl_state
        self.currentNegativeGoalStates = []
        self.currentPositiveGoalStates = []
        self.currentActionIndex = (
            0  # index of action in self.highLevelPlan should be execute next
        )

        self.startPosition = gameState.getAgentPosition(
            self.index
        )  # the start location of the agent
        CaptureAgent.registerInitialState(self, gameState)

        self.lowLevelPlan: List[Tuple[str, Tuple]] = []
        self.lowLevelActionIndex = 0

        # Initialize consecutive stop/reverse counter for this agent
        MixedAgent.CONSECUTIVE_STOP_REVERSE[self.index] = 0

        # Create a defensive distance calculator that treats enemy territory as walls
        # This helps defensive agents find optimal patrol positions at the border
        self.defensiveDistancer = self.createDefensiveDistancer(gameState)
        self.defensiveDistancer.getMazeDistances()

        self.debug = True

        # Calculate total starting food and thresholds (once per game)
        red_food = gameState.getRedFood().count()

        # Calculate thresholds based on percentages
        self.small_lead_threshold = int(red_food * SMALL_LEAD_PERCENTAGE)
        self.large_lead_threshold = int(red_food * LARGE_LEAD_PERCENTAGE)

        if self.debug:
            print(f"Total starting food: {red_food}")
            print(
                f"Small lead threshold (winning_gt3): {self.small_lead_threshold} points"
            )
            print(
                f"Large lead threshold (winning_gt10): {self.large_lead_threshold} points"
            )

        # REMEMBER TRUN TRAINNING TO FALSE when submit to contest server.
        self.epsilon = 0.1  # default exploration prob, change to take a random step
        self.alpha = 0.02  # default learning rate
        self.discountRate = (
            0.9  # default discount rate on successor state q value when update
        )

        # Initialize belief tracking for opponents
        MixedAgent.OPPONENT_BELIEFS = initialize_beliefs(gameState)

        # Build topological map (only once per layout)
        # Check if layout has changed or topology not yet built
        layout_str = str(gameState.data.layout)

        if MixedAgent.CURRENT_LAYOUT_STR != layout_str:
            # New layout detected - rebuild topology
            walls = gameState.getWalls()
            MixedAgent.MAP_TOPOLOGY = build_map_topology(walls)
            MixedAgent.CURRENT_LAYOUT_STR = layout_str

            if self.debug:
                print(f"\nAgent {self.index}: Built map topology for new layout")
                visualize_topology(MixedAgent.MAP_TOPOLOGY, walls)

        # Use a dictionary to save information about current agent.
        MixedAgent.CURRENT_ACTION[self.index] = {}

    def final(self, gameState: GameState):
        """
        This function write weights into files after the game is over.
        You may want to comment (disallow) this function when submit to contest server.
        """

    def updateEstimatedPositions(self, gameState: GameState):
        """
        Cache estimated enemy positions using belief tracking.
        Called once per turn for efficiency. O(num_enemies * w * h)
        """

        walls = gameState.getWalls()

        for enemy_idx in self.getOpponents(gameState):
            enemy_state = gameState.getAgentState(enemy_idx)
            exact_pos = enemy_state.getPosition()

            if exact_pos is not None:
                # Enemy is observable - use exact position
                MixedAgent.ESTIMATED_POSITIONS[enemy_idx] = exact_pos
            elif enemy_idx in MixedAgent.OPPONENT_BELIEFS:
                # Enemy not observable - estimate using belief distribution
                belief_grid = MixedAgent.OPPONENT_BELIEFS[enemy_idx]
                # belief_grid is a 2D list: belief_grid[x][y]
                # Convert to numpy array and find max probability position
                belief_array = np.array(belief_grid)
                max_idx = np.unravel_index(np.argmax(belief_array), belief_array.shape)
                estimated_pos = (max_idx[0], max_idx[1])

                # Validate position is not a wall
                if not walls[int(estimated_pos[0])][int(estimated_pos[1])]:
                    MixedAgent.ESTIMATED_POSITIONS[enemy_idx] = estimated_pos
                else:
                    # Position is a wall, use start position as fallback
                    MixedAgent.ESTIMATED_POSITIONS[
                        enemy_idx
                    ] = gameState.getInitialAgentPosition(enemy_idx)
            else:
                # Fallback: no position estimate available, use start position
                MixedAgent.ESTIMATED_POSITIONS[
                    enemy_idx
                ] = gameState.getInitialAgentPosition(enemy_idx)

    @profile
    def chooseAction(self, gameState: GameState):
        """
        This is the action entry point for the agent.
        In the game, this function is called when its current agent's turn to move.

        We first pick a high-level action.
        Then generate low-level action (up down left right wait) to achieve the high-level action.
        """

        # Update belief tracking for opponents

        MixedAgent.OPPONENT_BELIEFS = update_all_beliefs(
            MixedAgent.OPPONENT_BELIEFS, gameState, self.index
        )

        # Cache estimated enemy positions for this turn (computed once, reused many times)
        self.updateEstimatedPositions(gameState)

        # -------------High Level Plan Section-------------------
        # Get high level action from a pddl plan.

        # Collect objects and init states from gameState
        objects, initState = self.get_pddl_state(gameState)
        positiveGoal, negtiveGoal = self.getGoals(objects, initState)

        # Check if we can stick to current plan
        if not self.stateSatisfyCurrentPlan(initState, positiveGoal, negtiveGoal):
            # Cannot stick to current plan, prepare goals and replan
            if self.debug:
                print(f"Agent {self.index} replanning:")
                print(f"  Positive Goal: {positiveGoal}")
                print(f"  Negative Goal: {negtiveGoal}")
            self.highLevelPlan: List[Tuple[Action, pddl_state]] = self.getHighLevelPlan(
                objects, initState, positiveGoal, negtiveGoal
            )  # Plan is a list Action and pddl_state
            self.currentActionIndex = 0
            self.lowLevelPlan = []  # reset low level plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            if self.debug:
                print(f"  Plan: {[action.name for action, _ in self.highLevelPlan]}")

        if not self.highLevelPlan:
            if self.debug:
                print(f"No plan found for predicates: {initState}")
            highLevelAction = Action("default_defend", None, [], [], [], [])
        else:
            # Get next action from the plan
            highLevelAction = self.highLevelPlan[self.currentActionIndex][0]
        MixedAgent.CURRENT_ACTION[self.index] = highLevelAction

        if self.debug:
            print(f"Agent {self.index}: High-Level Action = {highLevelAction.name}")

        if highLevelAction.name != "default_defend":
            MixedAgent.DEFENSIVE_ASSIGNMENTS[self.index] = -1

        # -------------Low Level Plan Section-------------------
        # Get the low level plan using Q learning, and return a low level action at last.
        # A low level action is defined in Directions, whihc include {"North", "South", "East", "West", "Stop"}

        if not self.posSatisfyLowLevelPlan(gameState):
            self.lowLevelPlan = self.getLowLevelPlanQL(
                gameState, highLevelAction.name
            )  # Generate low level plan with q learning
            # you can replace the getLowLevelPlanQL with getLowLevelPlanHS and implement heuristic search planner
            self.lowLevelActionIndex = 0
        lowLevelAction = self.lowLevelPlan[self.lowLevelActionIndex][0]
        self.lowLevelActionIndex += 1

        # Update consecutive stop/reverse counter
        agentState = gameState.getAgentState(self.index)
        is_stop = lowLevelAction == Directions.STOP
        is_reverse = (
            lowLevelAction == Directions.REVERSE[agentState.configuration.direction]
        )

        if is_stop or is_reverse:
            # Increment counter for consecutive stop/reverse
            MixedAgent.CONSECUTIVE_STOP_REVERSE[self.index] += 1
        else:
            # Reset counter - agent is making a productive move
            MixedAgent.CONSECUTIVE_STOP_REVERSE[self.index] = 0

        # print("\tAgent:", self.index,lowLevelAction)
        return lowLevelAction

    # ------------------------------- PDDL and High-Level Action Functions -------------------------------

    @profile
    def getHighLevelPlan(
        self, objects, initState, positiveGoal, negtiveGoal
    ) -> List[Tuple[Action, pddl_state]]:
        """
        This function prepare the pddl problem, solve it and return pddl plan
        """
        # Prepare pddl problem
        self.pddl_solver.parser_.reset_problem()
        self.pddl_solver.parser_.set_objects(objects)
        self.pddl_solver.parser_.set_state(initState)
        self.pddl_solver.parser_.set_negative_goals(negtiveGoal)
        self.pddl_solver.parser_.set_positive_goals(positiveGoal)

        # Solve the problem and return the plan
        return self.pddl_solver.solve()

    @profile
    def get_pddl_state(self, gameState: GameState) -> Tuple[List[Tuple], List[Tuple]]:
        """
        This function collects pddl :objects and :init states from simulator gameState.
        """
        # Collect objects and states from the gameState

        states = []
        objects = []

        # Collect available foods on the map
        myPos = gameState.getAgentPosition(self.index)
        myObj = f"a{self.index}"
        opponents = self.getOpponents(gameState)

        # Collect capsule states
        capsules = self.getCapsules(gameState)
        for cap in capsules:
            my_distance = self.getMazeDistance(cap, myPos)
            opp_closer = False
            for opp_idx in opponents:
                opp_pos = MixedAgent.ESTIMATED_POSITIONS.get(opp_idx)
                assert opp_pos
                if self.getMazeDistance(cap, opp_pos) < my_distance:
                    opp_closer = True
                    break
            if not opp_closer:
                states.append(("closest_to_capsule",))
                break

        # Collect winning states
        currentScore = gameState.data.score
        team_score = currentScore if self.red else -currentScore
        opponent_score = -team_score
        if team_score > 0:
            states.append(("winning",))
        if team_score > self.small_lead_threshold:
            states.append(("winning_gt3",))
        if team_score < -self.small_lead_threshold:
            states.append(("losing_gt3",))

        # Time remaining predicates
        if hasattr(gameState.data, "timeleft"):
            if gameState.data.timeleft < 300:
                states.append(("low_time_remaining",))
            if gameState.data.timeleft < 100:
                states.append(("very_low_time_remaining",))

        def add_general_agent_states(agent_state, agent_obj, positive_score):
            if agent_state.scaredTimer > 0:
                states.append(("is_scared", agent_obj))

            if agent_state.numCarrying + positive_score > 0:
                states.append(("fat_agent", agent_obj))
            if agent_state.numCarrying + positive_score > self.small_lead_threshold:
                states.append(("fat_agent_gt3", agent_obj))
            if agent_state.numCarrying + positive_score > self.large_lead_threshold:
                states.append(("fat_agent_gt10", agent_obj))

            if agent_state.isPacman:
                states.append(("is_pacman", agent_obj))

        # Check if there is any food nearby
        food = self.getFood(gameState).asList()
        if any(self.getMazeDistance(f, myPos) <= CLOSE_DISTANCE for f in food):
            states.append(("near_food", myObj))

        # Check if there is any food available on enemy side
        if len(food) > 0:
            states.append(("food_available",))

        # Collect team agents states
        agents: List[Tuple[int, AgentState]] = [
            (i, gameState.getAgentState(i)) for i in self.getTeam(gameState)
        ]
        for agent_index, agent_state in agents:
            agent_object = f"a{agent_index}"
            agent_type = "current_agent" if agent_index == self.index else "ally"
            objects += [(agent_object, agent_type)]

            add_general_agent_states(agent_state, agent_object, team_score)

            # Ally coordination predicates
            if agent_index != self.index and MixedAgent.CURRENT_ACTION.get(agent_index):
                action = MixedAgent.CURRENT_ACTION.get(agent_index)
                assert hasattr(action, "name") and hasattr(action, "parameters")
                if action.name == "chase_enemy":
                    assert len(action.parameters) == 2
                    states.append(("ally_chasing", action.parameters[0]))
                if action.name == "default_defend":
                    states.append(("ally_defending",))

            pos = agent_state.getPosition()
            assert pos

            # Check if we are further back
            if (self.red and myPos[0] < pos[0]) or (not self.red and myPos[0] > pos[0]):
                states.append(("further_back",))

        my_closest_enemy = float("inf")
        teammate_closest_enemy = float("inf")

        # Collect enemy agents states
        enemies: List[Tuple[int, AgentState]] = [
            (i, gameState.getAgentState(i)) for i in self.getOpponents(gameState)
        ]
        noisyDistance = gameState.getAgentDistances()
        typeIndex = 1
        for enemy_index, enemy_state in enemies:
            enemy_object = f"e{enemy_index}"
            objects += [(enemy_object, "enemy{}".format(typeIndex))]

            add_general_agent_states(enemy_state, enemy_object, opponent_score)

            est_pos = MixedAgent.ESTIMATED_POSITIONS.get(enemy_index)
            if enemy_state.scaredTimer <= LOW_SCARED_TIMER:
                distance = self.getMazeDistance(est_pos, myPos)
                if distance <= BREATHING_DISTANCE:
                    states.append(("enemy_breathing_distance",))
                if distance <= CLOSE_DISTANCE:
                    states.append(("enemy_close_distance",))
                if distance <= MEDIUM_DISTANCE:
                    states.append(("enemy_medium_distance",))

            # Check if enemy is carrying food (always visible in AgentState)
            if enemy_state.numCarrying > 0:
                states.append(("enemy_carrying_food", enemy_object))

            # Use cached estimated position (computed once at start of chooseAction)
            estimated_pos = MixedAgent.ESTIMATED_POSITIONS.get(enemy_index)
            assert estimated_pos
            if (self.red and estimated_pos[0] <= myPos[0]) or (
                not self.red and estimated_pos[0] >= myPos[0]
            ):
                states.append(("enemy_past", enemy_object))

            # Determine if current agent is closer to this enemy than teammate
            my_distance = self.getMazeDistance(est_pos, myPos)
            teammate_pos = gameState.getAgentPosition((self.index + 2) % 4)
            teammate_distance = self.getMazeDistance(est_pos, teammate_pos)

            if my_distance <= teammate_distance:
                states.append(("closer_to_enemy", enemy_object))

            if my_distance < my_closest_enemy:
                my_closest_enemy = my_distance
            if teammate_distance < teammate_closest_enemy:
                teammate_closest_enemy = teammate_distance

            typeIndex += 1

        # Check if we're closer to our closest enemy than teammate
        if my_closest_enemy <= teammate_closest_enemy:
            states.append(("closer_to_closest_enemy",))

        return objects, states

    @profile
    def stateSatisfyCurrentPlan(
        self, init_state: List[Tuple], positiveGoal, negtiveGoal
    ):
        if self.highLevelPlan is None or len(self.highLevelPlan) == 0:
            # No plan, need a new plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            return False

        if (
            positiveGoal != self.currentPositiveGoalStates
            or negtiveGoal != self.currentNegativeGoalStates
        ):
            return False

        if self.pddl_solver.matchEffect(
            init_state, self.highLevelPlan[self.currentActionIndex][0]
        ):
            # The current state match the effect of current action, current action action done, move to next action
            if self.currentActionIndex < len(
                self.highLevelPlan
            ) - 1 and self.pddl_solver.satisfyPrecondition(
                init_state, self.highLevelPlan[self.currentActionIndex + 1][0]
            ):
                # Current action finished and next action is applicable
                self.currentActionIndex += 1
                self.lowLevelPlan = []  # reset low level plan
                return True
            else:
                # Current action finished, next action is not applicable or finish last action in the plan
                return False

        if self.pddl_solver.satisfyPrecondition(
            init_state, self.highLevelPlan[self.currentActionIndex][0]
        ):
            # Current action precondition satisfied, continue executing current action of the plan
            return True

        # Current action precondition not satisfied anymore, need new plan
        return False

    @profile
    def getGoals(self, objects: List[Tuple], initState: List[Tuple]):
        """
        Multi-tiered goal prioritization system based on game state.
        Returns positive and negative goal states for PDDL planning.
        """
        # Fat agent is one that has enough food to take the lead if they cross back

        # If we are ghost these are the orders of priority:
        # 1. If scared, aggressive attack until the timer runs out (attack)
        # 2. If there's a fat enemy and we are the closest ghost to it, chase it (the effect in pddl should be "chasing e") (chase e)
        # 3. If there's a fat enemy and our teammate isn't defending it, chase it (the effect in pddl should be "chasing e") (chase e)
        # 4. If an enemy is in our territory and we are the closest ghost to it, chase it (the effect in pddl should be "chasing e") (chase e)
        # 5. If an enemy is in our territory and our teammate isn't defending it, chase it (the effect in pddl should be "chasing e") (chase e)
        # 6. If we are winning by >3 points, defend default (default)
        # 6. (There should be no unchased enemies by now) If our teammate is chasing an enemy, then attack (attack)
        # 7. (There should be no enemies) If our teammate is not attacking and we are closer to the enemy territory, then attack (attack)
        # 8. If we aren't winning, and our teammate is also not attacking, then attack (attack)
        # 9. If we are losing by >3 points, attack (attack)
        # 10. We aren't losing by too much so just defend (default)

        # If we are pacman these are the orders of priority:
        # 1. If we are the closest out of all enemies to a capsule eat it (eat capsule)
        # 2. If we don't have enough food to take the lead, keep eating (eat food)
        # 3. If we have enough food to take the lead and there is no food nearby, escape (escape)
        # 3. If we have enough to take the lead by >10 points escape (escape)
        # 4. If we aren't under threat and timer is still high, keep eating (no non-scared ghosts in vacinity) (eat food)
        # 5. We have enough to take the lead now, just escape

        myObj = f"a{self.index}"
        is_pacman = ("is_pacman", myObj) in initState
        is_scared = ("is_scared", myObj) in initState

        # Get enemy objects for chase goals
        enemy_objects = [obj[0] for obj in objects if obj[1] in ["enemy1", "enemy2"]]
        # Check if teammate is also pacman
        teammateObj = f"a{(self.index + 2) % 4}"
        teammate_is_pacman = ("is_pacman", teammateObj) in initState

        # ==================== PACMAN LOGIC ====================
        if is_pacman:
            # Priority -1: No food available - escape immediately to score carried food
            if not ("food_available",) in initState:
                if self.debug:
                    print(f"Agent {self.index}: Pacman - priority -1")
                return self.goalEscape(objects, initState)

            if is_scared:
                if self.debug:
                    print(f"Agent {self.index}: Pacman - priority -0.5")
                return self.goalEatFood(objects, initState)

            # Priority 0: If both are pacman and enemy in our territory, handle multiple invaders
            if teammate_is_pacman:
                # Highest priority: If we're further back and enemy has passed us, intercept immediately
                if ("further_back",) in initState:
                    for enemy_obj in enemy_objects:
                        if ("enemy_past", enemy_obj) in initState and (
                            "fat_agent_gt10",
                            enemy_obj,
                        ) in initState:
                            if not (("ally_chasing", enemy_obj) in initState):
                                if self.debug:
                                    print(f"Agent {self.index}: Pacman - priority 0/0")
                                return self.goalChase(objects, initState, enemy_obj)
                    for enemy_obj in enemy_objects:
                        if ("enemy_past", enemy_obj) in initState and (
                            "fat_agent_gt3",
                            enemy_obj,
                        ) in initState:
                            if not (("ally_chasing", enemy_obj) in initState):
                                if self.debug:
                                    print(f"Agent {self.index}: Pacman - priority 0/1")
                                return self.goalChase(objects, initState, enemy_obj)
                    for enemy_obj in enemy_objects:
                        if ("enemy_past", enemy_obj) in initState and (
                            "fat_agent",
                            enemy_obj,
                        ) in initState:
                            if not (("ally_chasing", enemy_obj) in initState):
                                if self.debug:
                                    print(f"Agent {self.index}: Pacman - priority 0/2")
                                return self.goalChase(objects, initState, enemy_obj)
                    for enemy_obj in enemy_objects:
                        if ("enemy_past", enemy_obj) in initState:
                            if not (("ally_chasing", enemy_obj) in initState):
                                if self.debug:
                                    print(f"Agent {self.index}: Pacman - priority 0/3")
                                return self.goalChase(objects, initState, enemy_obj)

            # Priority 1: Eat capsule if we're closest
            if ("closest_to_capsule",) in initState:
                for enemy_obj in enemy_objects:
                    if not (("is_scared", enemy_obj) in initState):
                        if self.debug:
                            print(f"Agent {self.index}: Pacman - priority 1")
                        return self.goalEatCapsule(objects, initState)

            # Priority 2: Don't have enough food to take lead - keep eating
            if not (("fat_agent", myObj) in initState):
                if self.debug:
                    print(f"Agent {self.index}: Pacman - priority 2")
                return self.goalEatFood(objects, initState)

            # Priority 3: Have enough food to take lead and there is no food nearby - escape
            if ("fat_agent_gt10", myObj) in initState or not (
                "near_food",
                myObj,
            ) in initState:
                if self.debug:
                    print(f"Agent {self.index}: Pacman - priority 3")
                return self.goalEscape(objects, initState)

            # Priority 4: Not under threat and timer high - keep eating
            if not ("enemy_close_distance",) in initState:
                if self.debug:
                    print(f"Agent {self.index}: Pacman - priority 4")
                return self.goalEatFood(objects, initState)

            # Priority 5: Have enough for lead - escape
            if self.debug:
                print(f"Agent {self.index}: Pacman - priority 5")
            return self.goalEscape(objects, initState)

        # ==================== GHOST LOGIC ====================

        # Priority 1: If scared, attack
        if is_scared:
            if self.debug:
                print(f"Agent {self.index}: Ghost - priority 1")
            return self.goalEatFood(objects, initState)

        # Get enemy objects for chase goals
        enemy_objects = [obj[0] for obj in objects if obj[1] in ["enemy1", "enemy2"]]

        # Priority 2: Chase fat enemies in priority of fatness (only if we're closer)
        for enemy_obj in enemy_objects:
            if ("fat_agent_gt10", enemy_obj) in initState:
                if (
                    not (("ally_chasing", enemy_obj) in initState)
                    and ("closer_to_enemy", enemy_obj) in initState
                ):
                    if self.debug:
                        print(f"Agent {self.index}: Ghost - priority 2/0")
                    return self.goalChase(objects, initState, enemy_obj)
            if ("fat_agent_gt3", enemy_obj) in initState:
                if (
                    not (("ally_chasing", enemy_obj) in initState)
                    and ("closer_to_enemy", enemy_obj) in initState
                ):
                    if self.debug:
                        print(f"Agent {self.index}: Ghost - priority 2/1")
                    return self.goalChase(objects, initState, enemy_obj)
            if ("fat_agent", enemy_obj) in initState:
                if (
                    not (("ally_chasing", enemy_obj) in initState)
                    and ("closer_to_enemy", enemy_obj) in initState
                ):
                    if self.debug:
                        print(f"Agent {self.index}: Ghost - priority 2/2")
                    return self.goalChase(objects, initState, enemy_obj)
            if ("is_pacman", enemy_obj) in initState:
                if (
                    not (("ally_chasing", enemy_obj) in initState)
                    and ("closer_to_enemy", enemy_obj) in initState
                ):
                    if self.debug:
                        print(f"Agent {self.index}: Ghost - priority 2/3")
                    return self.goalChase(objects, initState, enemy_obj)

        # Priority 3: Winning by >3, defend
        if ("winning_gt3",) in initState:
            if self.debug:
                print(f"Agent {self.index}: Ghost - priority 3")
            return self.goalDefaultDefend(objects, initState)

        # Priority 4: Teammate is chasing, we should attack
        # Check if ally is chasing any enemy
        ally_is_chasing = any(
            ("ally_chasing", eobj) in initState for eobj in enemy_objects
        )
        if ally_is_chasing:
            if self.debug:
                print(f"Agent {self.index}: Ghost - priority 4")
            return self.goalEatFood(objects, initState)

        # Priority 5: Losing by >3, attack
        if ("losing_gt3",) in initState:
            if self.debug:
                print(f"Agent {self.index}: Ghost - priority 5")
            return self.goalEatFood(objects, initState)

        # Priority 6: If noone is defending, then we defend
        if (
            ("ally_defending",) not in initState
            and (
                (
                    "is_pacman",
                    teammateObj,
                )
                not in initState
                and ("closer_to_closest_enemy",) in initState
            )
            or ("is_pacman", teammateObj) in initState
        ):
            if self.debug:
                print(f"Agent {self.index}: Ghost - priority 6")
            return self.goalDefaultDefend(objects, initState)

        # Priority 7: If we already have a lead, defend
        if ("winning_gt3",) in initState:
            if self.debug:
                print(f"Agent {self.index}: Ghost - priority 7")
            return self.goalDefaultDefend(objects, initState)

        # Priority 5: Not winning, attack
        # if ("winning",) not in initState:
        #     if ("is_pacman", teammateObj) not in initState and (
        #         "closer_to_closest_enemy",
        #     ) not in initState:
        #         if self.debug:
        #             print(f"Agent {self.index}: Ghost - priority 5")
        #         return self.goalEatFood(objects, initState)

        # Priority 8: Default - attack
        if self.debug:
            print(f"Agent {self.index}: Ghost - priority 8")
        return self.goalEatFood(objects, initState)

    def goalEatCapsule(self, objects: List[Tuple], initState: List[Tuple]):
        positiveGoals = []
        for obj in objects:
            if obj[1] in ["enemy1", "enemy2"]:
                positiveGoals.append(("is_scared", obj[0]))

        return positiveGoals, []

    def goalEatFood(self, objects: List[Tuple], initState: List[Tuple]):
        myObj = f"a{self.index}"
        return [
            ("fat_agent", myObj),
            ("fat_agent_gt3", myObj),
            ("fat_agent_gt10", myObj),
        ], []

    def goalEscape(self, objects: List[Tuple], initState: List[Tuple]):
        myObj = f"a{self.index}"
        return [], [("is_pacman", myObj)]

    def goalChase(self, objects: List[Tuple], initState: List[Tuple], enemy_obj):
        return [], [("is_pacman", enemy_obj)]

    def goalDefaultDefend(self, objects: List[Tuple], initState: List[Tuple]):
        return [("defend_foods",)], []

    def posSatisfyLowLevelPlan(self, gameState: GameState):
        if (
            self.lowLevelPlan == None
            or len(self.lowLevelPlan) == 0
            or self.lowLevelActionIndex >= len(self.lowLevelPlan)
        ):
            return False
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(
            myPos, self.lowLevelPlan[self.lowLevelActionIndex][0]
        )
        if nextPos != self.lowLevelPlan[self.lowLevelActionIndex][1]:
            return False
        return True

    # ------------------------------- Q-learning low level plan Functions -------------------------------

    """
    Iterate through all q-values that we get from all
    possible actions, and return the action associated
    with the highest q-value.
    """

    @profile
    def getLowLevelPlanQL(
        self, gameState: GameState, highLevelAction: str
    ) -> List[Tuple[str, Tuple]]:
        values = []
        legalActions = gameState.getLegalActions(self.index)
        rewardFunction = None
        featureFunction = None
        weights = None
        learningRate = 0

        ##########
        # Classify high level actions into offensive, retreat, or defensive categories
        ##########
        # Offensive actions: attack, aggressive_attack, desperate_attack
        if highLevelAction == "attack":
            # Offensive actions - use offensive features and rewards
            featureFunction = self.getAttackFeatures
            weights = MixedAgent.QLWeights["attackWeights"]
        # Retreat actions: go_home_with_food, go_home_retreat, emergency_retreat, or any action with "retreat"/"escape"/"go_home" in name
        elif highLevelAction == "eat_food":
            # Escape actions - complete reward function implemented
            featureFunction = self.getEatFoodFeatures
            weights = MixedAgent.QLWeights["eatFoodWeights"]
        elif highLevelAction == "eat_capsule":
            # Escape actions - complete reward function implemented
            featureFunction = self.getEatCapsuleFeatures
            weights = MixedAgent.QLWeights["eatCapsuleWeights"]
        elif highLevelAction == "escape":
            # Escape actions - complete reward function implemented
            featureFunction = self.getEscapeFeatures
            weights = MixedAgent.QLWeights["escapeWeights"]
        elif highLevelAction == "chase_enemy":
            # Escape actions - complete reward function implemented
            featureFunction = self.getChaseFeatures
            weights = MixedAgent.QLWeights["chaseWeights"]
        elif highLevelAction == "default_defend":
            # Escape actions - complete reward function implemented
            featureFunction = self.getDefaultDefendFeatures
            weights = MixedAgent.QLWeights["defaultDefendWeights"]
        else:
            # Defensive actions - complete reward function implemented - default weights
            featureFunction = self.getDefaultDefendFeatures
            weights = MixedAgent.QLWeights["defaultDefendWeights"]

        stateInfo = self.getStateInfo(gameState)

        if len(legalActions) != 0:
            for action in legalActions:
                nextStateInfo = self.getStateInfo(self.getSuccessor(gameState, action))
                features = featureFunction(stateInfo, nextStateInfo, action)
                qval = self.getQValue(features, weights)
                values.append((qval, action, features))

            # Debug: print Q-values and feature breakdown for all actions
            if len(values) > 0 and self.debug:
                print(f"\nAgent {self.index} ({highLevelAction}) - Q-values:")
                for qval, act, feats in values:
                    # Show only non-zero features
                    feat_breakdown = {
                        k: (
                            round(feats[k], 3),
                            round(weights[k], 2),
                            round(feats[k] * weights[k], 2),
                        )
                        for k in feats
                        if feats[k] != 0
                    }
                    print(
                        f"  {act}: Q={round(qval, 2)} | Features (val, weight, contribution): {feat_breakdown}"
                    )

            action = max(values, key=lambda x: x[0])[1]
            if self.debug:
                print(f"Agent {self.index}: Best action: {action}")
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos, action)
        return [(action, nextPos)]

    """
    Iterate through all features (closest food, bias, ghost dist),
    multiply each of the features' value to the feature's weight,
    and return the sum of all these values to get the q-value.
    """

    def getQValue(self, features, weights):
        return features * weights

    """
    Iterate through all features and for each feature, update
    its weight values using the following formula:
    w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
    """

    # ------------------------------- Feature Related Action Functions -------------------------------

    def getStateInfo(self, gameState: GameState):
        enemyVirtualStates = {}
        myPos = gameState.getAgentPosition(self.index)

        for enemy_idx in self.getOpponents(gameState):
            # Get the real enemy state from gameState (has correct scaredTimer, isPacman, etc.)
            real_enemy_state = gameState.getAgentState(enemy_idx)
            enemy_state = real_enemy_state.copy()

            # Determine the position to use
            estimated_pos = MixedAgent.ESTIMATED_POSITIONS.get(enemy_idx)
            real_pos = real_enemy_state.getPosition()
            enemy_start_pos = gameState.getInitialAgentPosition(enemy_idx)

            # Check if enemy was just eaten:
            # - Estimated position was close to us (within CLOSE_DISTANCE)
            # - Real position is now None (can't see them anymore)
            # This likely means they got eaten and respawned at start position
            if (
                estimated_pos is not None
                and (real_pos is None or real_pos == enemy_start_pos)
                and self.getMazeDistance(myPos, estimated_pos) <= BREATHING_DISTANCE
            ):
                # Enemy likely got eaten, use their start position
                position_to_use = enemy_start_pos
            else:
                # Use estimated position as normal
                position_to_use = estimated_pos

            # Only override the position with our chosen position, keep everything else (scaredTimer, etc.) from real state
            enemy_state.configuration = Configuration(
                position_to_use,
                real_enemy_state.configuration.direction
                if real_enemy_state.configuration
                else -1,
            )
            enemyVirtualStates[enemy_idx] = enemy_state

        return StateInfo(
            gameState,
            gameState.getAgentState(self.index),
            gameState.getAgentState((self.index + 2) % 4),
            enemyVirtualStates,
        )

    # ==================== Shared Helper Functions ====================

    def getConsecutiveStopReversePenalty(self):
        count = MixedAgent.CONSECUTIVE_STOP_REVERSE.get(self.index, 0)
        return 2**count

    def getDeathDistanceGhosts(self, myPos, enemyVirtualStates: Dict[int, AgentState]):
        count = 0
        for enemy_idx, enemy_state in enemyVirtualStates.items():
            if not enemy_state.isPacman and enemy_state.scaredTimer == 0:
                enemy_pos = enemy_state.getPosition()
                if enemy_pos:
                    dist = self.getMazeDistance(myPos, enemy_pos)
                    if dist <= DEATH_DISTANCE:
                        count += 1
        return count

    def getDeathDistancePacman(
        self, myState, enemyVirtualStates: Dict[int, AgentState]
    ):
        count = 0
        myPos = myState.getPosition()
        is_scared = myState.scaredTimer > 0
        if is_scared:
            for enemy_idx, enemy_state in enemyVirtualStates.items():
                if enemy_state.isPacman:
                    enemy_pos = enemy_state.getPosition()
                    dist = self.getMazeDistance(myPos, enemy_pos)
                    if dist <= DEATH_DISTANCE:
                        count += 1
        return count

    def getBreathingDistanceGhosts(
        self, myPos, enemyVirtualStates: Dict[int, AgentState]
    ):
        """Count non-scared ghosts within breathing distance (2 steps)"""
        count = 0
        for enemy_idx, enemy_state in enemyVirtualStates.items():
            if not enemy_state.isPacman and enemy_state.scaredTimer == 0:
                enemy_pos = enemy_state.getPosition()
                if enemy_pos:
                    dist = self.getMazeDistance(myPos, enemy_pos)
                    if dist <= BREATHING_DISTANCE:
                        count += 1
        return count

    def getCloseDistanceGhosts(self, myPos, enemyVirtualStates: Dict[int, AgentState]):
        """Count non-scared ghosts within close distance (4 steps)"""
        count = 0
        for _, enemy_state in enemyVirtualStates.items():
            if not enemy_state.isPacman and enemy_state.scaredTimer == 0:
                enemy_pos = enemy_state.getPosition()
                if enemy_pos:
                    dist = self.getMazeDistance(myPos, enemy_pos)
                    if dist <= CLOSE_DISTANCE:
                        count += 1
        return count

    def getGotEaten(self, agentState: AgentState, enemyVirtualStates):
        """
        Check if agent got eaten (sent back to start position).
        Returns 1.0 if eaten, 0.0 otherwise.
        """
        nextPos = agentState.getPosition()
        if nextPos == self.startPosition:
            return 1.0

        # Check if we're within DEATH_DISTANCE of any enemy ghost
        if agentState.isPacman:
            count = self.getDeathDistanceGhosts(nextPos, enemyVirtualStates)
            if count > 0:
                return 1.0

        # Check if we're within DEATH_DISTANCE of any enemy Pacman
        if not agentState.isPacman:
            count = self.getDeathDistancePacman(agentState, enemyVirtualStates)
            if count > 0:
                return 1.0

        return 0.0

    def getDistanceToHome(self, pos, gameState: GameState):
        """Get maze distance to home territory border"""
        if self.isInHome(pos, gameState):
            return 0

        # Get x-coordinate of the border
        walls = gameState.getWalls()
        width = walls.width
        if self.red:
            border_x = width // 2 - 1
        else:
            border_x = width // 2

        # Find closest border position
        height = walls.height
        min_dist = float("inf")
        for y in range(height):
            if not walls[border_x][y]:
                dist = self.getMazeDistance(pos, (border_x, y))
                if dist < min_dist:
                    min_dist = dist
        return min_dist if min_dist != float("inf") else 0

    def getDistanceToEnemyTerritory(self, pos, gameState: GameState):
        """Get maze distance to enemy territory border"""
        if self.isInEnemyTerritory(pos, gameState):
            return 0

        walls = gameState.getWalls()
        width = walls.width
        if self.red:
            border_x = width // 2  # Red wants to go to blue side
        else:
            border_x = width // 2 - 1  # Blue wants to go to red side

        # Find closest border position
        height = walls.height
        min_dist = float("inf")
        for y in range(height):
            if not walls[border_x][y]:
                dist = self.getMazeDistance(pos, (border_x, y))
                if dist < min_dist:
                    min_dist = dist
        return min_dist if min_dist != float("inf") else 0

    def isInEnemyTerritory(self, pos, gameState: GameState):
        """Check if position is in enemy territory"""
        walls = gameState.getWalls()
        width = walls.width
        x = int(pos[0])
        if self.red:
            return x >= width // 2
        else:
            return x < width // 2

    def wentHome(self, stateInfo: StateInfo):
        """Check if position is in home territory"""
        agentState = stateInfo.agentState
        pos = agentState.getPosition()
        isHome = not self.isInEnemyTerritory(pos, stateInfo.gameState)
        return isHome and pos != self.startPosition

    def isInHome(self, pos, gameState: GameState):
        """Check if position is in home territory"""
        return not self.isInEnemyTerritory(pos, gameState)

    def createDefensiveDistancer(self, gameState: GameState):
        """
        Create a distance calculator that treats enemy territory as walls.
        This is used for defensive positioning to avoid local minima at the border.
        """
        walls = gameState.getWalls().copy()
        width = walls.width
        height = walls.height

        # Mark enemy territory as walls
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:  # If not already a wall
                    if self.isInEnemyTerritory((x, y), gameState):
                        walls[x][y] = True

        # Create a new distancer with the modified walls
        return distanceCalculator.Distancer(gameState.data.layout, walls)

    def getDefensiveMazeDistance(self, pos1, pos2, gameState: GameState):
        """
        Get maze distance using the defensive distancer (enemy territory as walls).
        Returns distance to border position aligned with target if target is in enemy territory.
        """
        # If target is in enemy territory, calculate distance to aligned border position instead
        if self.isInEnemyTerritory(pos2, gameState):
            pos2 = self.getBorderPositionAlignedWith(pos2)

        return self.defensiveDistancer.getDistance(pos1, pos2)

    def getDistanceToNearestFood(self, pos, gameState: GameState):
        """Get distance to nearest food pellet on enemy side"""
        food = self.getFood(gameState).asList()
        if len(food) == 0:
            return 0
        return min([self.getMazeDistance(pos, f) for f in food])

    def getDistanceToNearestCapsule(self, pos, gameState: GameState):
        """Get distance to nearest capsule on enemy side"""
        capsules = self.getCapsules(gameState)
        if len(capsules) == 0:
            return 0
        return min([self.getMazeDistance(pos, c) for c in capsules])

    def getDistanceToTeammate(self, myPos, teammateState: AgentState):
        """Get distance to teammate"""
        teammate_pos = teammateState.getPosition()
        assert teammate_pos
        return self.getMazeDistance(myPos, teammate_pos)

    def getDistanceToAttackingTeammate(self, myPos, teammateState: AgentState):
        """Get distance to teammate"""
        teammate_pos = teammateState.getPosition()
        assert teammate_pos
        if teammateState.isPacman:
            return self.getMazeDistance(myPos, teammate_pos)
        return 0

    def getDistanceToNearestEnemy(
        self,
        myPos,
        enemyVirtualStates: Dict[int, AgentState],
        gameState=None,
        defendMode=False,
    ):
        """
        Get distance to nearest enemy (any type)

        Args:
            myPos: Current position
            enemyVirtualStates: Dict of enemy states
            defendMode: If True, use defensive distancer (enemy territory as walls) for optimal border positioning
        """
        min_dist = float("inf")
        def_dist = float("inf")
        teammate_assignment = MixedAgent.DEFENSIVE_ASSIGNMENTS[(self.index + 2) % 4]
        for enemy_idx, enemy_state in enemyVirtualStates.items():
            if defendMode and (teammate_assignment == enemy_idx):
                continue
            enemy_pos = enemy_state.getPosition()
            assert enemy_pos
            dist = self.getMazeDistance(myPos, enemy_pos)
            if dist < min_dist:
                min_dist = dist
                if defendMode:
                    MixedAgent.DEFENSIVE_ASSIGNMENTS[self.index] = enemy_idx
                    assert gameState
                    # For defense: use defensive distancer to avoid local minima at border
                    def_dist = self.getDefensiveMazeDistance(
                        myPos, enemy_pos, gameState
                    )
        if defendMode:
            return def_dist
        return min_dist

    def getBorderPositionAlignedWith(self, enemy_pos):
        """
        Get the border position at the same y-coordinate as the enemy.
        This is used for defensive positioning to align with enemy threats.
        """
        walls = self.getCurrentObservation().getWalls()
        width = walls.width

        # Get the x-coordinate of the border (our side)
        if self.red:
            border_x = width // 2 - 1
        else:
            border_x = width // 2

        enemy_y = int(enemy_pos[1])

        # Find the closest non-wall position at the border near enemy's y
        # Check enemy_y first, then spiral outward
        for y_offset in range(walls.height):
            for dy in [0] if y_offset == 0 else [-y_offset, y_offset]:
                check_y = enemy_y + dy
                if 0 <= check_y < walls.height and not walls[border_x][check_y]:
                    return (border_x, check_y)

        # Fallback: return middle of border if all else fails
        return (border_x, walls.height // 2)

    def getDistanceToEnemy(
        self, myPos, enemyVirtualStates: Dict[int, AgentState], target_enemy_idx
    ):
        """Get distance to a specific target enemy"""
        assert target_enemy_idx in enemyVirtualStates
        enemy_pos = enemyVirtualStates[target_enemy_idx].getPosition()
        assert enemy_pos

        return self.getMazeDistance(myPos, enemy_pos)

    def isBetweenEnemyAndEscape(
        self, myPos, target_enemy_state: AgentState, gameState: GameState
    ):
        """
        Check if we're between enemy and their escape route (home).
        Only relevant when enemy is a Pacman invading our territory.
        """
        # Only intercept if enemy is a Pacman (invading)
        if not target_enemy_state.isPacman:
            return 0.0

        target_enemy_pos = target_enemy_state.getPosition()
        assert target_enemy_pos

        width = gameState.getWalls().width
        my_x = int(myPos[0])
        enemy_x = int(target_enemy_pos[0])

        # Enemy is Pacman invading us, we want to block their escape to their home
        if self.red:
            border_x = width // 2
            return 1.0 if enemy_x <= my_x <= border_x else 0.0
        else:
            border_x = width // 2 - 1
            return 1.0 if border_x <= my_x <= enemy_x else 0.0

    # ==================== Feature Functions ====================

    def getAttackFeatures(self, stateInfo: StateInfo, nextStateInfo: StateInfo, action):
        """
        Features for crossing into enemy territory.
        Priority: avoid ghosts > get to enemy territory > spread out from teammate
        """
        features = util.Counter()
        myPos = stateInfo.agentState.getPosition()
        nextPos = nextStateInfo.agentState.getPosition()

        # Priority 1: Avoid breathing distance ghosts (most critical)
        features["breathing-distance-ghosts"] = self.getBreathingDistanceGhosts(
            nextPos, nextStateInfo.enemyVirtualStates
        )

        features["close-distance-ghosts"] = self.getCloseDistanceGhosts(
            nextPos, nextStateInfo.enemyVirtualStates
        )

        # Priority 2: Reward being in enemy territory
        features["in-enemy-territory"] = (
            1.0 if self.isInEnemyTerritory(nextPos, nextStateInfo.gameState) else 0.0
        )

        # Priority 3: Get closer to enemy territory
        features["distance-to-enemy-territory"] = self.getDistanceToEnemyTerritory(
            nextPos, nextStateInfo.gameState
        )

        # Priority 4: Spread out from teammate
        features["distance-to-teammate"] = self.getDistanceToTeammate(
            nextPos, nextStateInfo.teammateState
        )

        # Penalty: Getting eaten
        features["got-eaten"] = self.getGotEaten(
            nextStateInfo.agentState, nextStateInfo.enemyVirtualStates
        )

        exponential_penalty = self.getConsecutiveStopReversePenalty()
        features["stop-reverse"] = (
            exponential_penalty
            if action == Directions.STOP
            or action
            == Directions.REVERSE[stateInfo.agentState.configuration.direction]
            else 0.0
        )

        return features

    def getEatFoodFeatures(
        self, stateInfo: StateInfo, nextStateInfo: StateInfo, action
    ):
        """
        Features for eating food in enemy territory.
        Priority: avoid ghosts > eat food > get to food > avoid close ghosts
        """
        features = util.Counter()
        nextPos = nextStateInfo.agentState.getPosition()

        # Priority 1: Avoid breathing distance ghosts
        features["breathing-distance-ghosts"] = self.getBreathingDistanceGhosts(
            nextPos, nextStateInfo.enemyVirtualStates
        )

        # Priority 0: Must be in enemy territory
        # features["in-enemy-territory"] = (
        #     1.0 if self.isInEnemyTerritory(nextPos, nextStateInfo.gameState) else 0.0
        # )

        # Priority 2: Reward eating food
        currentCarrying = stateInfo.agentState.numCarrying
        nextCarrying = nextStateInfo.agentState.numCarrying
        features["ate-food"] = 1.0 if nextCarrying > currentCarrying else 0.0

        # Priority 3: Get closer to food
        features["distance-to-nearest-food"] = self.getDistanceToNearestFood(
            nextPos, nextStateInfo.gameState
        )

        # Priority 4: Avoid close distance ghosts
        features["close-distance-ghosts"] = self.getCloseDistanceGhosts(
            nextPos, nextStateInfo.enemyVirtualStates
        )

        # Priority 5: Avoid close to attacking teammate
        features[
            "distance-to-attacking-teammate"
        ] = self.getDistanceToAttackingTeammate(nextPos, nextStateInfo.teammateState)

        # Penalty: Getting eaten
        features["got-eaten"] = self.getGotEaten(
            nextStateInfo.agentState, nextStateInfo.enemyVirtualStates
        )

        exponential_penalty = self.getConsecutiveStopReversePenalty()
        features["stop-reverse"] = (
            exponential_penalty
            if action == Directions.STOP
            or action
            == Directions.REVERSE[stateInfo.agentState.configuration.direction]
            else 0.0
        )

        return features

    def getEatCapsuleFeatures(
        self, stateInfo: StateInfo, nextStateInfo: StateInfo, action
    ):
        """
        Features for eating capsules.
        Priority: avoid ghosts > eat capsule > get to capsule > avoid close ghosts
        """
        features = util.Counter()
        nextPos = nextStateInfo.agentState.getPosition()

        # Priority 1: Avoid breathing distance ghosts
        features["breathing-distance-ghosts"] = self.getBreathingDistanceGhosts(
            nextPos, nextStateInfo.enemyVirtualStates
        )

        # Priority 0: Must be in enemy territory
        # features["in-enemy-territory"] = (
        #     1.0 if self.isInEnemyTerritory(nextPos, nextStateInfo.gameState) else 0.0
        # )

        # Priority 2: Reward eating capsule
        currentCapsules = self.getCapsules(stateInfo.gameState)
        nextCapsules = self.getCapsules(nextStateInfo.gameState)
        features["ate-capsule"] = (
            1.0 if len(nextCapsules) < len(currentCapsules) else 0.0
        )

        # Priority 3: Get closer to capsule
        features["distance-to-nearest-capsule"] = self.getDistanceToNearestCapsule(
            nextPos, nextStateInfo.gameState
        )

        # Priority 4: Avoid close distance ghosts
        features["close-distance-ghosts"] = self.getCloseDistanceGhosts(
            nextPos, nextStateInfo.enemyVirtualStates
        )

        # # Priority 5: Avoid close to attacking teammate
        # features["distance-to-attacking-teammate"] = self.getDistanceToAttackingTeammate(
        #     nextPos, nextStateInfo.teammateState
        # )

        # Penalty: Getting eaten
        features["got-eaten"] = self.getGotEaten(
            nextStateInfo.agentState, nextStateInfo.enemyVirtualStates
        )

        exponential_penalty = self.getConsecutiveStopReversePenalty()
        features["stop-reverse"] = (
            exponential_penalty
            if action == Directions.STOP
            or action
            == Directions.REVERSE[stateInfo.agentState.configuration.direction]
            else 0.0
        )

        return features

    def getEscapeFeatures(self, stateInfo: StateInfo, nextStateInfo: StateInfo, action):
        """
        Features for escaping back home with food.
        Priority: reach home > avoid breathing ghosts > get closer to home > avoid close ghosts
        """
        features = util.Counter()
        myPos = stateInfo.agentState.getPosition()
        nextPos = nextStateInfo.agentState.getPosition()

        # Priority 1: Reward reaching home
        features["in-home"] = 1.0 if self.wentHome(stateInfo) else 0.0

        # Priority 2: Avoid breathing distance ghosts
        features["breathing-distance-ghosts"] = self.getBreathingDistanceGhosts(
            nextPos, nextStateInfo.enemyVirtualStates
        )

        # Priority 2: Get closer to home
        features["distance-to-home"] = self.getDistanceToHome(
            nextPos, nextStateInfo.gameState
        )

        # Priority 3: Avoid close distance ghosts
        features["close-distance-ghosts"] = self.getCloseDistanceGhosts(
            nextPos, nextStateInfo.enemyVirtualStates
        )

        # Penalty: Getting eaten
        features["got-eaten"] = self.getGotEaten(
            nextStateInfo.agentState, nextStateInfo.enemyVirtualStates
        )

        # Tie-breaking: Penalize stop and reverse
        exponential_penalty = self.getConsecutiveStopReversePenalty()
        features["stop-reverse"] = (
            exponential_penalty
            if action == Directions.STOP
            or action
            == Directions.REVERSE[stateInfo.agentState.configuration.direction]
            else 0.0
        )

        return features

    def getChaseFeatures(self, stateInfo: StateInfo, nextStateInfo: StateInfo, action):
        """
        Features for chasing a specific enemy.
        Priority: stay home > get closer to enemy > intercept escape route > spread out from teammate
        """
        current_action = MixedAgent.CURRENT_ACTION.get(self.index)
        assert hasattr(current_action, "name") and hasattr(current_action, "parameters")
        assert (
            current_action.name == "chase_enemy" and len(current_action.parameters) == 2
        )

        # Extract enemy index from parameter like "e1" or "e3"
        enemy_obj = current_action.parameters[1]
        enemy_index = int(enemy_obj[1:])  # "e1" -> 1, "e3" -> 3

        assert (
            enemy_index in nextStateInfo.enemyVirtualStates
        ), f"{enemy_index} not in {nextStateInfo.enemyVirtualStates}"

        features = util.Counter()
        nextPos = nextStateInfo.agentState.getPosition()
        target_enemy_state = nextStateInfo.enemyVirtualStates[enemy_index]

        # Priority 1: Must stay in home territory
        # features["in-home"] = (
        #     1.0 if self.isInHome(nextPos, nextStateInfo.gameState) else 0.0
        # )

        features["ate_enemy"] = (
            1.0
            if target_enemy_state.getPosition()
            == target_enemy_state.start.getPosition()
            else 0.0
        )

        # Priority 2: Get closer to target enemy
        features["distance-to-enemy"] = self.getDistanceToEnemy(
            nextPos, nextStateInfo.enemyVirtualStates, enemy_index
        )

        # Priority 3: Position between enemy and their escape
        features["between-enemy-and-escape"] = self.isBetweenEnemyAndEscape(
            nextPos, target_enemy_state, nextStateInfo.gameState
        )

        # Priority 4: Spread out from teammate
        features["distance-to-teammate"] = self.getDistanceToTeammate(
            nextPos, nextStateInfo.teammateState
        )

        # Penalty: Getting eaten (shouldn't happen while defending, but just in case)
        features["got-eaten"] = self.getGotEaten(
            nextStateInfo.agentState, nextStateInfo.enemyVirtualStates
        )

        # Tie-breaking: Penalize stop and reverse
        exponential_penalty = self.getConsecutiveStopReversePenalty()
        features["stop-reverse"] = (
            exponential_penalty
            if action == Directions.STOP
            or action
            == Directions.REVERSE[stateInfo.agentState.configuration.direction]
            else 0.0
        )

        return features

    def getDefaultDefendFeatures(
        self, stateInfo: StateInfo, nextStateInfo: StateInfo, action
    ):
        """
        Features for default defensive behavior (patrolling).
        Priority: stay home > position near potential threats > spread out from teammate
        """
        features = util.Counter()
        nextPos = nextStateInfo.agentState.getPosition()

        # Priority 1: Must stay in home territory
        features["in-home"] = (
            1.0 if self.isInHome(nextPos, nextStateInfo.gameState) else 0.0
        )

        # Priority 2: Position near enemies (using defensive distancer to avoid border local minima)
        distance_to_nearest_enemy = self.getDistanceToNearestEnemy(
            nextPos,
            nextStateInfo.enemyVirtualStates,
            nextStateInfo.gameState,
            defendMode=True,
        )
        features["distance-to-nearest-enemy"] = distance_to_nearest_enemy

        # Priority 3: Spread out from teammate
        features["distance-to-teammate"] = self.getDistanceToTeammate(
            nextPos, nextStateInfo.teammateState
        )

        # Check if we're both defending
        # current_action = MixedAgent.CURRENT_ACTION.get((self.index + 2) % 4)
        # if hasattr(current_action, "name") and current_action.name == "defend":
        #     features[
        #         "distance-to-teammate-both-defending"
        #     ] = self.getDistanceToTeammate(nextPos, nextStateInfo.teammateState)

        # Penalty: Getting eaten (shouldn't happen while defending, but just in case)
        features["got-eaten"] = self.getGotEaten(
            nextStateInfo.agentState, nextStateInfo.enemyVirtualStates
        )

        # Tie-breaking: Penalize stop and reverse
        # exponential_penalty = self.getConsecutiveStopReversePenalty()
        # features["stop-reverse"] = (
        #     exponential_penalty
        #     if action == Directions.STOP
        #     or action
        #     == Directions.REVERSE[stateInfo.agentState.configuration.direction]
        #     else 0.0
        # )

        return features

    def closestFood(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no food found
        return None

    def stateClosestFood(self, gameState: GameState):
        pos = gameState.getAgentPosition(self.index)
        food = self.getFood(gameState)
        walls = gameState.getWalls()
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        # no food found
        return None

    def getSuccessor(self, gameState: GameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getGhostLocs(self, gameState: GameState):
        ghosts = []
        opAgents = CaptureAgent.getOpponents(self, gameState)
        # Get ghost locations and states if observable
        if opAgents:
            for opponent in opAgents:
                opPos = gameState.getAgentPosition(opponent)
                opIsPacman = gameState.getAgentState(opponent).isPacman
                if opPos and not opIsPacman:
                    ghosts.append(opPos)
        return ghosts


# ==================== Topological Map Preprocessing ====================


def build_map_topology(walls) -> MapTopology:
    """
    Convert the tile-based map into a topological graph for strategic analysis.

    This identifies junctions (decision points), corridors (edges), articulation points
    (critical choke points), and dead-end zones.

    Args:
        walls: Grid of walls from gameState.getWalls()

    Returns:
        MapTopology object with all preprocessed data
    """
    width, height = walls.width, walls.height

    # Step 1: Identify junctions (nodes in our graph)
    junctions = find_junctions(walls, width, height)

    # Step 2: Build corridors (edges connecting junctions)
    corridors, tile_to_corridor = build_corridors(junctions, walls, width, height)

    # Step 3: Find articulation points (critical choke points)
    articulation_points = find_articulation_points(junctions)

    # Step 4: Identify dead-end zones
    dead_end_zones = find_dead_end_zones(
        junctions, articulation_points, walls, width, height
    )

    topology = MapTopology(
        junctions=junctions,
        corridors=corridors,
        tile_to_corridor=tile_to_corridor,
        articulation_points=articulation_points,
        dead_end_zones=dead_end_zones,
    )

    return topology


def visualize_topology(topology: MapTopology, walls):
    """
    Print visual representations of the map with different features highlighted.

    Args:
        topology: MapTopology object
        walls: Grid of walls from gameState.getWalls()
    """
    width, height = walls.width, walls.height

    # Helper function to create a map string
    def create_map_string(highlight_tiles: Dict[Tuple[int, int], str], title: str):
        """Create a string representation of the map with highlighted tiles."""
        result = [f"\n{'=' * 60}", f"{title:^60}", "=" * 60]

        # Build map from bottom to top (y decreases as we go down in output)
        for y in range(height - 1, -1, -1):
            row = []
            for x in range(width):
                pos = (x, y)
                if walls[x][y]:
                    row.append("%")  # Wall
                elif pos in highlight_tiles:
                    row.append(highlight_tiles[pos])  # Highlighted tile
                else:
                    row.append(" ")  # Empty space
            result.append("".join(row))

        result.append("=" * 60)
        return "\n".join(result)

    # 1. Junctions map
    junction_tiles = {}
    for pos, junction in topology.junctions.items():
        if junction.junction_type == "junction":
            junction_tiles[pos] = "J"  # Junction (3+ neighbors)
        elif junction.junction_type == "dead_end":
            junction_tiles[pos] = "D"  # Dead end (1 neighbor)

    print(create_map_string(junction_tiles, "JUNCTIONS (J=junction, D=dead end)"))

    # 2. Corridors map
    corridor_tiles = {}
    # Use different characters for different corridors (cycling through a set)
    corridor_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    for corridor_id, corridor in topology.corridors.items():
        char = corridor_chars[corridor_id % len(corridor_chars)]
        for pos in corridor.path:
            if pos not in topology.junctions:  # Don't overwrite junctions
                corridor_tiles[pos] = char

    # Add junctions as endpoints
    for pos in topology.junctions.keys():
        corridor_tiles[pos] = "+"

    print(
        create_map_string(
            corridor_tiles, "CORRIDORS (+= junction, 0-9A-Za-z = corridor ID)"
        )
    )

    # 3. Articulation points map
    articulation_tiles = {}
    for pos in topology.articulation_points:
        articulation_tiles[pos] = "A"  # Articulation point

    # Also show other junctions for context
    for pos, junction in topology.junctions.items():
        if pos not in articulation_tiles:
            articulation_tiles[pos] = "."  # Regular junction

    print(
        create_map_string(
            articulation_tiles, "ARTICULATION POINTS (A=critical choke, .=junction)"
        )
    )

    # 4. Dead-end zones map
    dead_zone_tiles = {}

    # Group dead-end zones by their exit junction
    exit_junction_to_char = {}
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    char_index = 0

    for tile_pos, exit_junction in topology.dead_end_zones.items():
        if exit_junction not in exit_junction_to_char:
            exit_junction_to_char[exit_junction] = chars[char_index % len(chars)]
            char_index += 1

        dead_zone_tiles[tile_pos] = exit_junction_to_char[exit_junction]

    # Mark exit junctions with 'X'
    for exit_junction in exit_junction_to_char.keys():
        dead_zone_tiles[exit_junction] = "X"

    print(create_map_string(dead_zone_tiles, "DEAD-END ZONES (X=exit, A-Z0-9=zone ID)"))

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print(f"{'TOPOLOGY SUMMARY':^60}")
    print("=" * 60)
    print(f"Total junctions: {len(topology.junctions)}")
    print(
        f"  - Decision points (3+ neighbors): {sum(1 for j in topology.junctions.values() if j.junction_type == 'junction')}"
    )
    print(
        f"  - Dead ends (1 neighbor): {sum(1 for j in topology.junctions.values() if j.junction_type == 'dead_end')}"
    )
    print(f"Total corridors: {len(topology.corridors)}")
    print(f"Articulation points (critical chokes): {len(topology.articulation_points)}")
    print(f"Tiles in dead-end zones: {len(topology.dead_end_zones)}")
    print(f"Dead-end zones: {len(set(topology.dead_end_zones.values()))}")
    print("=" * 60 + "\n")


def find_junctions(walls, width: int, height: int) -> Dict[Tuple[int, int], Junction]:
    """
    Identify all junctions (decision points) and dead ends in the map.

    A junction is a tile with 3+ non-wall neighbors (decision point).
    A dead end is a tile with 1 non-wall neighbor.
    """
    junctions = {}

    for x in range(width):
        for y in range(height):
            if walls[x][y]:
                continue

            # Get non-wall neighbors (includes current position)
            neighbors = Actions.getLegalNeighbors((x, y), walls)

            # Remove current position from neighbors if present
            neighbors = [n for n in neighbors if n != (x, y)]
            num_neighbors = len(neighbors)

            # Junction: 3+ neighbors (decision point) or dead end (1 neighbor)
            # Tiles with exactly 2 neighbors are corridor tiles, not junctions
            if num_neighbors == 1 or num_neighbors >= 3:
                junction_type = "dead_end" if num_neighbors == 1 else "junction"
                junctions[(x, y)] = Junction(
                    pos=(x, y),
                    neighbors=neighbors,
                    junction_type=junction_type,
                    connected_junctions={},
                )

    return junctions


def build_corridors(
    junctions: Dict[Tuple[int, int], Junction], walls, width: int, height: int
):
    """
    Build corridors (edges) connecting junctions by following paths with only 2 neighbors.

    Returns:
        - corridors: Dict of corridor_id -> Corridor
        - tile_to_corridor: Dict of tile_pos -> corridor_id
    """
    corridors = {}
    tile_to_corridor = {}
    corridor_id = 0

    # For each junction, trace corridors to connected junctions
    for junction_pos, junction in junctions.items():
        for neighbor_pos in junction.neighbors:
            # Check if we've already explored this corridor from the other end
            if neighbor_pos in junctions:
                # Direct connection between two junctions (no corridor tiles)
                other_junction_pos = neighbor_pos
                if other_junction_pos not in junction.connected_junctions:
                    junction.connected_junctions[other_junction_pos] = 1
                    junctions[other_junction_pos].connected_junctions[junction_pos] = 1

                    # Create corridor
                    corridor = Corridor(
                        corridor_id=corridor_id,
                        junction_a=junction_pos,
                        junction_b=other_junction_pos,
                        length=1,
                        path=[junction_pos, other_junction_pos],
                    )
                    corridors[corridor_id] = corridor
                    corridor_id += 1
            else:
                # Trace the corridor
                path, end_junction = trace_corridor(
                    junction_pos, neighbor_pos, junctions, walls
                )

                if end_junction and end_junction not in junction.connected_junctions:
                    corridor_length = (
                        len(path) - 1
                    )  # Subtract 1 to not count the starting junction
                    junction.connected_junctions[end_junction] = corridor_length
                    junctions[end_junction].connected_junctions[
                        junction_pos
                    ] = corridor_length

                    # Create corridor
                    corridor = Corridor(
                        corridor_id=corridor_id,
                        junction_a=junction_pos,
                        junction_b=end_junction,
                        length=corridor_length,
                        path=path,
                    )
                    corridors[corridor_id] = corridor

                    # Map all tiles in this corridor
                    for tile_pos in path:
                        if (
                            tile_pos not in junctions
                        ):  # Don't override junction positions
                            tile_to_corridor[tile_pos] = corridor_id

                    corridor_id += 1

    return corridors, tile_to_corridor


def trace_corridor(
    start_junction: Tuple[int, int],
    first_tile: Tuple[int, int],
    junctions: Dict[Tuple[int, int], Junction],
    walls,
) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """
    Trace a corridor from a junction until we hit another junction.

    Returns:
        - path: List of tiles in the corridor (including both endpoints)
        - end_junction: The junction we ended at (or None if we hit a wall)
    """
    path = [start_junction]  # Start with just the starting junction
    current = first_tile
    previous = start_junction

    while True:
        # Check if current position is a junction FIRST
        if current in junctions:
            path.append(current)
            return path, current

        # Not a junction yet, add to path
        path.append(current)

        # Get neighbors (excluding current position)
        neighbors = Actions.getLegalNeighbors(current, walls)
        neighbors = [n for n in neighbors if n != current]

        # Find the forward neighbor (not the one we came from)
        forward_neighbors = [n for n in neighbors if n != previous]

        if len(forward_neighbors) != 1:
            # This shouldn't happen if junctions are identified correctly
            # If we have 0 neighbors: dead end (should be a junction)
            # If we have 2+ neighbors: decision point (should be a junction)
            return path, None

        # Continue down the corridor
        next_tile = forward_neighbors[0]
        previous = current
        current = next_tile


def find_articulation_points(junctions: Dict[Tuple[int, int], Junction]) -> set:
    """
    Find articulation points (cut vertices) in the junction graph using DFS.

    An articulation point is a junction that, if removed, would disconnect the graph.
    These are critical choke points.
    """
    if not junctions:
        return set()

    articulation_points = set()
    visited = set()
    disc = {}  # Discovery time
    low = {}  # Lowest discovery time reachable
    parent = {}
    time = [0]

    def dfs(u):
        children = 0
        visited.add(u)
        disc[u] = low[u] = time[0]
        time[0] += 1

        for v in junctions[u].connected_junctions.keys():
            if v not in visited:
                children += 1
                parent[v] = u
                dfs(v)

                # Check if subtree rooted at v has connection back to ancestors of u
                low[u] = min(low[u], low[v])

                # u is articulation point in two cases:
                # 1) u is root of DFS tree and has two or more children
                # 2) u is not root and low[v] >= disc[u]
                if parent.get(u) is None and children > 1:
                    articulation_points.add(u)
                if parent.get(u) is not None and low[v] >= disc[u]:
                    articulation_points.add(u)
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    # Run DFS from first junction
    first_junction = next(iter(junctions.keys()))
    parent[first_junction] = None
    dfs(first_junction)

    return articulation_points


def find_dead_end_zones(
    junctions: Dict[Tuple[int, int], Junction],
    articulation_points: set,
    walls,
    width: int,
    height: int,
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Identify tiles that belong to dead-end zones (regions with only one exit).

    Returns:
        Dict mapping tile_pos -> exit_junction_pos
    """
    # Two-pass algorithm:
    # Pass 1: Find all components for all articulation points
    # Pass 2: Only mark components that appear for exactly one articulation point (true dead-ends)

    component_to_exits = (
        {}
    )  # Maps frozenset of junctions -> list of articulation points

    # Pass 1: Find all components
    for art_point in articulation_points:
        visited = set([art_point])  # Don't traverse through the articulation point

        # Start BFS/DFS from each neighbor of the articulation point
        for neighbor_junction in junctions[art_point].connected_junctions.keys():
            # Skip other articulation points - they separate zones
            if neighbor_junction in articulation_points:
                continue

            if neighbor_junction not in visited:
                # This starts a new component
                component_junctions = set()
                queue = [neighbor_junction]
                visited.add(neighbor_junction)

                while queue:
                    current_junction = queue.pop(0)
                    component_junctions.add(current_junction)

                    for next_junction in junctions[
                        current_junction
                    ].connected_junctions.keys():
                        # Don't traverse through other articulation points
                        if next_junction in articulation_points:
                            continue
                        if next_junction not in visited:
                            visited.add(next_junction)
                            queue.append(next_junction)

                # Track which articulation points can access this component
                component_id = frozenset(component_junctions)
                if component_id not in component_to_exits:
                    component_to_exits[component_id] = []
                component_to_exits[component_id].append(art_point)

    # Pass 2: Only mark TRUE dead-end zones (components with exactly one exit)
    dead_end_zones = {}

    for component_junctions, exit_points in component_to_exits.items():
        if len(exit_points) == 1:
            # This is a true dead-end zone with only one exit
            exit_junction = exit_points[0]

            # Flood fill from all junctions in this component
            for junction in component_junctions:
                flood_fill_dead_end_zone(
                    junction,
                    exit_junction,
                    dead_end_zones,
                    walls,
                    width,
                    height,
                    junctions,
                    articulation_points,
                )

            # Also mark the articulation point itself as part of this zone (it's the exit)
            if exit_junction not in dead_end_zones:
                dead_end_zones[exit_junction] = exit_junction

    return dead_end_zones


def flood_fill_dead_end_zone(
    start_pos: Tuple[int, int],
    exit_junction: Tuple[int, int],
    dead_end_zones: Dict[Tuple[int, int], Tuple[int, int]],
    walls,
    width: int,
    height: int,
    junctions: Dict[Tuple[int, int], Junction],
    articulation_points: set,
):
    """
    Flood fill to mark all tiles in a dead-end zone.
    Stops at articulation points (don't cross into other zones).
    """
    if start_pos in dead_end_zones:
        return  # Already processed

    visited = set()
    queue = [start_pos]

    while queue:
        current_pos = queue.pop(0)

        if current_pos in visited:
            continue

        visited.add(current_pos)

        # Mark this tile as part of the dead-end zone
        if current_pos not in dead_end_zones:
            dead_end_zones[current_pos] = exit_junction

        # Explore neighbors
        neighbors = Actions.getLegalNeighbors(current_pos, walls)
        for neighbor in neighbors:
            # Don't cross ANY articulation points (they separate zones)
            if neighbor in articulation_points:
                continue
            if neighbor not in visited:
                queue.append(neighbor)


@profile
def initialize_beliefs(game_state: GameState) -> Dict[int, List[List[float]]]:
    """
    Initialize belief distributions for all 4 agents to their exact starting positions.

    Args:
        game_state: Initial game state

    Returns:
        beliefs: Dict mapping agent_idx -> 2D probability array
                 Each agent gets a belief distribution initialized to their start position
    """
    walls = game_state.getWalls()
    width, height = walls.width, walls.height

    beliefs = {}

    for agent_idx in range(4):
        # Create empty probability array
        prob_array = [[0.0 for _ in range(height)] for _ in range(width)]

        # Get agent's exact starting position
        agent_pos = game_state.getAgentState(agent_idx).start.pos
        x, y = int(agent_pos[0]), int(agent_pos[1])
        prob_array[x][y] = 1.0  # Certain they're at starting position

        beliefs[agent_idx] = prob_array

    return beliefs


@profile
def update_belief(
    prev_belief: List[List[float]],
    game_state: GameState,
    opponent_idx: int,
    observer_idx: int,
) -> List[List[float]]:
    """
    Update belief distribution for one opponent.

    Steps:
    1. If opponent is visible, reset belief to exact position
    2. If opponent just moved, propagate belief forward
    3. Apply Bayesian update using noisy distance observation

    Args:
        prev_belief: Previous belief distribution [[prob]]
        game_state: Current game state
        opponent_idx: Index of opponent to update
        observer_idx: Index of agent doing the observing

    Returns:
        Updated belief distribution
    """
    walls = game_state.getWalls()
    width, height = walls.width, walls.height

    # Step 1: Check if opponent is visible
    if game_state.getAgentPosition(opponent_idx) is not None:
        # EXACT OBSERVATION - reset belief to certain position
        opponent_pos = game_state.getAgentPosition(opponent_idx)
        new_belief = [[0.0 for _ in range(height)] for _ in range(width)]
        x, y = int(opponent_pos[0]), int(opponent_pos[1])
        new_belief[x][y] = 1.0
        return new_belief

    # Step 2: Propagate beliefs forward ONLY if this opponent just moved
    # The agent who just moved before observer is (observer_idx - 1) % 4
    prev_agent_idx = (observer_idx - 1) % 4

    if opponent_idx == prev_agent_idx:
        new_beliefs = [[0.0 for _ in range(height)] for _ in range(width)]

        for x in range(width):
            for y in range(height):
                if prev_belief[x][y] > 0:
                    # From position (x,y), distribute probability to reachable neighbors
                    neighbors = Actions.getLegalNeighbors((x, y), walls)
                    prob_per_neighbor = prev_belief[x][y] / len(neighbors)

                    for nx, ny in neighbors:
                        new_beliefs[nx][ny] += prob_per_neighbor

        prev_belief = new_beliefs

    # Step 3: Bayesian update using noisy distance observation
    observer_pos = game_state.getAgentPosition(observer_idx)
    noisy_distances = game_state.getAgentDistances()
    assert noisy_distances
    noisy_dist = noisy_distances[opponent_idx]

    updated_belief = [[0.0 for _ in range(height)] for _ in range(width)]
    for x in range(width):
        for y in range(height):
            if prev_belief[x][y] > 0:
                # P(pos | observation) ∝ P(observation | pos) * P(pos)
                prior = prev_belief[x][y]
                true_dist = manhattanDistance(observer_pos, (x, y))
                likelihood = game_state.getDistanceProb(true_dist, noisy_dist)
                updated_belief[x][y] = prior * likelihood

    # Step 4: Normalize
    total_prob = sum(sum(row) for row in updated_belief)
    if total_prob > 0:
        for x in range(width):
            for y in range(height):
                updated_belief[x][y] /= total_prob

    return updated_belief


@profile
def update_all_beliefs(
    prev_beliefs: Dict[int, List[List[float]]],
    game_state: GameState,
    observer_idx: int,
) -> Dict[int, List[List[float]]]:
    """
    Update belief distributions for all opponents from observer's perspective.

    Args:
        prev_beliefs: Previous beliefs for all agents
        game_state: Current game state
        observer_idx: Index of agent doing the observing

    Returns:
        Updated beliefs for all agents
    """
    # Determine observer's opponents
    if observer_idx in [0, 2]:
        opponents = [1, 3]  # Red team observes Blue team
    else:
        opponents = [0, 2]  # Blue team observes Red team

    updated_beliefs = prev_beliefs.copy()

    for opponent_idx in opponents:
        updated_beliefs[opponent_idx] = update_belief(
            prev_beliefs[opponent_idx],
            game_state,
            opponent_idx,
            observer_idx,
        )

    return updated_beliefs
