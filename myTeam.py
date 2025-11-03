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
from collections import defaultdict, namedtuple


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

    # Also can use class variable to exchange information between agents.
    CURRENT_ACTION = {}
    ESTIMATED_POSITIONS = {}  # Cache for estimated enemy positions using beliefs
    MAP_TOPOLOGY: MapTopology = None  # Cached topological map analysis
    CURRENT_ADVANTAGES = {}  # Cache for current distance advantage on junctions
    CURRENT_LAYOUT_STR = (
        None  # String representation of current layout for cache invalidation
    )
    RED_FOOD = []
    RED_CAPSULES = []
    BLUE_FOOD = []
    BLUE_CAPSULES = []
    RED_FOOD_JUNCTIONS = util.Counter()
    RED_CAPSULE_JUNCTIONS = util.Counter()
    BLUE_FOOD_JUNCTIONS = util.Counter()
    BLUE_CAPSULE_JUNCTIONS = util.Counter()
    RED_FOOD_CORRIDORS = set()
    RED_CAPSULE_CORRIDORS = set()
    BLUE_FOOD_CORRIDORS = set()
    BLUE_CAPSULE_CORRIDORS = set()
    RED_ESCAPE_POINTS = []
    BLUE_ESCAPE_POINTS = []
    WINS = 0

    def registerInitialState(self, gameState: GameState):
        if MixedAgent.WINS >= 28:
            raise Exception("Forfeiting...")

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

        self.debug = False

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

        self.walls = gameState.getWalls()

        if MixedAgent.CURRENT_LAYOUT_STR != layout_str:
            # New layout detected - rebuild topology
            MixedAgent.MAP_TOPOLOGY = build_map_topology(self.walls)
            MixedAgent.CURRENT_LAYOUT_STR = layout_str

            if self.debug:
                print(f"\nAgent {self.index}: Built map topology for new layout")
                visualize_topology(MixedAgent.MAP_TOPOLOGY, self.walls)

        # Initialize values for distance advantage at all junctions
        MixedAgent.CURRENT_ADVANTAGES = self.calculate_advantages(gameState)

        # Use a dictionary to save information about current agent.
        MixedAgent.CURRENT_ACTION[self.index] = {}

        self.update_critical_junctions(gameState)
        self.initializeEscapePoints()

    def final(self, gameState: GameState):
        """
        This function write weights into files after the game is over.
        You may want to comment (disallow) this function when submit to contest server.
        """
        score = gameState.getScore()
        score = score if self.red else -score
        if self.index in [0, 1] and score > 0:
            MixedAgent.WINS += 1

    @profile
    def chooseAction(self, gameState: GameState):
        """
        This is the action entry point for the agent.
        In the game, this function is called when its current agent's turn to move.

        We first pick a high-level action.
        Then generate low-level action (up down left right wait) to achieve the high-level action.
        """

        if gameState.getAgentState(self.index).getPosition() == self.startPosition:
            if self.debug:
                print(f"Agent {self.index} died")
        # Update belief tracking for opponents

        MixedAgent.OPPONENT_BELIEFS = update_all_beliefs(
            MixedAgent.OPPONENT_BELIEFS, gameState, self.index
        )

        # Update advantages at all junctions
        self.advantages = self.calculate_advantages(gameState)

        # Find critical attacking and defending junctions
        self.update_critical_junctions(gameState)

        # TODO uncomment
        # -------------High Level Plan Section-------------------
        # Get high level action from a pddl plan.

        # Collect objects and init states from gameState
        # objects, initState = self.get_pddl_state(gameState)
        # positiveGoal, negtiveGoal = self.getGoals(objects, initState)
        #
        # # Check if we can stick to current plan
        # if not self.stateSatisfyCurrentPlan(initState, positiveGoal, negtiveGoal):
        #     # Cannot stick to current plan, prepare goals and replan
        #     if self.debug:
        #         print(f"Agent {self.index} replanning:")
        #         print(f"  Positive Goal: {positiveGoal}")
        #         print(f"  Negative Goal: {negtiveGoal}")
        #     self.highLevelPlan: List[Tuple[Action, pddl_state]] = self.getHighLevelPlan(
        #         objects, initState, positiveGoal, negtiveGoal
        #     )  # Plan is a list Action and pddl_state
        #     self.currentActionIndex = 0
        #     self.lowLevelPlan = []  # reset low level plan
        #     self.currentNegativeGoalStates = negtiveGoal
        #     self.currentPositiveGoalStates = positiveGoal
        #     if self.debug:
        #         print(f"  Plan: {[action.name for action, _ in self.highLevelPlan]}")
        #
        # if not self.highLevelPlan:
        #     if self.debug:
        #         print(f"No plan found for predicates: {initState}")
        #     highLevelAction = Action("default_defend", None, [], [], [], [])
        # else:
        #     # Get next action from the plan
        #     highLevelAction = self.highLevelPlan[self.currentActionIndex][0]
        # MixedAgent.CURRENT_ACTION[self.index] = highLevelAction
        #
        # if self.debug:
        #     print(f"Agent {self.index}: High-Level Action = {highLevelAction.name}")
        #
        # if highLevelAction.name != "default_defend":
        #     MixedAgent.DEFENSIVE_ASSIGNMENTS[self.index] = -1

        # -------------Low Level Plan Section-------------------
        # Get the low level plan using Q learning, and return a low level action at last.
        # A low level action is defined in Directions, whihc include {"North", "South", "East", "West", "Stop"}

        if not self.posSatisfyLowLevelPlan(gameState):
            # TODO just hardcoding defend for now
            self.lowLevelPlan = self.getLowLevelPlanHS(gameState, "defend")
            if self.debug:
                print(f"Agent {self.index} chose {self.lowLevelPlan[0][0]}")
            self.lowLevelActionIndex = 0

        # Safety check in case plan is still empty
        if not self.lowLevelPlan or self.lowLevelActionIndex >= len(self.lowLevelPlan):
            if self.debug:
                print(f"Agent {self.index}: Empty plan, returning STOP")
            return Directions.STOP

        lowLevelAction = self.lowLevelPlan[self.lowLevelActionIndex][0]
        self.lowLevelActionIndex += 1

        # TODO: I removed the stop/reverse counter here, not sure if we still want it

        # print("\tAgent:", self.index,lowLevelAction)
        return lowLevelAction

    # ------------------------------- PDDL and High-Level Action Functions -------------------------------

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

    def get_pddl_state(self, gameState: GameState) -> Tuple[List[Tuple], List[Tuple]]:
        """
        This function collects pddl :objects and :init states from simulator gameState.
        """
        pass

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

    def getGoals(self, objects: List[Tuple], initState: List[Tuple]):
        """
        Multi-tiered goal prioritization system based on game state.
        Returns positive and negative goal states for PDDL planning.
        """
        pass

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

    @profile
    def lowLevelDefault(self, gameState: GameState, actions):
        pos = gameState.getAgentPosition(self.index)
        max_positive_count = -1
        best_next = []
        valid_moves = []

        # Cache topology attributes to avoid repeated attribute lookups
        junctions = MixedAgent.MAP_TOPOLOGY.junctions
        dead_end_zones = MixedAgent.MAP_TOPOLOGY.dead_end_zones
        tile_to_corridor = MixedAgent.MAP_TOPOLOGY.tile_to_corridor
        corridors = MixedAgent.MAP_TOPOLOGY.corridors

        # Find closest food where we have advantage > 1 and are closer than teammate
        target_food = None
        target_food_is_capsule = False
        min_food_dist = float("inf")
        current_advantages = self.advantages
        teammate_index = (self.index + 2) % 4
        teammate_pos = gameState.getAgentPosition(teammate_index)

        for food in self.getFood():
            my_advantage = self.get_advantage(
                food, gameState, current_advantages, teammate=self.index
            )

            # Check if we have advantage > 1 (closer than enemies by at least 1)
            if my_advantage >= 1:
                my_dist = self.getMazeDistance(pos, food)
                teammate_dist = (
                    self.getMazeDistance(teammate_pos, food)
                    if teammate_pos
                    else float("inf")
                )

                # Also require being closer than teammate
                if my_dist < teammate_dist and my_dist < min_food_dist:
                    min_food_dist = my_dist
                    target_food = food

        for food in self.getCapsules():
            my_advantage = self.get_advantage(
                food, gameState, current_advantages, teammate=self.index
            )

            # Check if we have advantage > 1 (closer than enemies by at least 1)
            if my_advantage >= 1:
                my_dist = self.getMazeDistance(pos, food)
                teammate_dist = (
                    self.getMazeDistance(teammate_pos, food)
                    if teammate_pos
                    else float("inf")
                )

                # Also require being closer than teammate
                if my_dist < teammate_dist and (
                    not target_food_is_capsule or my_dist < min_food_dist
                ):
                    min_food_dist = my_dist
                    target_food = food
                    target_food_is_capsule = True

        if self.debug:
            print(f"Target food for Agent {self.index} is {target_food}")

        for action, advantages in actions:
            successor = self.getSuccessor(gameState, action)
            is_pacman = successor.getAgentState(self.index).isPacman

            succ_pos = successor.getAgentPosition(self.index)

            if not is_pacman:
                # Can kill
                if any(
                    succ_pos == gameState.getAgentPosition(i)
                    for i in self.getOpponents(gameState)
                ):
                    return [(action, successor.getAgentPosition(self.index))]

                # Can trap enemy
                for i in self.getOpponents(gameState):
                    if gameState.getAgentState(i).scaredTimer > 0:
                        continue
                    opp_pos = successor.getAgentPosition(i)
                    if isinstance(opp_pos, tuple) and opp_pos in dead_end_zones:
                        exit_pos = dead_end_zones[opp_pos]

                        # We are at the exit already or in a dead end
                        if pos == exit_pos or (
                            pos in dead_end_zones and dead_end_zones[pos] == exit_pos
                        ):
                            # If corridor, follow corridor
                            if opp_pos in tile_to_corridor or (
                                opp_pos in junctions
                                and junctions[opp_pos].junction_type == "dead_end"
                            ):
                                if opp_pos in junctions:
                                    target = junctions[opp_pos].pos
                                else:
                                    corridor_id = tile_to_corridor[opp_pos]
                                    corridor = corridors[corridor_id]
                                    junction_a = junctions[corridor.junction_a]
                                    junction_b = junctions[corridor.junction_b]
                                    target = (
                                        junction_a.pos
                                        if junction_a.junction_type == "dead_end"
                                        else junction_b.pos
                                    )

                                current_distance = self.getMazeDistance(pos, target)
                                next_distance = self.getMazeDistance(succ_pos, target)
                                if next_distance < current_distance:
                                    return [(action, succ_pos)]
                            else:
                                # Stay put
                                return [
                                    (
                                        Directions.STOP,
                                        gameState.getAgentPosition(self.index),
                                    )
                                ]
                        # Check if we can reach the exit before the opponent
                        elif self.getMazeDistance(
                            succ_pos, exit_pos
                        ) <= self.getMazeDistance(opp_pos, exit_pos):
                            if self.getMazeDistance(
                                succ_pos, exit_pos
                            ) < self.getMazeDistance(pos, exit_pos):
                                return [
                                    (
                                        action,
                                        succ_pos,
                                    )
                                ]

            if is_pacman:
                # # Skip positions where we'll be trapped
                if succ_pos in MixedAgent.MAP_TOPOLOGY.dead_end_zones:
                    exit_pos = MixedAgent.MAP_TOPOLOGY.dead_end_zones[succ_pos]
                    skip_action = False
                    for opp in self.getOpponents(gameState):
                        my_advantage = self.get_advantage(exit_pos, gameState, advantages, teammate=self.index)
                        if my_advantage <= 1:
                            skip_action = True
                    if skip_action:
                        if self.debug:
                            print(
                                f"Skipping action {action} for Agent {self.index} (trapped)"
                            )
                        continue

                # Skip positions that lose us access to escape points
                if not target_food_is_capsule:
                    escape_points = self.getEscapePoints()
                    can_escape = False
                    for escape in escape_points:
                        if (
                            self.get_advantage(
                                escape, gameState, advantages, teammate=self.index
                            )
                            >= 1
                        ):
                            can_escape = True
                            break
                    if not can_escape:
                        if self.debug:
                            print(
                                f"Skipping action {action} for Agent {self.index} (lose escape)"
                            )
                        continue

            valid_moves.append((action, succ_pos))

            # If we have a priority target food, check if this action gets us closer
            if target_food is not None:
                current_dist = self.getMazeDistance(pos, target_food)
                next_dist = self.getMazeDistance(succ_pos, target_food)
                if self.debug:
                    print(
                        f"Action {action} old dist: {current_dist}, new dist: {next_dist}"
                    )
                if next_dist < current_dist:
                    return [(action, succ_pos)]

            positive_count = 0
            total_advantage = 0

            for food in self.getFood():
                advantage = self.get_advantage(food, gameState, advantages)
                positive_count += int(advantage > 0)
                total_advantage += advantage

            for capsule in self.getCapsules():
                advantage = self.get_advantage(capsule, gameState, advantages)
                positive_count += int(advantage > 0) * 40
                total_advantage += advantage

            # Check if any enemies are carrying food
            for opp in self.getOpponents(gameState):
                opp_state = gameState.getAgentState(opp)
                if opp_state.numCarrying > 0:
                    # Check if they are trapped in our territory
                    escape_points = self.getEscapePointsYouAreDefending()
                    trapped = True
                    for escape in escape_points:
                        if (
                            self.get_advantage(escape, gameState, advantages, enemy=opp)
                            <= 0
                        ):
                            trapped = False
                            break
                    if not trapped:
                        positive_count -= opp_state.numCarrying

            # order of priority: maximize number of non-neg adv -> maximize adv
            if positive_count > max_positive_count:  # or \
                # (positive_count == max_positive_count and total_advantage > max_advantage):
                best_next = [
                    (
                        action,
                        successor.getAgentPosition(self.index),
                        total_advantage,
                    )
                ]
                max_positive_count = positive_count
            elif (
                positive_count == max_positive_count
            ):  # and total_advantage == max_advantage:
                best_next.append(
                    (
                        action,
                        successor.getAgentPosition(self.index),
                        total_advantage,
                    )
                )

        if len(best_next) == 0:
            if len(valid_moves) == 0:
                return [(actions[0][0], pos)]

            chosen = random.choice(valid_moves)
            return [chosen]

        actual_best = best_next[0]
        width, height = self.walls.width, self.walls.height
        uncertain = lambda x: x < 1 and x > 0.01
        if len(best_next) > 1:
            best_min_dist = 9999
            ops = self.getOpponents(gameState)
            enemy_poss = [
                (x, y)
                for x in range(width)
                for y in range(height)
                if uncertain(
                    max(
                        MixedAgent.OPPONENT_BELIEFS[ops[0]][x][y],
                        MixedAgent.OPPONENT_BELIEFS[ops[1]][x][y],
                    )
                )
            ]
            if len(enemy_poss) == 0:
                max_adv = max(max(adv for _, _, adv in best_next), 1e-6)
                weights = [adv / max_adv for _, _, adv in best_next]
                actual_best = random.choices(best_next, weights=weights, k=1)[0]
                return [actual_best[:2]]

            for action, next_pos, adv in best_next:
                min_dist = min(
                    self.getMazeDistance(next_pos, e_pos) for e_pos in enemy_poss
                )
                if min_dist <= best_min_dist:
                    actual_best = (action, next_pos)
                    best_min_dist = min_dist

        return [actual_best[:2]]

    @profile
    def lowLevelEscape(self, gameState: GameState, actions):
        pos = gameState.getAgentPosition(self.index)
        best_next = (actions[0][0], pos)

        min_escape_dist = min(
            (
                self.getMazeDistance(pos, escape)
                for escape in self.getEscapePoints()
                if self.get_advantage(
                    escape, gameState, self.advantages, teammate=self.index
                )
                > 1
            ),
            default=float("inf"),
        )

        for action, advantages in actions:
            successor = self.getSuccessor(gameState, action)
            succ_pos = successor.getAgentPosition(self.index)

            if self.debug:
                print(
                    f"Agent {self.index} is at {succ_pos}, at home: {self.isInHome(succ_pos)}"
                )
            if self.isInHome(succ_pos):
                return [(action, succ_pos)]

            # Skip positions that lose us access to escape points
            escape_points = self.getEscapePoints()
            escapable_points = []
            positive_count = 0
            for escape in escape_points:
                advantage = self.get_advantage(
                    escape, gameState, advantages, teammate=self.index
                )
                if advantage > 1:
                    positive_count += 1
                    escapable_points.append(escape)

            if self.debug:
                print(f"Agent {self.index} has {positive_count} positive escapes")
            if positive_count > 0:
                new_min_escape = min(
                    (
                        self.getMazeDistance(escape, succ_pos)
                        for escape in escapable_points
                    ),
                    default=float("inf"),
                )
                if self.debug:
                    print(
                        f"Agent {self.index} was {min_escape_dist} but now {new_min_escape} away"
                    )

                if new_min_escape < min_escape_dist:
                    if self.debug:
                        print(f"Updating best action to {action}")
                    best_next = (action, successor.getAgentPosition(self.index))

        return [best_next]

    @profile
    def lowLevelDefend(self, gameState: GameState, actions):
        pos = gameState.getAgentPosition(self.index)
        max_positive_count = -1
        best_next = []
        op_holdings = tuple(
            (idx, gameState.getAgentState(idx).numCarrying)
            for idx in self.getOpponents(gameState)
        )
        def_enemy_pos = tuple(
            gameState.getAgentPosition(i)
            for i in self.getOpponents(gameState)
            if gameState.getAgentPosition(i) is not None
        )

        # Cache topology attributes to avoid repeated attribute lookups
        junctions = MixedAgent.MAP_TOPOLOGY.junctions
        dead_end_zones = MixedAgent.MAP_TOPOLOGY.dead_end_zones
        tile_to_corridor = MixedAgent.MAP_TOPOLOGY.tile_to_corridor
        corridors = MixedAgent.MAP_TOPOLOGY.corridors

        for action, advantages in actions:
            successor = self.getSuccessor(gameState, action)
            def_enemy_pos = tuple(
                successor.getAgentPosition(i)
                for i in self.getOpponents(successor)
                if successor.getAgentPosition(i) is not None
            )
            next_pos = successor.getAgentPosition(self.index)

            trapped = all(
                self.get_advantage(border, successor, advantages, teammate=self.index)
                < 2
                for border in self.getEscapePoints()
            )
            trapped = trapped or (
                min(
                    (self.getMazeDistance(next_pos, e_pos) for e_pos in def_enemy_pos),
                    default=999,
                )
                <= 1
            )

            if gameState.getAgentState(self.index).isPacman and trapped:
                continue

            succ_pos = successor.getAgentPosition(self.index)

            ###########
            if any(
                succ_pos == gameState.getAgentPosition(i)
                for i in self.getOpponents(gameState)
            ) and self.isInHome(succ_pos):
                return [(action, successor.getAgentPosition(self.index))]

            if self.isInHome(succ_pos):
                for i in self.getOpponents(gameState):
                    opp_pos = successor.getAgentPosition(i)
                    if isinstance(opp_pos, tuple) and opp_pos in dead_end_zones:
                        exit_pos = dead_end_zones[opp_pos]

                        # We are at the exit already or in a dead end
                        if pos == exit_pos or (
                            pos in dead_end_zones and dead_end_zones[pos] == exit_pos
                        ):
                            # If corridor, follow corridor
                            if opp_pos in tile_to_corridor or (
                                opp_pos in junctions
                                and junctions[opp_pos].junction_type == "dead_end"
                            ):
                                if opp_pos in junctions:
                                    target = junctions[opp_pos].pos
                                else:
                                    corridor_id = tile_to_corridor[opp_pos]
                                    corridor = corridors[corridor_id]
                                    junction_a = junctions[corridor.junction_a]
                                    junction_b = junctions[corridor.junction_b]
                                    target = (
                                        junction_a.pos
                                        if junction_a.junction_type == "dead_end"
                                        else junction_b.pos
                                    )

                                current_distance = self.getMazeDistance(pos, target)
                                next_distance = self.getMazeDistance(succ_pos, target)
                                if next_distance < current_distance:
                                    return [(action, succ_pos)]
                            else:
                                # Stay put
                                return [
                                    (
                                        Directions.STOP,
                                        gameState.getAgentPosition(self.index),
                                    )
                                ]
                        # Check if we can reach the exit before the opponent
                        elif self.getMazeDistance(
                            succ_pos, exit_pos
                        ) <= self.getMazeDistance(opp_pos, exit_pos):
                            if self.getMazeDistance(
                                succ_pos, exit_pos
                            ) < self.getMazeDistance(pos, exit_pos):
                                return [
                                    (
                                        action,
                                        succ_pos,
                                    )
                                ]
            #########
            # advantages = self.calculate_advantages(successor)

            positive_count = 0

            for food in self.getFoodYouAreDefending():
                positive_count += int(
                    self.get_advantage(food, gameState, advantages) >= 0
                )

            for capsule in self.getCapsules():
                advantage = self.get_advantage(capsule, gameState, advantages)
                positive_count += int(advantage > 0) * 40

            # we need to consider guarding the border when enemy is holding
            if any(holding > 0 for _, holding in op_holdings):
                for op, holding in op_holdings:
                    if all(
                        self.get_advantage(border, successor, advantages, enemy=op) >= 0
                        for border in self.getEscapePoints()
                    ):
                        positive_count += holding

            if positive_count > max_positive_count:
                best_next = [
                    (
                        action,
                        successor.getAgentPosition(self.index),
                    )
                ]
                max_positive_count = positive_count
            elif (
                positive_count == max_positive_count
            ):  # and total_advantage == max_advantage:
                best_next.append(
                    (
                        action,
                        successor.getAgentPosition(self.index),
                    )
                )

        # Safety check: if no valid moves found, return STOP
        if len(best_next) == 0:
            return [(Directions.STOP, pos)]

        actual_best = best_next[0]
        width, height = self.walls.width, self.walls.height
        uncertain = lambda x: x < 1 and x > 0.02
        if len(best_next) > 1:
            best_min_dist = 9999
            ops = self.getOpponents(gameState)
            enemy_poss = [
                (x, y)
                for x in range(width)
                for y in range(height)
                if uncertain(
                    max(
                        MixedAgent.OPPONENT_BELIEFS[ops[0]][x][y],
                        MixedAgent.OPPONENT_BELIEFS[ops[1]][x][y],
                    )
                )
            ]
            if len(enemy_poss) == 0:
                actual_best = random.choice(best_next)
                return [actual_best]
            for action, next_pos in best_next:
                teammate_pos = gameState.getAgentPosition(self.index)
                if teammate_pos == next_pos:
                    continue

                min_dist = min(
                    self.getMazeDistance(next_pos, e_pos) for e_pos in enemy_poss
                )
                if min_dist <= best_min_dist:
                    actual_best = (action, next_pos)
                    best_min_dist = min_dist

        return [actual_best]

    @profile
    def getLowLevelPlanHS(
        self, gameState: GameState, highLevelAction: str
    ) -> List[Tuple[str, Tuple]]:
        score = gameState.getScore()
        score = score if self.red else -score
        winning = True if score > 0 else False
        is_pacman = gameState.getAgentState(self.index).isPacman
        total_holding = (
            gameState.getAgentState(self.index).numCarrying
            + gameState.getAgentState((self.index + 2) % 4).numCarrying
        )
        # Find any opponent agents that are carrying and have access to our border
        for opp in self.getOpponents(gameState):
            if gameState.getAgentState(opp).numCarrying == 0:
                continue
            escape_points = self.getEscapePointsYouAreDefending()
            trapped = True
            for escape in escape_points:
                if self.get_advantage(escape, gameState, self.advantages, enemy=opp) > 1:
                    trapped = False
                    break

            if not trapped:
                total_holding -= gameState.getAgentState(opp).numCarrying

        can_win = total_holding + score > 0

        # Check if we have advantage over our capsules
        capsule_advantage = True
        for capsule in self.getCapsules():
            advantage = self.get_advantage(capsule, gameState, self.advantages)
            if advantage < 0:
                capsule_advantage = False
                break


        legalActions = gameState.getLegalActions(self.index)
        filteredActions = []
        for action in legalActions:
            successor = self.getSuccessor(gameState, action)
            advantages = self.calculate_advantages(successor)
            succ_pos = successor.getAgentPosition(self.index)
            is_pacman = successor.getAgentState(self.index).isPacman
            is_scared_ghost = (
                successor.getAgentState(self.index).scaredTimer > 0 and not is_pacman
            )

            # Skip if we die this turn
            if succ_pos == self.startPosition:
                continue

            # Skip if we are too close to the enemy next turn
            skip_action = False
            for opp in self.getOpponents(gameState):
                opp_pos = gameState.getAgentPosition(opp)
                opp_pacman = gameState.getAgentState(opp).isPacman
                opp_scared_ghost = gameState.getAgentState(opp).scaredTimer > 0

                if (
                    opp_pos is not None
                    and self.getMazeDistance(succ_pos, opp_pos) <= 1
                    and (
                        (opp_pacman and is_scared_ghost)
                        or (is_pacman and not opp_pacman and not opp_scared_ghost)
                    )
                ):
                    skip_action = True
                    break
            if skip_action:
                continue

            # Skip if we lose advantage over our capsules
            new_capsule_advantage = True
            for capsule in self.getCapsules():
                advantage = self.get_advantage(capsule, gameState, advantages)
                if advantage < 0:
                    new_capsule_advantage = False
                    break

            if capsule_advantage and not new_capsule_advantage:
                continue

            filteredActions.append((action, advantages))

        if len(filteredActions) == 0:
            return [(Directions.STOP, gameState.getAgentPosition(self.index))]

        if can_win and is_pacman:
            highLevelAction = "escape"

        elif not winning:
            highLevelAction = "default"

        if self.debug:
            print(f"Agent {self.index} chose {highLevelAction}")
            print(f"Legal actions:     {legalActions}")
            print(f"Potential actions: {[act[0] for act in filteredActions]}")

        if highLevelAction == "default":
            return self.lowLevelDefault(gameState, filteredActions)
        elif highLevelAction == "escape":
            return self.lowLevelEscape(gameState, filteredActions)
        else:
            return self.lowLevelDefend(gameState, filteredActions)

    # ==================== Shared Helper Functions ====================
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

    @profile
    def calculate_advantages(self, gameState: GameState, beliefs: dict = None):
        """
        Finds the current advantages at each cell: {(x, y) : (agent_1 dist, agent_2 dist, ...))
        """
        walls = gameState.getWalls()
        width, height = walls.width, walls.height

        if not beliefs:
            beliefs = MixedAgent.OPPONENT_BELIEFS

        advantages = (
            {}
        )  # {(junct_x, junct_y) : (((ally_idx, dist), (ally2_idx, dist)), num_adv)}
        enemy_pos = {}
        for op in self.getOpponents(gameState):
            belief = beliefs[op]
            enemy_pos[op] = set()
            for x in range(width):
                for y in range(height):
                    if belief[x][y]:
                        enemy_pos[op].add((x, y))

        for x in range(width):
            for y in range(height):
                node_pos = (x, y)
                if not walls[x][y]:
                    advantages[node_pos] = []
                    distance_to_exit = 0
                    if node_pos in MixedAgent.MAP_TOPOLOGY.dead_end_zones:
                        distance_to_exit = (
                            self.getMazeDistance(
                                node_pos,
                                MixedAgent.MAP_TOPOLOGY.dead_end_zones[node_pos],
                            )
                            + 1
                        )

                    for idx in range(4):
                        if idx in self.getTeam(gameState):
                            distance = self.getMazeDistance(
                                gameState.getAgentPosition(idx), node_pos
                            ) + (
                                distance_to_exit
                                if self.isInEnemyTerritory(node_pos)
                                else 0
                            )
                            if gameState.getAgentState(
                                idx
                            ).scaredTimer > distance and self.isInEnemyTerritory(
                                node_pos
                            ):
                                distance = gameState.getAgentState(idx).scaredTimer
                            advantages[node_pos].append(distance)
                        else:
                            min_enemy_dist = min(
                                self.getMazeDistance(pos, node_pos)
                                for pos in enemy_pos[idx]
                            )
                            distance = min_enemy_dist + (
                                distance_to_exit if self.isInHome(node_pos) else 0
                            )
                            if gameState.getAgentState(
                                idx
                            ).scaredTimer > distance and self.isInHome(node_pos):
                                distance = gameState.getAgentState(idx).scaredTimer

                            advantages[node_pos].append(distance)

        # # Adjust for trapped agents - detect who is trapped and in which zone
        # trapped_in_zone = {}  # {agent_idx: zone_exit_pos}
        #
        # for idx in range(4):
        #     if idx in self.getTeam(gameState):
        #         pos = gameState.getAgentPosition(idx)
        #         if pos in MixedAgent.MAP_TOPOLOGY.dead_end_zones:
        #             zone_exit = MixedAgent.MAP_TOPOLOGY.dead_end_zones[pos]
        #             # Check if exit is blocked - advantage <= 1 means enemy can get there before/same time
        #             if (
        #                 self.get_advantage(
        #                     zone_exit, gameState, advantages, teammate=idx
        #                 )
        #                 <= 1
        #             ):
        #                 print(f"trapped agent {idx} in zone {zone_exit}")
        #                 trapped_in_zone[idx] = zone_exit
        #     else:
        #         # Check if enemy is trapped - all their possible positions must be in same zone
        #         zone_exit = None
        #         all_in_same_zone = True
        #         for pos in enemy_pos[idx]:
        #             if pos not in MixedAgent.MAP_TOPOLOGY.dead_end_zones:
        #                 all_in_same_zone = False
        #                 break
        #             current_exit = MixedAgent.MAP_TOPOLOGY.dead_end_zones[pos]
        #             if zone_exit is None:
        #                 zone_exit = current_exit
        #             elif zone_exit != current_exit:
        #                 all_in_same_zone = False
        #                 break
        #
        #         if all_in_same_zone and zone_exit is not None:
        #             # Check if exit is blocked - advantage <= 1 means we can get there before/same time
        #             if (
        #                 self.get_advantage(zone_exit, gameState, advantages, enemy=idx)
        #                 >= 1
        #             ):
        #                 trapped_in_zone[idx] = zone_exit
        #
        # # Update advantages for trapped agents - set distance to infinity for positions outside their zone
        # for idx, zone_exit in trapped_in_zone.items():
        #     for node_pos in advantages:
        #         # If node_pos is outside the trapped agent's zone, they can't reach it
        #         if (
        #             node_pos not in MixedAgent.MAP_TOPOLOGY.dead_end_zones
        #             or MixedAgent.MAP_TOPOLOGY.dead_end_zones[node_pos] != zone_exit
        #         ):
        #             advantages[node_pos][idx] = float("inf")

        return advantages

    def get_advantage(
        self, pos, gameState, advantages, teammate=None, enemy=None, max_lookahead=50
    ):
        if not teammate:
            teammate = self.getTeam(gameState)
        else:
            teammate = (teammate,)
        if not enemy:
            enemy = self.getOpponents(gameState)
        else:
            enemy = (enemy,)

        op_min_dist = min(advantages[pos][op] for op in enemy)
        adv = op_min_dist - min(advantages[pos][tm] for tm in teammate)
        if op_min_dist > max_lookahead:
            return 1
        return adv

    def update_critical_junctions(self, gameState: GameState):
        """Update cached junction sets for food and capsules when counts change."""
        MixedAgent.RED_FOOD = gameState.getRedFood().asList()
        MixedAgent.RED_CAPSULES = gameState.getRedCapsules()
        MixedAgent.BLUE_FOOD = gameState.getBlueFood().asList()
        MixedAgent.BLUE_CAPSULES = gameState.getBlueCapsules()

    def getFood(self):
        if self.red:
            return MixedAgent.BLUE_FOOD
        else:
            return MixedAgent.RED_FOOD

    def getFoodYouAreDefending(self):
        if self.red:
            return MixedAgent.RED_FOOD
        else:
            return MixedAgent.BLUE_FOOD

    def getCapsules(self):
        if self.red:
            return MixedAgent.BLUE_CAPSULES
        else:
            return MixedAgent.RED_CAPSULES

    def getCapsulesYouAreDefending(self):
        if self.red:
            return MixedAgent.RED_CAPSULES
        else:
            return MixedAgent.BLUE_CAPSULES

    def initializeEscapePoints(self):
        width, height = self.walls.width, self.walls.height
        MixedAgent.RED_ESCAPE_POINTS = []
        MixedAgent.BLUE_ESCAPE_POINTS = []

        red_x_border = width // 2 - 1
        blue_x_border = width // 2

        for y in range(height):
            if not self.walls[red_x_border][y]:
                MixedAgent.RED_ESCAPE_POINTS.append((red_x_border, y))

            if not self.walls[blue_x_border][y]:
                MixedAgent.BLUE_ESCAPE_POINTS.append((blue_x_border, y))

    def getEscapePoints(self):
        if self.red:
            return MixedAgent.BLUE_ESCAPE_POINTS
        else:
            return MixedAgent.RED_ESCAPE_POINTS

    def getEscapePointsYouAreDefending(self):
        if self.red:
            return MixedAgent.RED_ESCAPE_POINTS
        else:
            return MixedAgent.BLUE_ESCAPE_POINTS

    def isInEnemyTerritory(self, pos):
        """Check if position is in enemy territory"""
        walls = self.walls
        width = walls.width
        x = int(pos[0])
        if self.red:
            return x >= width // 2
        else:
            return x < width // 2

    def isInHome(self, pos):
        """Check if position is in home territory"""
        return not self.isInEnemyTerritory(pos)


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

    # 3. Dead-end zone exit locations map
    exit_tiles = {}

    # Find all unique exit junctions
    unique_exits = set(topology.dead_end_zones.values())

    for pos in unique_exits:
        exit_tiles[pos] = "X"  # Exit junction

    # Show other junctions for context
    for pos, junction in topology.junctions.items():
        if pos not in exit_tiles:
            exit_tiles[pos] = "."  # Regular junction

    print(
        create_map_string(
            exit_tiles, "DEAD-END ZONE EXITS (X=exit from dead-end, .=junction)"
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

    Simplified approach: For each articulation point, temporarily remove it and
    do a tile-level flood fill from the map center. Tiles that become unreachable
    form a dead-end zone with that articulation point as the exit.

    Returns:
        Dict mapping tile_pos -> exit_junction_pos
    """
    dead_end_zones = {}

    # Find a safe starting position (center of the map, adjusted to non-wall)
    center_x, center_y = width // 2, height // 2
    start_pos = None
    for dx in range(max(width, height)):
        for sx, sy in [
            (center_x + dx, center_y),
            (center_x - dx, center_y),
            (center_x, center_y + dx),
            (center_x, center_y - dx),
        ]:
            if 0 <= sx < width and 0 <= sy < height and not walls[sx][sy]:
                start_pos = (sx, sy)
                break
        if start_pos:
            break

    if not start_pos:
        return dead_end_zones  # No valid starting position

    # For each articulation point, find what becomes unreachable without it
    # Store all zones temporarily to process them later
    temp_zones = {}  # art_point -> list of unreachable tiles

    for art_point in articulation_points:
        # Flood fill from start_pos, treating art_point as a wall
        reachable = set()
        queue = [start_pos]

        while queue:
            current_pos = queue.pop(0)

            if current_pos in reachable:
                continue

            # Skip the articulation point itself
            if current_pos == art_point:
                continue

            reachable.add(current_pos)

            # Explore neighbors
            neighbors = Actions.getLegalNeighbors(current_pos, walls)
            for neighbor in neighbors:
                if neighbor not in reachable and neighbor != art_point:
                    queue.append(neighbor)

        # Any non-wall tile that's not reachable is in a dead-end zone
        unreachable_tiles = []
        for x in range(width):
            for y in range(height):
                pos = (x, y)
                if not walls[x][y] and pos not in reachable:
                    unreachable_tiles.append(pos)

        # Only mark this as a dead-end zone if it's small (not the main map)
        total_non_wall_tiles = sum(
            1 for x in range(width) for y in range(height) if not walls[x][y]
        )

        if unreachable_tiles and len(unreachable_tiles) < total_non_wall_tiles * 0.5:
            # Store this zone temporarily
            temp_zones[art_point] = unreachable_tiles

    # Now process zones: only keep zones whose exit is NOT trapped in another zone
    # This eliminates nested zones (only keep the outermost zone)
    for art_point, zone_tiles in temp_zones.items():
        # Check if this articulation point (exit) is trapped in any other zone
        is_trapped = False
        for other_art_point, other_zone_tiles in temp_zones.items():
            if other_art_point != art_point and art_point in other_zone_tiles:
                # This exit is trapped in another zone - skip it
                is_trapped = True
                break

        if not is_trapped:
            # This is a valid outermost zone - mark all tiles
            for tile_pos in zone_tiles:
                if tile_pos not in dead_end_zones:
                    dead_end_zones[tile_pos] = art_point

            # Also mark the articulation point itself
            if art_point not in dead_end_zones:
                dead_end_zones[art_point] = art_point

    return dead_end_zones


def initialize_beliefs(
    game_state: GameState, agents=None
) -> Dict[int, List[List[float]]]:
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

    if not agents:
        agents = range(4)

    beliefs = {}

    for agent_idx in agents:
        # Create empty probability array
        prob_array = [[0.0 for _ in range(height)] for _ in range(width)]

        # Get agent's exact starting position
        agent_pos = game_state.getAgentState(agent_idx).start.pos
        x, y = int(agent_pos[0]), int(agent_pos[1])
        prob_array[x][y] = 1.0  # Certain they're at starting position

        beliefs[agent_idx] = prob_array

    return beliefs


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

    assert (
        sum(sum(i) for i in prev_belief) > 0.5
    ), f"lost track of opponent {opponent_idx}"

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
                teammate_dist = manhattanDistance(
                    game_state.getAgentPosition((observer_idx + 2) % 4), (x, y)
                )
                if (
                    min(true_dist, teammate_dist) > 5
                ):  # we know that the position is not in our sight range
                    updated_belief[x][y] = prior * likelihood

    # Step 4: Normalize
    total_prob = sum(sum(row) for row in updated_belief)
    if total_prob > 0:
        for x in range(width):
            for y in range(height):
                updated_belief[x][y] /= total_prob
    else:  # we lost track of opponent. this (probably) always means that they just
        # died. so we reinitialize beliefs for this opponent
        updated_belief = initialize_beliefs(game_state, [opponent_idx])[opponent_idx]
        # need to propagate from their spawn if they just moved
        prev_agent_idx = (observer_idx - 1) % 4

        if opponent_idx == prev_agent_idx:
            new_beliefs = [[0.0 for _ in range(height)] for _ in range(width)]

            for x in range(width):
                for y in range(height):
                    if updated_belief[x][y] > 0:
                        # From position (x,y), distribute probability to reachable neighbors
                        neighbors = Actions.getLegalNeighbors((x, y), walls)
                        prob_per_neighbor = updated_belief[x][y] / len(neighbors)

                        for nx, ny in neighbors:
                            new_beliefs[nx][ny] += prob_per_neighbor

            updated_belief = new_beliefs

    return updated_belief


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


# ====================SEARCH STUFF==========================================

from lib_piglet.utils.tools import eprint
from lib_piglet.search import (
    tree_search,
    graph_search,
    base_search,
    search_node,
    iterative_deepening,
    graph_search_anytime,
)
from lib_piglet.utils.data_structure import queue, stack, bin_heap
from lib_piglet.expanders.base_expander import base_expander
from lib_piglet.solution.solution import solution_to_state_list

debug = False


def getPossibleActions(pos, walls):
    possible = []
    x, y = pos
    x_int, y_int = int(x + 0.5), int(y + 0.5)

    for dir, vec in Actions._directionsAsList:
        dx, dy = vec
        next_y = y_int + dy
        next_x = x_int + dx
        if not walls[next_x][next_y]:
            possible.append(dir)

    return possible


class PacAct:
    def __init__(self, cost):
        self.cost_ = cost


# The search state is a tuple of location and interception time from path start
# (x, y, int), with the default value for int being -1


class pacman_expander(base_expander):
    def __init__(self, gameState: GameState, agent: MixedAgent, max_timestep=9999):
        self.domain_ = agent
        self.domain_.is_goal = lambda node, goals: any(
            all(node[i] == goal[i] for i in range(2)) for goal in goals
        )
        self.gameState = gameState
        self.succ_: list = []
        self.max_timestep = max_timestep
        self.verbose = False

    def expand(self, current_node: search_node):
        self.succ_.clear()
        state = current_node.state_
        x, y, inter = state
        pos = (x, y)
        walls = self.gameState.getWalls()
        successors_acts = []

        valid_transitions = getPossibleActions(pos, walls)
        if self.verbose:
            print(f"currently at cell {state[:2]}")

        for act in valid_transitions:
            new_pos = Actions.getSuccessor(pos, act)
            if self.verbose:
                print(f"    looking at transition to {new_pos}")

            # we need to check for interception
            adv = self.domain_.get_advantage(
                new_pos, self.gameState, MixedAgent.CURRENT_ADVANTAGES
            )
            # intercept time is -1 until an interception is detected.
            # subsequent successors will all have the same intercept time

            # we still have adv / we've already been intercepted
            if adv > 0 or self.domain_.isInHome(new_pos) or inter >= 0:
                successors_acts.append(((*new_pos, inter), PacAct(1)))
            # we just got intercepted
            else:
                successors_acts.append(
                    ((*new_pos, current_node.g_ + (adv // 2) + 1), PacAct(1))
                )
        else:
            successors_acts.append(((*new_pos, inter), PacAct(1)))

        return successors_acts

    def __str__(self):
        return self.domain_.domain_file_


# = true_maze_dist + (99999 - intercept_time) IF intercept time >= 0
# ELSE = true_maze_dist
# should be used when finding food, calculating escape path, etc...
# ALWAYS PASS IN AN ITERABLE OF GOAL STATES
def offensive_heuristic(domain: MixedAgent, current_state, goal_state):
    agent = domain
    dist = min(
        agent.getMazeDistance(current_state[:2], g_state[:2]) for g_state in goal_state
    )
    inter_time = current_state[-1]
    if inter_time >= 0:
        dist += 99999999 - inter_time * 500
    return dist


from collections.abc import Iterable


def get_path(
    start: tuple,
    goals: Iterable,
    agent: MixedAgent,
    gameState: GameState,
    heuristic,
    max_timestep: int = None,
):
    """Returns path and time until intercept. time = -1 if not intercepted"""
    if not hasattr(get_path, "pac_searcher"):
        open_lst = bin_heap(search_node.compare_node_f)
        get_path.pac_searcher = graph_search.graph_search(
            open_lst,
            expander=pacman_expander(gameState, agent),
            heuristic_function=heuristic,
        )
    else:
        get_path.pac_searcher.expander_.domain_ = agent
        get_path.pac_searcher.expander_.gameState = gameState

    assert isinstance(goals[0], Iterable), "goals should be an iterable of state tuples"

    path = get_path.pac_searcher.get_path((*start, -1), goals)
    assert path is not None, "didn't find any valid paths? this should not happen"

    path = path.paths_
    print(path)
    intercept_t = path[-1].state_[-1]

    path = [state.state_[0:2] for state in path]
    print(get_path.pac_searcher.get_statistic())
    return path, intercept_t
