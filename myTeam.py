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
import random, time, util, sys, os
from capture import GameState, noisyDistance
from game import Configuration, Directions, Actions, AgentState, Agent
from util import nearestPoint, manhattanDistance
import sys, os
import pickle
from dataclasses import dataclass


@dataclass
class StateInfo:
    gameState: GameState
    agentState: AgentState
    teammateState: AgentState
    enemyVirtualStates: Dict[int, AgentState]


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

BREATHING_DISTANCE = 2
CLOSE_DISTANCE = 4
MEDIUM_DISTANCE = 15
LONG_DISTANCE = 25

LOW_SCARED_TIMER = 10

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
            "breathing-distance-ghosts": -10000,  # Priority 1 (less better) - CRITICAL: avoid ghosts
            "in-enemy-territory": 1000,  # Priority 2 (more better) - main goal
            "distance-to-enemy-territory": -100,  # Priority 3 (closer better) - guide to border
            "distance-to-teammate": 10,  # Priority 4 (further better) - spread out
            "stop": -1,
            "reverse": -1,
        },
        "eatFoodWeights": {
            "breathing-distance-ghosts": -100000,  # Priority 0 (less better) - CRITICAL: avoid ghosts
            # "in-enemy-territory": 10000,  # Priority 1 (more better) - MUST be in enemy territory
            "ate-food": 5000,  # Priority 2 - big reward for eating
            "distance-to-nearest-food": -100,  # Priority 3 (closer better) - guide to food
            "close-distance-ghosts": -10,  # Priority 4 (less better) - minor ghost avoidance
            "distance-to-attacking-teammate": 90,  # Priority 5 (closer better) - guide to attacking teammate
            "stop": -1,
            "reverse": -1,
        },
        "eatCapsuleWeights": {
            "breathing-distance-ghosts": -10000,  # Priority 1 (less better) - CRITICAL: avoid ghosts
            # "in-enemy-territory": 100000,  # Priority 0 (more better) - MUST be in enemy territory
            "ate-capsule": 1000,  # Priority 2 - big reward for eating capsule
            "distance-to-nearest-capsule": -100,  # Priority 3 (closer better) - guide to capsule
            "close-distance-ghosts": -10,  # Priority 4 (less better) - minor ghost avoidance
            "stop": -1,
            "reverse": -1,
        },
        "escapeWeights": {
            "in-home": 10000,  # Priority 1 - CRITICAL: get home safely
            "breathing-distance-ghosts": -1000,  # Priority 2 (less better) - avoid immediate danger
            "distance-to-home": -1000,  # Priority 2 - get home quickly
            "close-distance-ghosts": -100,  # Priority 3 (less better) - avoid nearby ghosts
            "stop": -1,
            "reverse": -1,
        },
        "chaseWeights": {
            # "in-home": 100000,  # Priority 1 - MUST stay in home territory
            "distance-to-enemy": -1000,  # Priority 2 (closer better) - get to target
            "between-enemy-and-escape": 100,  # Priority 3 - intercept bonus
            "distance-to-teammate": 10,  # Priority 4 (further better) - spread out
            "stop": -1,
            "reverse": -1,
        },
        "defaultDefendWeights": {
            "in-home": 100000,  # Priority 1 - MUST stay in home territory
            "distance-to-nearest-enemy": -1000,  # Priority 2 (closer better) - position near threats
            "distance-to-teammate": 10,  # Priority 3 (further better) - spread out
            "stop": -1,
            "reverse": -1,
        },
    }
    QLWeightsFile = BASE_FOLDER + "/QLWeightsMyTeam.pkl"

    # Also can use class variable to exchange information between agents.
    CURRENT_ACTION = {}
    ESTIMATED_POSITIONS = {}  # Cache for estimated enemy positions using beliefs

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

        # REMEMBER TRUN TRAINNING TO FALSE when submit to contest server.
        self.debug = True
        self.trainning = False  # trainning mode to true will keep update weights and generate random movements by prob.
        self.epsilon = 0.1  # default exploration prob, change to take a random step
        self.alpha = 0.02  # default learning rate
        self.discountRate = (
            0.9  # default discount rate on successor state q value when update
        )

        # Load learned weights if they exist
        try:
            if os.path.exists(MixedAgent.QLWeightsFile):
                with open(MixedAgent.QLWeightsFile, "rb") as f:
                    MixedAgent.QLWeights = pickle.load(f)
                if self.debug:
                    print(
                        f"Agent {self.index}: Loaded learned weights from {MixedAgent.QLWeightsFile}"
                    )
        except Exception as e:
            print(f"Agent {self.index}: Could not load weights: {e}")

        # Initialize belief tracking for opponents
        MixedAgent.OPPONENT_BELIEFS = initialize_beliefs(gameState)

        # Use a dictionary to save information about current agent.
        MixedAgent.CURRENT_ACTION[self.index] = {}

    def final(self, gameState: GameState):
        """
        This function write weights into files after the game is over.
        You may want to comment (disallow) this function when submit to contest server.
        """
        if self.trainning:
            # Only save from one agent to avoid race condition
            if self.index == 0:
                try:
                    with open(MixedAgent.QLWeightsFile, "wb") as f:
                        pickle.dump(MixedAgent.QLWeights, f)
                    if self.debug:
                        print(
                            f"Agent {self.index}: Saved learned weights to {MixedAgent.QLWeightsFile}"
                        )
                        print(
                            "Offensive weights:",
                            MixedAgent.QLWeights["offensiveWeights"],
                        )
                        print(
                            "Defensive weights:",
                            MixedAgent.QLWeights["defensiveWeights"],
                        )
                        print("Escape weights:", MixedAgent.QLWeights["escapeWeights"])
                except Exception as e:
                    print(f"Agent {self.index}: Could not save weights: {e}")

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
            highLevelAction = Action("defend", None, [], [], [], [])
        else:
            # Get next action from the plan
            highLevelAction = self.highLevelPlan[self.currentActionIndex][0]
        MixedAgent.CURRENT_ACTION[self.index] = highLevelAction

        if self.debug:
            print(f"Agent {self.index}: High-Level Action = {highLevelAction.name}")

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
        if team_score > 3:
            states.append(("winning_gt3",))
        if team_score < -3:
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
            if agent_state.numCarrying + positive_score > 3:
                states.append(("fat_agent_gt3", agent_obj))
            if agent_state.numCarrying + positive_score > 10:
                states.append(("fat_agent_gt10", agent_obj))

            if agent_state.isPacman:
                states.append(("is_pacman", agent_obj))

        # Check if there is any food nearby
        food = self.getFood(gameState).asList()
        if any(self.getMazeDistance(f, myPos) <= CLOSE_DISTANCE for f in food):
            states.append(("near_food", myObj))

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

            pos = agent_state.getPosition()
            assert pos

            # Check if we are further back
            if (self.red and myPos[0] < pos[0]) or (not self.red and myPos[0] > pos[0]):
                states.append(("further_back",))

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

            if my_distance < teammate_distance:
                states.append(("closer_to_enemy", enemy_object))

            typeIndex += 1

        return objects, states

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
            if is_scared:
                return self.goalAttack(objects, initState)

            # Priority 0: If both are pacman and enemy in our territory, handle multiple invaders
            if teammate_is_pacman:
                # Count enemies invading our territory
                invading_enemies = [
                    e_obj
                    for e_obj in enemy_objects
                    if ("is_pacman", e_obj) in initState
                ]

                # If there are multiple invaders, we need to help defend even if not closer
                multiple_invaders = len(invading_enemies) > 1

                # Chase enemies in our territory, prioritizing by fatness (same logic as ghost chase)
                # Priority: enemy_past + further_back > fat_agent_gt10 > fat_agent_gt3 > fat_agent > any invader

                # Highest priority: If we're further back and enemy has passed us, intercept immediately
                if ("further_back",) in initState:
                    for enemy_obj in enemy_objects:
                        if ("enemy_past", enemy_obj) in initState and ("fat_agent_gt10", enemy_obj) in initState:
                            if not (("ally_chasing", enemy_obj) in initState):
                                return self.goalChase(objects, initState, enemy_obj)
                    for enemy_obj in enemy_objects:
                        if ("enemy_past", enemy_obj) in initState and ("fat_agent_gt3", enemy_obj) in initState:
                            if not (("ally_chasing", enemy_obj) in initState):
                                return self.goalChase(objects, initState, enemy_obj)
                    for enemy_obj in enemy_objects:
                        if ("enemy_past", enemy_obj) in initState and ("fat_agent", enemy_obj) in initState:
                            if not (("ally_chasing", enemy_obj) in initState):
                                return self.goalChase(objects, initState, enemy_obj)
                    for enemy_obj in enemy_objects:
                        if ("enemy_past", enemy_obj) in initState:
                            if not (("ally_chasing", enemy_obj) in initState):
                                return self.goalChase(objects, initState, enemy_obj)

                # Normal priority: Chase invaders based on fatness and proximity
                for enemy_obj in enemy_objects:
                    if ("is_pacman", enemy_obj) in initState and ("fat_agent_gt10", enemy_obj) in initState:
                        if (("closer_to_enemy", enemy_obj) in initState or multiple_invaders) and not (("ally_chasing", enemy_obj) in initState):
                            return self.goalChase(objects, initState, enemy_obj)

                for enemy_obj in enemy_objects:
                    if ("is_pacman", enemy_obj) in initState and ("fat_agent_gt3", enemy_obj) in initState:
                        if (("closer_to_enemy", enemy_obj) in initState or multiple_invaders) and not (("ally_chasing", enemy_obj) in initState):
                            return self.goalChase(objects, initState, enemy_obj)

                for enemy_obj in enemy_objects:
                    if ("is_pacman", enemy_obj) in initState and ("fat_agent", enemy_obj) in initState:
                        if (("closer_to_enemy", enemy_obj) in initState or multiple_invaders) and not (("ally_chasing", enemy_obj) in initState):
                            return self.goalChase(objects, initState, enemy_obj)

                for enemy_obj in enemy_objects:
                    if ("is_pacman", enemy_obj) in initState:
                        if (("closer_to_enemy", enemy_obj) in initState or multiple_invaders) and not (("ally_chasing", enemy_obj) in initState):
                            return self.goalChase(objects, initState, enemy_obj)

            # Priority 1: Eat capsule if we're closest
            if ("closest_to_capsule",) in initState:
                return self.goalEatCapsule(objects, initState)

            # Priority 2: Don't have enough food to take lead - keep eating
            elif not (("fat_agent", myObj) in initState):
                return self.goalEatFood(objects, initState)

            # Priority 3: Have enough food to take lead and there is no food nearby - escape
            elif ("fat_agent_gt10", myObj) in initState and not (
                "near_food",
                myObj,
            ) in initState:
                return self.goalEscape(objects, initState)

            # Priority 3: Have enough for >10 point lead - escape
            elif ("fat_agent_gt10", myObj) in initState:
                return self.goalEscape(objects, initState)

            # Priority 4: Not under threat and timer high - keep eating
            elif not ("enemy_close_distance",) in initState:
                return self.goalEatFood(objects, initState)

            # Priority 5: Have enough for lead - escape
            else:
                return self.goalEscape(objects, initState)

        # ==================== GHOST LOGIC ====================

        # Priority 1: If scared, attack
        if is_scared:
            return self.goalAttack(objects, initState)

        # Get enemy objects for chase goals
        enemy_objects = [obj[0] for obj in objects if obj[1] in ["enemy1", "enemy2"]]

        # Priority 2: Chase fat enemies in priority of fatness (only if we're closer)
        for enemy_obj in enemy_objects:
            if ("fat_agent_gt10", enemy_obj) in initState:
                if (
                    not (("ally_chasing", enemy_obj) in initState)
                    and ("closer_to_enemy", enemy_obj) in initState
                ):
                    return self.goalChase(objects, initState, enemy_obj)
            elif ("fat_agent_gt3", enemy_obj) in initState:
                if (
                    not (("ally_chasing", enemy_obj) in initState)
                    and ("closer_to_enemy", enemy_obj) in initState
                ):
                    return self.goalChase(objects, initState, enemy_obj)
            elif ("fat_agent", enemy_obj) in initState:
                if (
                    not (("ally_chasing", enemy_obj) in initState)
                    and ("closer_to_enemy", enemy_obj) in initState
                ):
                    return self.goalChase(objects, initState, enemy_obj)
            elif ("is_pacman", enemy_obj) in initState:
                if (
                    not (("ally_chasing", enemy_obj) in initState)
                    and ("closer_to_enemy", enemy_obj) in initState
                ):
                    return self.goalChase(objects, initState, enemy_obj)

        # Priority 3: Winning by >3, defend
        if ("winning_gt3",) in initState:
            return self.goalDefaultDefend(objects, initState)

        # Priority 4: Teammate is chasing, we should attack
        # Check if ally is chasing any enemy
        ally_is_chasing = any(
            ("ally_chasing", eobj) in initState for eobj in enemy_objects
        )
        if ally_is_chasing:
            return self.goalAttack(objects, initState)

        # Priority 5: Not winning, attack
        if ("winning",) not in initState:
            teammateAction = MixedAgent.CURRENT_ACTION.get((self.index + 2) % 4)
            if not teammateAction or teammateAction.name not in [
                "attack",
                "eat_food",
                "eat_capsule",
                "escape",
            ]:
                return self.goalAttack(objects, initState)

        # Priority 6: Losing by >3, attack
        if ("losing_gt3",) in initState:
            return self.goalAttack(objects, initState)

        # Priority 7: Default - defend
        return self.goalDefaultDefend(objects, initState)

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

    def goalAttack(self, objects: List[Tuple], initState: List[Tuple]):
        myObj = f"a{self.index}"
        return [("is_pacman", myObj)], []

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
            prob = util.flipCoin(self.epsilon)  # get change of perform random movement
            if prob and self.trainning:
                action = random.choice(legalActions)
            else:
                for action in legalActions:
                    # if self.trainning:
                    #     self.updateWeights(
                    #         gameState,
                    #         action,
                    #         rewardFunction,
                    #         featureFunction,
                    #         weights,
                    #         learningRate,
                    #     )
                    # print("Agent",self.index," weights:", weights)
                    nextStateInfo = self.getStateInfo(
                        self.getSuccessor(gameState, action)
                    )
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

    @profile
    def updateWeights(
        self, gameState, action, rewardFunction, featureFunction, weights, learningRate
    ):
        features = featureFunction(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        reward = rewardFunction(gameState, nextState)
        for feature in features:
            correction = (
                reward
                + self.discountRate * self.getValue(nextState, featureFunction, weights)
            ) - self.getQValue(features, weights)
            weights[feature] = (
                weights[feature] + learningRate * correction * features[feature]
            )

    """
    Iterate through all q-values that we get from all
    possible actions, and return the highest q-value
    """

    @profile
    def getValue(self, nextState: GameState, featureFunction, weights):
        qVals = []
        legalActions = nextState.getLegalActions(self.index)

        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                features = featureFunction(nextState, action)
                qVals.append(self.getQValue(features, weights))
            return max(qVals)

    # ------------------------------- Feature Related Action Functions -------------------------------

    def getStateInfo(self, gameState: GameState):
        enemyVirtualStates = {}
        for enemy_idx in self.getOpponents(gameState):
            # Get the real enemy state from gameState (has correct scaredTimer, isPacman, etc.)
            real_enemy_state = gameState.getAgentState(enemy_idx)
            enemy_state = real_enemy_state.copy()

            # Only override the position with our estimate, keep everything else (scaredTimer, etc.) from real state
            enemy_state.configuration = Configuration(
                MixedAgent.ESTIMATED_POSITIONS.get(enemy_idx),
                real_enemy_state.configuration.direction if real_enemy_state.configuration else -1
            )
            enemyVirtualStates[enemy_idx] = enemy_state

        return StateInfo(
            gameState,
            gameState.getAgentState(self.index),
            gameState.getAgentState((self.index + 2) % 4),
            enemyVirtualStates,
        )

    # ==================== Shared Helper Functions ====================

    def getBreathingDistanceGhosts(
        self, myPos, enemyVirtualStates: Dict[int, AgentState]
    ):
        """Count non-scared ghosts within breathing distance (2 steps)"""
        count = 0
        for enemy_idx, enemy_state in enemyVirtualStates.items():
            # Debug: print ALL enemy states
            if self.debug:
                print(f"Agent {self.index}: Enemy {enemy_idx} - isPacman={enemy_state.isPacman}, scaredTimer={enemy_state.scaredTimer}")

            if not enemy_state.isPacman and enemy_state.scaredTimer == 0:
                enemy_pos = enemy_state.getPosition()
                if enemy_pos:
                    dist = self.getMazeDistance(myPos, enemy_pos)
                    if dist <= BREATHING_DISTANCE:
                        count += 1
                        if self.debug:
                            print(f"Agent {self.index}: Counting ghost {enemy_idx} at breathing distance")
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
        self, myPos, enemyVirtualStates: Dict[int, AgentState]
    ):
        """Get distance to nearest enemy (any type)"""
        min_dist = float("inf")
        for enemy_idx, enemy_state in enemyVirtualStates.items():
            enemy_pos = enemy_state.getPosition()
            if enemy_pos:
                dist = self.getMazeDistance(myPos, enemy_pos)
                if dist < min_dist:
                    min_dist = dist
        return min_dist if min_dist != float("inf") else 0

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
        nextPos = nextStateInfo.agentState.getPosition()

        # Priority 1: Avoid breathing distance ghosts (most critical)
        features["breathing-distance-ghosts"] = self.getBreathingDistanceGhosts(
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

        # Tie-breaking: Penalize stop and reverse
        features["stop"] = 1.0 if action == Directions.STOP else 0.0
        features["reverse"] = (
            1.0
            if action
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

        # Tie-breaking: Penalize stop and reverse
        features["stop"] = 1.0 if action == Directions.STOP else 0.0
        features["reverse"] = (
            1.0
            if action
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

        # Tie-breaking: Penalize stop and reverse
        features["stop"] = 1.0 if action == Directions.STOP else 0.0
        features["reverse"] = (
            1.0
            if action
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

        # Tie-breaking: Penalize stop and reverse
        features["stop"] = 1.0 if action == Directions.STOP else 0.0
        features["reverse"] = (
            1.0
            if action
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

        # Tie-breaking: Penalize stop and reverse
        features["stop"] = 1.0 if action == Directions.STOP else 0.0
        features["reverse"] = (
            1.0
            if action
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
        distance_to_nearest_enemy = self.getDistanceToNearestEnemy(
            nextPos, nextStateInfo.enemyVirtualStates
        )

        if distance_to_nearest_enemy <= CLOSE_DISTANCE:
            features["in-home"] = (
                1.0 if self.isInHome(nextPos, nextStateInfo.gameState) else 0.0
            )

        # Priority 2: Position near enemies (to detect/intercept)
        features["distance-to-nearest-enemy"] = distance_to_nearest_enemy

        # Priority 3: Spread out from teammate
        features["distance-to-teammate"] = self.getDistanceToTeammate(
            nextPos, nextStateInfo.teammateState
        )

        # Tie-breaking: Penalize stop and reverse
        features["stop"] = 1.0 if action == Directions.STOP else 0.0
        features["reverse"] = (
            1.0
            if action
            == Directions.REVERSE[stateInfo.agentState.configuration.direction]
            else 0.0
        )

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
