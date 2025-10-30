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
from typing import List, Tuple

from numpy import true_divide
import numpy as np
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os
from capture import GameState, noisyDistance
from game import Directions, Actions, AgentState, Agent
from util import nearestPoint
import sys, os
import pickle

from belief_tracking import initialize_beliefs, update_all_beliefs

# the folder of current file.
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

from lib_piglet.utils.pddl_solver import pddl_solver
from lib_piglet.domains.pddl import pddl_state
from lib_piglet.utils.pddl_parser import Action

CLOSE_DISTANCE = 4
MEDIUM_DISTANCE = 15
LONG_DISTANCE = 25


#################
# Team creation #
#################


def createTeam(
    firstIndex, secondIndex, isRed, first="MixedAgent", second="MixedAgent", **kwargs
):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
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
        "offensiveWeights": {
            "bias": 1,
            "successorScore": 100,
            "closest-food": -10,  # Primary driver - get to food
            "#-of-ghosts-1-step-away": -200,  # Critical - never move next to ghost
            "ghost-very-close": -100,  # 1-2 steps away = very dangerous
            "ghost-close": -30,  # 3-4 steps away = concerning
            "ghost-nearby": -10,  # 5-6 steps away = minor concern
            "scared-ghost-nearby": 30,  # Opportunity to be aggressive
            "chance-return-food": 10,
            "carrying-lots-of-food": 5,
            "distance-to-home": -15,  # When carrying lots, closer to home is better
            "food-density": 20,  # Prefer areas with more food
            "eats-food": 50,  # Strongly reward eating food
            "stop": -20,  # Discourage stopping
            "reverse": -10,  # Discourage reversing
            "in-dead-end": -50,  # Avoid dead ends when possible
        },
        "defensiveWeights": {
            "bias": 1,
            "onDefense": 100,
            "teamDistance": 2,
            "numInvaders": -1000,
            "invaderDistance": -10,
            "about-to-catch": 200,  # Big reward for being next to invader
            "invader-has-food": -50,  # Priority: chase invaders with food
            "distance-to-threatened-food": -20,  # Get to threatened food
            "intercepting-food-threat": 100,  # Reward being closer than invader
            "distance-to-our-capsule": -30,  # Protect capsules
            "capsule-under-threat": -100,  # Big penalty if capsule threatened
            "protecting-capsule": 150,  # Big reward for protecting capsule
            "blocking-escape": 50,  # Reward blocking invader's escape
            "stop": -100,
            "reverse": -2,
        },
        "escapeWeights": {
            "bias": 1,
            "onDefense": 1000,  # Huge reward for making it home
            "distanceToHome": -50,  # Get home quickly
            "carrying-food": 10,  # Value of food being carried
            "enemyDistance": 30,  # Stay away from ghosts
            "imminent-danger": -200,  # Ghost 1 step away is critical
            "close-danger": -100,  # Ghost 2 steps away is dangerous
            "ghost-blocking-home": -150,  # Ghost between us and home
            "moving-toward-home": 50,  # Reward progress toward home
            "in-dead-end": -100,  # Avoid dead ends when escaping
            "stop": -100,
            "reverse": -5,
        },
    }
    QLWeightsFile = BASE_FOLDER + "/QLWeightsClaudeTeam.pkl"

    # Also can use class variable to exchange information between agents.
    CURRENT_ACTION = {}
    ESTIMATED_POSITIONS = {}  # Cache for estimated enemy positions using beliefs

    def registerInitialState(self, gameState: GameState):
        self.pddl_solver = pddl_solver(BASE_FOLDER + "/claudeTeam.pddl")
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
                print(
                    f"Agent {self.index}: Loaded learned weights from {MixedAgent.QLWeightsFile}"
                )
        except Exception as e:
            print(f"Agent {self.index}: Could not load weights: {e}")

        # Initialize belief tracking for opponents

        self.use_beliefs = True
        if (
            not hasattr(MixedAgent, "OPPONENT_BELIEFS")
            or len(MixedAgent.OPPONENT_BELIEFS) == 0
        ):
            MixedAgent.OPPONENT_BELIEFS = initialize_beliefs(gameState)

        # Use a dictionary to save information about current agent.
        MixedAgent.CURRENT_ACTION[self.index] = {}
        raise Exception("I'm a fugitive")

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
                    print(
                        f"Agent {self.index}: Saved learned weights to {MixedAgent.QLWeightsFile}"
                    )
                    print(
                        "Offensive weights:", MixedAgent.QLWeights["offensiveWeights"]
                    )
                    print(
                        "Defensive weights:", MixedAgent.QLWeights["defensiveWeights"]
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
            elif self.use_beliefs and enemy_idx in MixedAgent.OPPONENT_BELIEFS:
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
        if self.use_beliefs:
            try:
                from belief_tracking import update_all_beliefs

                MixedAgent.OPPONENT_BELIEFS = update_all_beliefs(
                    MixedAgent.OPPONENT_BELIEFS, gameState, self.index
                )
            except:
                pass

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
            print(f"  Plan: {[action.name for action, _ in self.highLevelPlan]}")

        if not self.highLevelPlan:
            print(f"No plan found for predicates: {initState}")
            highLevelAction = "defense"
        else:
            # Get next action from the plan
            highLevelAction = self.highLevelPlan[self.currentActionIndex][0].name
        MixedAgent.CURRENT_ACTION[self.index] = highLevelAction
        print(f"Agent {self.index}: High-Level Action = {highLevelAction}")

        # -------------Low Level Plan Section-------------------
        # Get the low level plan using Q learning, and return a low level action at last.
        # A low level action is defined in Directions, whihc include {"North", "South", "East", "West", "Stop"}

        if not self.posSatisfyLowLevelPlan(gameState):
            self.lowLevelPlan = self.getLowLevelPlanQL(
                gameState, highLevelAction
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
        foodLeft = self.getFood(gameState).asList()
        if len(foodLeft) > 0:
            states.append(("food_available",))
        myPos = gameState.getAgentPosition(self.index)
        myObj = "a{}".format(self.index)
        cloestFoodDist = self.closestFood(
            myPos, self.getFood(gameState), gameState.getWalls()
        )
        if cloestFoodDist != None and cloestFoodDist <= CLOSE_DISTANCE:
            states.append(("near_food", myObj))

        # Collect capsule states
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0:
            states.append(("capsule_available",))
        for cap in capsules:
            if self.getMazeDistance(cap, myPos) <= CLOSE_DISTANCE:
                states.append(("near_capsule", myObj))
                break

        # Collect winning states
        currentScore = gameState.data.score
        if gameState.isOnRedTeam(self.index):
            if currentScore > 0:
                states.append(("winning",))
            if currentScore > 3:
                states.append(("winning_gt3",))
            if currentScore > 5:
                states.append(("winning_gt5",))
            if currentScore > 10:
                states.append(("winning_gt10",))
            if currentScore > 20:
                states.append(("winning_gt20",))
        else:
            if currentScore < 0:
                states.append(("winning",))
            if currentScore < -3:
                states.append(("winning_gt3",))
            if currentScore < -5:
                states.append(("winning_gt5",))
            if currentScore < -10:
                states.append(("winning_gt10",))
            if currentScore < -20:
                states.append(("winning_gt20",))

        # Time remaining predicates
        if hasattr(gameState.data, "timeleft"):
            if gameState.data.timeleft < 300:
                states.append(("low_time_remaining",))
            if gameState.data.timeleft < 100:
                states.append(("very_low_time_remaining",))

        # Check for food clusters (3+ food within close distance)
        foodClusterCount = sum(
            1
            for food in foodLeft
            if self.getMazeDistance(myPos, food) <= CLOSE_DISTANCE
        )
        if foodClusterCount >= 3:
            states.append(("food_cluster_nearby", myObj))

        # Collect team agents states
        agents: List[Tuple[int, AgentState]] = [
            (i, gameState.getAgentState(i)) for i in self.getTeam(gameState)
        ]
        for agent_index, agent_state in agents:
            agent_object = "a{}".format(agent_index)
            agent_type = "current_agent" if agent_index == self.index else "ally"
            objects += [(agent_object, agent_type)]

            if (
                agent_index != self.index
                and self.getMazeDistance(
                    gameState.getAgentPosition(self.index),
                    gameState.getAgentPosition(agent_index),
                )
                <= CLOSE_DISTANCE
            ):
                states.append(("near_ally",))

            if agent_state.scaredTimer > 0:
                states.append(("is_scared", agent_object))

            if agent_state.numCarrying > 0:
                states.append(("food_in_backpack", agent_object))
                if agent_state.numCarrying >= 20:
                    states.append(("20_food_in_backpack", agent_object))
                if agent_state.numCarrying >= 10:
                    states.append(("10_food_in_backpack", agent_object))
                if agent_state.numCarrying >= 5:
                    states.append(("5_food_in_backpack", agent_object))
                if agent_state.numCarrying >= 3:
                    states.append(("3_food_in_backpack", agent_object))

            if agent_state.isPacman:
                states.append(("is_pacman", agent_object))

            # Ally coordination predicates
            if agent_index != self.index:
                # Check what ally is doing based on shared CURRENT_ACTION
                if agent_index in MixedAgent.CURRENT_ACTION:
                    ally_action = MixedAgent.CURRENT_ACTION.get(agent_index, "")
                    if (
                        "defence" in str(ally_action)
                        or "patrol" in str(ally_action)
                        or "chase" in str(ally_action)
                    ):
                        states.append(("ally_defending", agent_object))
                    elif "attack" in str(ally_action) or "collect" in str(ally_action):
                        states.append(("ally_attacking", agent_object))

        # Collect enemy agents states
        enemies: List[Tuple[int, AgentState]] = [
            (i, gameState.getAgentState(i)) for i in self.getOpponents(gameState)
        ]
        noisyDistance = gameState.getAgentDistances()
        typeIndex = 1
        for enemy_index, enemy_state in enemies:
            enemy_position = enemy_state.getPosition()
            enemy_object = "e{}".format(enemy_index)
            objects += [(enemy_object, "enemy{}".format(typeIndex))]

            if enemy_state.scaredTimer > 0:
                states.append(("is_scared", enemy_object))

            if enemy_position != None:
                for agent_index, agent_state in agents:
                    if (
                        self.getMazeDistance(agent_state.getPosition(), enemy_position)
                        <= CLOSE_DISTANCE
                    ):
                        states.append(
                            ("enemy_around", enemy_object, "a{}".format(agent_index))
                        )
            else:
                if noisyDistance[enemy_index] >= LONG_DISTANCE:
                    states.append(
                        ("enemy_long_distance", enemy_object, "a{}".format(self.index))
                    )
                elif noisyDistance[enemy_index] >= MEDIUM_DISTANCE:
                    states.append(
                        (
                            "enemy_medium_distance",
                            enemy_object,
                            "a{}".format(self.index),
                        )
                    )
                else:
                    states.append(
                        ("enemy_short_distance", enemy_object, "a{}".format(self.index))
                    )

            if enemy_state.isPacman:
                states.append(("is_pacman", enemy_object))

            # Check if enemy is carrying food (always visible in AgentState)
            if enemy_state.numCarrying > 0:
                states.append(("enemy_carrying_food", enemy_object))

            # Check if enemy is near our food (threatening)
            ourFood = self.getFoodYouAreDefending(gameState).asList()
            # Use cached estimated position (computed once at start of chooseAction)
            estimated_pos = MixedAgent.ESTIMATED_POSITIONS.get(enemy_index)
            if estimated_pos is not None:
                for food_pos in ourFood:
                    if self.getMazeDistance(estimated_pos, food_pos) <= 3:
                        states.append(("enemy_nearby_food", enemy_object))
                        break

            typeIndex += 1

        # Strategic predicates based on overall game state
        myState = gameState.getAgentState(self.index)
        walls = gameState.getWalls()

        # Determine if it's safe to attack (no nearby non-scared ghosts)
        # Use estimated positions to check ALL ghosts, not just observable ones
        ghosts_nearby = []
        for enemy_idx in self.getOpponents(gameState):
            enemy_state = gameState.getAgentState(enemy_idx)
            if not enemy_state.isPacman and enemy_state.scaredTimer == 0:
                # Non-scared ghost - check if nearby using estimated position
                est_pos = MixedAgent.ESTIMATED_POSITIONS.get(enemy_idx)
                if est_pos is not None:
                    dist = self.getMazeDistance(myPos, est_pos)
                    # Only consider CLOSE ghosts as dangerous for safe_to_attack
                    # Medium/long distance ghosts shouldn't prevent attacking
                    if dist <= CLOSE_DISTANCE:
                        ghosts_nearby.append((enemy_idx, enemy_state))

        # Safe to attack if no ghosts nearby OR all nearby ghosts are scared
        if len(ghosts_nearby) == 0:
            states.append(("safe_to_attack", myObj))

        # Can catch enemy (we're ghost, they're Pacman nearby, we're not scared)
        if not myState.isPacman and myState.scaredTimer == 0:
            for enemy_idx in self.getOpponents(gameState):
                enemy_state = gameState.getAgentState(enemy_idx)
                if enemy_state.isPacman:
                    est_pos = MixedAgent.ESTIMATED_POSITIONS.get(enemy_idx)
                    if (
                        est_pos is not None
                        and self.getMazeDistance(myPos, est_pos) <= CLOSE_DISTANCE
                    ):
                        states.append(
                            ("can_catch_enemy", "e{}".format(enemy_idx), myObj)
                        )

        # Should retreat: either ghosts nearby OR have enough food to take/extend lead
        if myState.isPacman:
            # Calculate score differential from OUR perspective
            # getScore() returns positive when red winning, negative when blue winning
            raw_score = self.getScore(gameState)
            our_score_diff = raw_score if self.red else -raw_score  # positive = we're winning

            # Calculate food needed to take/extend the lead
            # If tied (our_score_diff == 0): need 1 food to lead
            # If losing by 2 (our_score_diff == -2): need 3 food to lead by 1
            # If winning (our_score_diff > 0): even 1 food extends the lead
            if our_score_diff >= 0:
                # Tied or winning: 1 food is valuable
                food_needed_to_lead = 1
            else:
                # Losing: need enough to overcome deficit + 1
                food_needed_to_lead = abs(our_score_diff) + 1

            # Set predicates
            if myState.numCarrying >= food_needed_to_lead:
                states.append(("enough_food_to_lead", myObj))

            # Should retreat if: (1) ghosts nearby OR (2) have enough food to lead
            if len(ghosts_nearby) > 0 or myState.numCarrying >= food_needed_to_lead:
                states.append(("should_retreat", myObj))

        # Check if ally has more enemies around (for support_ally action)
        from game import Actions
        for agent_index, agent_state in agents:
            if agent_index != self.index:
                ally_pos = agent_state.getPosition()
                # Count enemies near me using estimated positions
                my_nearby_enemies = 0
                ally_nearby_enemies = 0
                for enemy_idx in self.getOpponents(gameState):
                    est_pos = MixedAgent.ESTIMATED_POSITIONS.get(enemy_idx)
                    if est_pos is not None:
                        if self.getMazeDistance(myPos, est_pos) <= CLOSE_DISTANCE:
                            my_nearby_enemies += 1
                        if self.getMazeDistance(ally_pos, est_pos) <= CLOSE_DISTANCE:
                            ally_nearby_enemies += 1

                if ally_nearby_enemies > my_nearby_enemies and ally_nearby_enemies > 0:
                    states.append(("more_enemies_around_ally",))

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
        # Check if current agent is a Pacman (on enemy territory)
        myObj = f"a{self.index}"
        is_pacman = ("is_pacman", myObj) in initState

        # If we're already a Pacman, we should focus on collecting food or going home
        # NOT on defensive goals!
        if is_pacman:
            # Check if we should retreat
            # This is true if: (1) ghosts nearby OR (2) have enough food to take/extend lead
            should_retreat = ("should_retreat", myObj) in initState
            enough_food_to_lead = ("enough_food_to_lead", myObj) in initState

            if should_retreat:
                # Goal: get home safely (either threatened by ghosts OR have strategic food)
                positiveGoal = []
                negtiveGoal = [
                    ("is_pacman", myObj)
                ]  # Want to stop being Pacman (cross back home)
                return positiveGoal, negtiveGoal
            else:
                # Goal: continue attacking to collect food
                return self.goalAggressive(objects, initState)

        # Not a Pacman - use strategic goal selection based on game state
        # Priority 1: Endgame scenarios (very low time remaining)
        if ("very_low_time_remaining",) in initState:
            if ("winning",) in initState:
                return self.goalSecureWin(objects, initState)
            else:
                return self.goalDesperateAttack(objects, initState)

        # Priority 2: Strong winning position (>10 points ahead)
        if ("winning_gt10",) in initState:
            return self.goalDefensive(objects, initState)

        # Priority 3: Moderate lead (5-10 points ahead) - balanced approach
        if ("winning_gt5",) in initState:
            return self.goalBalanced(objects, initState)

        # Priority 4: Close game or losing - aggressive offensive
        if ("low_time_remaining",) in initState:
            return self.goalTimeAggressive(objects, initState)

        # Priority 5: Standard offensive play
        return self.goalAggressive(objects, initState)

    def goalSecureWin(self, objects: List[Tuple], initState: List[Tuple]):
        """
        Endgame with winning position: Defend and run out the clock.
        """
        positiveGoal = [("defend_foods",)]
        negtiveGoal = []
        # Ensure no enemies are invading
        for obj in objects:
            if obj[1] in ["enemy1", "enemy2"]:
                negtiveGoal.append(("is_pacman", obj[0]))
        return positiveGoal, negtiveGoal

    def goalDesperateAttack(self, objects: List[Tuple], initState: List[Tuple]):
        """
        Endgame while losing: All-out attack to score points.
        """
        positiveGoal = []
        negtiveGoal = [("food_available",)]  # Must eat all food
        return positiveGoal, negtiveGoal

    def goalDefensive(self, objects: List[Tuple], initState: List[Tuple]):
        """
        Strong lead: Focus on defense and preventing enemy scores.
        """
        positiveGoal = [("defend_foods",)]
        negtiveGoal = []
        # Keep all enemies out
        for obj in objects:
            if obj[1] in ["enemy1", "enemy2"]:
                negtiveGoal.append(("is_pacman", obj[0]))
        return positiveGoal, negtiveGoal

    def goalBalanced(self, objects: List[Tuple], initState: List[Tuple]):
        """
        Moderate lead: Balance between offense and defense.
        One agent attacks if safe, one defends.
        """
        positiveGoal = []
        negtiveGoal = []

        # Still want to score more, but also defend
        if ("food_available",) in initState:
            negtiveGoal.append(("food_available",))

        # Defend against invaders
        for obj in objects:
            if obj[1] in ["enemy1", "enemy2"]:
                enemy_is_pacman = ("is_pacman", obj[0]) in initState
                if enemy_is_pacman:
                    negtiveGoal.append(("is_pacman", obj[0]))

        return positiveGoal, negtiveGoal

    def goalTimeAggressive(self, objects: List[Tuple], initState: List[Tuple]):
        """
        Low time remaining but not very low: Push for more points.
        """
        positiveGoal = []
        negtiveGoal = [("food_available",)]

        # Also defend against serious threats
        for obj in objects:
            if obj[1] in ["enemy1", "enemy2"]:
                if ("enemy_carrying_food", obj[0]) in initState:
                    negtiveGoal.append(("is_pacman", obj[0]))

        return positiveGoal, negtiveGoal

    def goalAggressive(self, objects: List[Tuple], initState: List[Tuple]):
        """
        Standard aggressive offensive play: Score points while defending.
        """
        positiveGoal = []
        negtiveGoal = [("food_available",)]  # Want to eat all enemy food

        # Also stop invaders
        for obj in objects:
            if obj[1] in ["enemy1", "enemy2"]:
                negtiveGoal.append(("is_pacman", obj[0]))

        return positiveGoal, negtiveGoal

    # ------------------------------- Heuristic search low level plan Functions -------------------------------
    def getLowLevelPlanHS(
        self, gameState: GameState, highLevelAction: str
    ) -> List[Tuple[str, Tuple]]:
        # This is a function for plan low level actions using heuristic search.
        # You need to implement this function if you want to solve low level actions using heuristic search.
        # Here, we list some function you might need, read the GameState and CaptureAgent code for more useful functions.
        # These functions also useful for collecting features for Q learnning low levels.

        map = (
            gameState.getWalls()
        )  # a 2d array matrix of obstacles, map[x][y] = true means a obstacle(wall) on x,y, map[x][y] = false indicate a free location
        foods = self.getFood(
            gameState
        )  # a 2d array matrix of food,  foods[x][y] = true if there's a food.
        capsules = self.getCapsules(gameState)  # a list of capsules
        foodNeedDefend = self.getFoodYouAreDefending(
            gameState
        )  # return food will be eatan by enemy (food next to enemy)
        capsuleNeedDefend = self.getCapsulesYouAreDefending(
            gameState
        )  # return capsule will be eatan by enemy (capsule next to enemy)
        Raise(NotImplementedError("Heuristic Search low level "))
        return (
            []
        )  # You should return a list of tuple of move action and target location (exclude current location).

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
        # Offensive actions: attack, aggressive_attack, collect_food_cluster, eat_capsule, desperate_attack
        if (
            highLevelAction == "attack"
            or highLevelAction == "aggressive_attack"
            or highLevelAction == "collect_food_cluster"
            or highLevelAction == "eat_capsule"
            or highLevelAction == "desperate_attack"
        ):
            # Offensive actions - use offensive features and rewards
            rewardFunction = self.getOffensiveReward
            featureFunction = self.getOffensiveFeatures
            weights = self.getOffensiveWeights()
            learningRate = self.alpha
        # Retreat actions: go_home_with_food, go_home_retreat, emergency_retreat, or any action with "retreat"/"escape"/"go_home" in name
        elif (
            highLevelAction == "go_home_with_food"
            or highLevelAction == "go_home_retreat"
            or highLevelAction == "emergency_retreat"
            or "retreat" in highLevelAction
            or "escape" in highLevelAction
            or "go_home" in highLevelAction
        ):
            # Escape actions - complete reward function implemented
            rewardFunction = self.getEscapeReward
            featureFunction = self.getEscapeFeatures
            weights = self.getEscapeWeights()
            learningRate = self.alpha  # Enable learning for escape
        else:
            print("DEFENSE", highLevelAction)
            # Defensive actions - complete reward function implemented
            rewardFunction = self.getDefensiveReward
            featureFunction = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()
            learningRate = self.alpha  # Enable learning for defense

        if len(legalActions) != 0:
            prob = util.flipCoin(self.epsilon)  # get change of perform random movement
            if prob and self.trainning:
                action = random.choice(legalActions)
            else:
                for action in legalActions:
                    if self.trainning:
                        self.updateWeights(
                            gameState,
                            action,
                            rewardFunction,
                            featureFunction,
                            weights,
                            learningRate,
                        )
                        # print("Agent",self.index," weights:", weights)
                    features = featureFunction(gameState, action)
                    qval = self.getQValue(features, weights)
                    values.append((qval, action, features))

                # Debug: print Q-values and feature breakdown for all actions
                if self.index == 0 and len(values) > 0:
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

    def getOffensiveReward(self, gameState: GameState, nextState: GameState):
        """
        Enhanced offensive reward function.
        Rewards: eating food, returning food, staying safe from ghosts.
        Penalties: getting caught, wasting time, dangerous positions.
        """
        currentAgentState: AgentState = gameState.getAgentState(self.index)
        nextAgentState: AgentState = nextState.getAgentState(self.index)
        walls = gameState.getWalls()

        # Base reward (small penalty for time to encourage efficiency)
        reward = -1

        # Food collection rewards
        food_eaten = nextAgentState.numCarrying - currentAgentState.numCarrying
        if food_eaten > 0:
            reward += food_eaten * 10  # Reward for eating food

        # Food return rewards
        food_returned = nextAgentState.numReturned - currentAgentState.numReturned
        if food_returned > 0:
            reward += food_returned * 30  # Big reward for successfully returning food

        # Score improvement
        score_diff = self.getScore(nextState) - self.getScore(gameState)
        reward += score_diff * 50

        # Ghost danger penalties (using estimated positions)
        nextPos = nextAgentState.getPosition()
        dangerous_ghosts = []
        for enemy_idx in self.getOpponents(gameState):
            enemy_state = gameState.getAgentState(enemy_idx)
            if not enemy_state.isPacman and enemy_state.scaredTimer == 0:
                est_pos = MixedAgent.ESTIMATED_POSITIONS.get(enemy_idx)
                if est_pos is not None:
                    dist = self.getMazeDistance(nextPos, est_pos)
                    if dist <= 3:
                        dangerous_ghosts.append(dist)

        # Penalty based on proximity to ghosts
        if len(dangerous_ghosts) > 0:
            min_dist = min(dangerous_ghosts)
            if min_dist == 1:
                reward -= 50  # Very close to ghost!
            elif min_dist == 2:
                reward -= 20
            elif min_dist == 3:
                reward -= 5

        # Getting caught penalty
        if (
            currentAgentState.numCarrying > 0
            and nextAgentState.getPosition() == self.startPosition
        ):
            # Check if we got caught (not a successful return)
            if nextAgentState.numReturned == currentAgentState.numReturned:
                reward -= (
                    currentAgentState.numCarrying * 25
                )  # Penalty for getting caught

        # Carrying food safely
        if nextAgentState.isPacman:
            reward += nextAgentState.numCarrying  # Small reward for carrying food

        # Encourage returning when carrying a lot
        if nextAgentState.numCarrying >= 5:
            dist_home = self.getMazeDistance(nextPos, self.startPosition)
            # Reward for being closer to home when carrying lots
            reward += (10 - dist_home) if dist_home < 10 else 0

        return reward

    def getDefensiveReward(self, gameState, nextState):
        """
        Complete defensive reward function.
        Rewards: eliminating invaders, staying on defense, protecting food.
        Penalties: allowing food to be eaten, letting invaders escape.
        Uses ESTIMATED_POSITIONS to include unobserved invaders.
        """
        reward = 0

        # Get invader information - using estimated positions for all invaders
        opponents = self.getOpponents(nextState)
        invaders = []
        for idx in opponents:
            enemy_state = nextState.getAgentState(idx)
            if enemy_state.isPacman:
                # Get position (exact or estimated)
                pos = MixedAgent.ESTIMATED_POSITIONS.get(idx)
                if pos is not None:
                    invaders.append((idx, pos))

        # Previous invaders
        prevInvaders = []
        for idx in opponents:
            enemy_state = gameState.getAgentState(idx)
            if enemy_state.isPacman:
                pos = MixedAgent.ESTIMATED_POSITIONS.get(idx)
                if pos is not None:
                    prevInvaders.append((idx, pos))

        # Big reward for reducing number of invaders (caught one!)
        invader_diff = len(prevInvaders) - len(invaders)
        reward += invader_diff * 100

        # Reward for being on defense
        myState = nextState.getAgentState(self.index)
        if not myState.isPacman:
            reward += 10
        else:
            reward -= 20  # Penalty for leaving defense

        # Reward/penalty based on distance to invaders
        if len(invaders) > 0:
            myPos = nextState.getAgentState(self.index).getPosition()
            minDistToInvader = min(
                [self.getMazeDistance(myPos, inv_pos) for _, inv_pos in invaders]
            )
            # Closer to invader is better (negative distance becomes positive reward)
            reward -= minDistToInvader * 2

            # Extra penalty if invader is very close to our food
            ourFood = self.getFoodYouAreDefending(nextState).asList()
            if len(ourFood) > 0:
                minInvaderDistToFood = min(
                    [
                        self.getMazeDistance(inv_pos, food)
                        for _, inv_pos in invaders
                        for food in ourFood
                    ]
                )
                if minInvaderDistToFood <= 2:
                    reward -= 20  # Invader threatening our food!

        # Big penalty if our food is eaten
        prevFoodCount = len(self.getFoodYouAreDefending(gameState).asList())
        currFoodCount = len(self.getFoodYouAreDefending(nextState).asList())
        foodLost = prevFoodCount - currFoodCount
        reward -= foodLost * 50

        # Reward for being between invader and their home (cut them off)
        if len(invaders) > 0 and not myState.isPacman:
            myPos = myState.getPosition()
            for _, invPos in invaders:
                # Get the boundary (center line) of our territory
                walls = gameState.getWalls()
                if gameState.isOnRedTeam(self.index):
                    boundary_x = walls.width // 2 - 1
                else:
                    boundary_x = walls.width // 2

                # Check if we're between invader and boundary
                if gameState.isOnRedTeam(self.index):
                    if myPos[0] < invPos[0]:  # We're between invader and their escape
                        reward += 15
                else:
                    if myPos[0] > invPos[0]:
                        reward += 15

        return reward

    def getEscapeReward(self, gameState, nextState):
        """
        Complete escape reward function for returning home with food.
        Rewards: getting closer to home, successfully returning, avoiding ghosts.
        Penalties: getting caught, moving away from home.
        """
        reward = 0

        myState = nextState.getAgentState(self.index)
        myPrevState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()

        # Distinguish between successful return vs getting caught
        food_was_carrying = myPrevState.numCarrying
        food_now_carrying = myState.numCarrying
        food_returned_count = myState.numReturned - myPrevState.numReturned

        # Case 1: Successfully returned home with food (crossed boundary safely)
        if food_returned_count > 0:
            reward += (
                food_returned_count * 20
            )  # Big reward proportional to food returned

        # Case 2: Got caught (was carrying food, now at start position, but numReturned didn't increase)
        elif (
            food_was_carrying > 0
            and food_now_carrying == 0
            and food_returned_count == 0
        ):
            # Check if we're at start position (indicates we got caught)
            if myPos == self.startPosition:
                reward -= food_was_carrying * 30  # Harsh penalty for getting caught

        # Reward for getting closer to home while carrying food
        if myState.isPacman and myState.numCarrying > 0:
            distToHome = self.getMazeDistance(myPos, self.startPosition)
            prevDistToHome = self.getMazeDistance(
                myPrevState.getPosition(), self.startPosition
            )
            improvement = prevDistToHome - distToHome
            reward += improvement * 10  # Reward for each step closer

            # Extra reward when very close to home with food
            if distToHome <= 3:
                reward += 20

        # Penalty for moving away from home when carrying food
        if myState.isPacman and myState.numCarrying > 0:
            distToHome = self.getMazeDistance(myPos, self.startPosition)
            prevDistToHome = self.getMazeDistance(
                myPrevState.getPosition(), self.startPosition
            )
            if distToHome > prevDistToHome:
                reward -= 15  # Moving wrong direction!

        # Reward for increasing distance from ghosts
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [
            e
            for e in enemies
            if not e.isPacman and e.getPosition() != None and e.scaredTimer == 0
        ]

        if len(ghosts) > 0 and myState.isPacman:
            minGhostDist = min(
                [self.getMazeDistance(myPos, g.getPosition()) for g in ghosts]
            )
            # Being farther from ghosts is good, but with diminishing returns
            if minGhostDist >= 5:
                reward += 5
            elif minGhostDist >= 3:
                reward += 2
            elif minGhostDist == 2:
                reward -= 10  # Danger!
            elif minGhostDist <= 1:
                reward -= 30  # Critical danger!

            # Extra penalty if ghost is between us and home
            walls = gameState.getWalls()
            if gameState.isOnRedTeam(self.index):
                boundary_x = walls.width // 2 - 1
                for ghost in ghosts:
                    ghostPos = ghost.getPosition()
                    # If ghost x is between us and boundary, that's bad
                    if myPos[0] > ghostPos[0] > boundary_x:
                        reward -= 15
            else:
                boundary_x = walls.width // 2
                for ghost in ghosts:
                    ghostPos = ghost.getPosition()
                    if myPos[0] < ghostPos[0] < boundary_x:
                        reward -= 15

        return reward

    # ------------------------------- Feature Related Action Functions -------------------------------

    def getOffensiveFeatures(self, gameState: GameState, action):
        """
        Enhanced offensive features using belief tracking and strategic information.
        """
        food = self.getFood(gameState)
        currAgentState = gameState.getAgentState(self.index)
        walls = gameState.getWalls()

        # Initialize features
        features = util.Counter()
        nextState = self.getSuccessor(gameState, action)
        nextAgentState = nextState.getAgentState(self.index)
        next_x, next_y = nextState.getAgentPosition(self.index)

        # Bias
        features["bias"] = 1.0

        # Successor Score (normalized)
        features["successorScore"] = (
            self.getScore(nextState) / (walls.width + walls.height) * 10
        )

        # Ghost proximity features (using estimated positions for all ghosts)
        ghosts = self.getGhostLocs(gameState)  # Observable ghosts
        all_ghost_positions = []
        for enemy_idx in self.getOpponents(gameState):
            enemy_state = gameState.getAgentState(enemy_idx)
            if not enemy_state.isPacman and enemy_state.scaredTimer == 0:
                # This is a non-scared ghost
                est_pos = MixedAgent.ESTIMATED_POSITIONS.get(enemy_idx)
                if est_pos is not None:
                    all_ghost_positions.append(est_pos)

        # Number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls)
            for g in all_ghost_positions
        )

        # Ghost threat levels (only care when they're close enough to be dangerous)
        if len(all_ghost_positions) > 0:
            min_ghost_dist = min(
                [self.getMazeDistance((next_x, next_y), g) for g in all_ghost_positions]
            )

            # Only penalize when ghost is within threatening range
            if min_ghost_dist <= 2:
                features["ghost-very-close"] = 1.0  # 1-2 steps away = critical danger
            elif min_ghost_dist <= 4:
                features["ghost-close"] = 1.0  # 3-4 steps away = moderate danger
            elif min_ghost_dist <= 6:
                features["ghost-nearby"] = 1.0  # 5-6 steps away = minor concern
            # Otherwise no ghost feature (ghost too far to matter)

        # Scared ghost features (opportunity to push deeper)
        scared_ghosts = []
        for enemy_idx in self.getOpponents(gameState):
            enemy_state = gameState.getAgentState(enemy_idx)
            if not enemy_state.isPacman and enemy_state.scaredTimer > 0:
                est_pos = MixedAgent.ESTIMATED_POSITIONS.get(enemy_idx)
                if est_pos is not None:
                    scared_ghosts.append((est_pos, enemy_state.scaredTimer))

        features["scared-ghost-nearby"] = 1.0 if len(scared_ghosts) > 0 else 0.0

        # Food carrying and return urgency
        dist_home = self.getMazeDistance((next_x, next_y), self.startPosition) + 1
        features["chance-return-food"] = (currAgentState.numCarrying) * (
            1 - dist_home / (walls.width + walls.height)
        )

        # Carrying high-value food (should return soon)
        if currAgentState.numCarrying >= 5:
            features["carrying-lots-of-food"] = 1.0
            features["distance-to-home"] = dist_home / (walls.width + walls.height)
        else:
            features["carrying-lots-of-food"] = 0.0
            features["distance-to-home"] = 0.0

        # Closest food distance (normalized)
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = dist / (walls.width + walls.height)
        else:
            features["closest-food"] = 0

        # Food density in area (encourages going to food clusters)
        food_in_range = sum(
            1 for f in food.asList() if self.getMazeDistance((next_x, next_y), f) <= 5
        )
        features["food-density"] = food_in_range / 10.0  # Normalize

        # Eating food this turn
        if nextAgentState.numCarrying > currAgentState.numCarrying:
            features["eats-food"] = 1.0
        else:
            features["eats-food"] = 0.0

        # Stop penalty
        if action == Directions.STOP:
            features["stop"] = 1.0
        else:
            features["stop"] = 0.0

        # Reverse direction penalty (going back and forth is inefficient)
        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]
        if action == rev:
            features["reverse"] = 1.0
        else:
            features["reverse"] = 0.0

        # Dead end detection (positions with only one exit are dangerous)
        legal_neighbors = Actions.getLegalNeighbors((next_x, next_y), walls)
        if len(legal_neighbors) <= 1:
            features["in-dead-end"] = 1.0
        else:
            features["in-dead-end"] = 0.0

        return features

    def getOffensiveWeights(self):
        return MixedAgent.QLWeights["offensiveWeights"]

    def getEscapeFeatures(self, gameState, action):
        """
        Enhanced escape features for returning home safely with food.
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        walls = gameState.getWalls()

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        currState = gameState.getAgentState(self.index)

        # Bias
        features["bias"] = 1.0

        # Successfully made it home
        features["onDefense"] = 1.0 if not myState.isPacman else 0.0

        # Distance to home (normalized)
        dist_home = self.getMazeDistance(myPos, self.startPosition)
        features["distanceToHome"] = dist_home / (walls.width + walls.height)

        # Food carrying value (more food = more urgent to escape)
        if myState.numCarrying > 0:
            features["carrying-food"] = myState.numCarrying / 10.0
        else:
            features["carrying-food"] = 0.0

        # Ghost danger (using estimated positions)
        dangerous_ghosts = []
        for enemy_idx in self.getOpponents(successor):
            enemy_state = successor.getAgentState(enemy_idx)
            if not enemy_state.isPacman and enemy_state.scaredTimer == 0:
                est_pos = MixedAgent.ESTIMATED_POSITIONS.get(enemy_idx)
                if est_pos is not None:
                    dist = self.getMazeDistance(myPos, est_pos)
                    if dist <= 5:  # Only care about nearby ghosts
                        dangerous_ghosts.append((dist, est_pos))

        if len(dangerous_ghosts) > 0:
            min_ghost_dist = min([dist for dist, _ in dangerous_ghosts])
            features["enemyDistance"] = min_ghost_dist / (walls.width + walls.height)

            # Critical danger levels
            if min_ghost_dist <= 1:
                features["imminent-danger"] = 1.0
            elif min_ghost_dist <= 2:
                features["close-danger"] = 1.0
            else:
                features["imminent-danger"] = 0.0
                features["close-danger"] = 0.0

            # Ghost blocking path home (between us and home)
            boundary_x = (
                walls.width // 2 - 1
                if gameState.isOnRedTeam(self.index)
                else walls.width // 2
            )
            ghost_blocking = False
            for _, ghost_pos in dangerous_ghosts:
                if gameState.isOnRedTeam(self.index):
                    # Red team: home is on left, need to get past ghosts on right
                    if myPos[0] > ghost_pos[0] > boundary_x:
                        ghost_blocking = True
                else:
                    # Blue team: home is on right, need to get past ghosts on left
                    if myPos[0] < ghost_pos[0] < boundary_x:
                        ghost_blocking = True

            features["ghost-blocking-home"] = 1.0 if ghost_blocking else 0.0
        else:
            features["enemyDistance"] = 1.0  # No ghosts nearby = safe
            features["imminent-danger"] = 0.0
            features["close-danger"] = 0.0
            features["ghost-blocking-home"] = 0.0

        # Progress toward home
        prev_dist_home = self.getMazeDistance(
            currState.getPosition(), self.startPosition
        )
        if dist_home < prev_dist_home:
            features["moving-toward-home"] = 1.0
        else:
            features["moving-toward-home"] = 0.0

        # Dead end danger (especially bad when escaping)
        legal_neighbors = Actions.getLegalNeighbors(myPos, walls)
        if len(legal_neighbors) <= 1:
            features["in-dead-end"] = 1.0
        else:
            features["in-dead-end"] = 0.0

        # Action penalties
        if action == Directions.STOP:
            features["stop"] = 1.0
        else:
            features["stop"] = 0.0

        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]
        if action == rev:
            features["reverse"] = 1.0
        else:
            features["reverse"] = 0.0

        return features

    def getEscapeWeights(self):
        return MixedAgent.QLWeights["escapeWeights"]

    def getDefensiveFeatures(self, gameState, action):
        """
        Enhanced defensive features using belief tracking and strategic positioning.
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        walls = gameState.getWalls()

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Bias
        features["bias"] = 1.0

        # Defense status (strongly prefer staying on defense)
        features["onDefense"] = 1.0 if not myState.isPacman else 0.0

        # Team coordination - distance to ally
        team = [successor.getAgentState(i) for i in self.getTeam(successor)]
        team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())
        features["teamDistance"] = team_dist / (walls.width + walls.height)

        # Invader tracking (using estimated positions from beliefs)
        invaders_with_pos = []
        for enemy_idx in self.getOpponents(successor):
            enemy_state = successor.getAgentState(enemy_idx)
            if enemy_state.isPacman:
                est_pos = MixedAgent.ESTIMATED_POSITIONS.get(enemy_idx)
                if est_pos is not None:
                    invaders_with_pos.append((enemy_idx, est_pos, enemy_state))

        features["numInvaders"] = (
            len(invaders_with_pos) / 2.0
        )  # Normalize (max 2 invaders)

        # Distance to closest invader
        if len(invaders_with_pos) > 0:
            dists = [
                self.getMazeDistance(myPos, inv_pos)
                for _, inv_pos, _ in invaders_with_pos
            ]
            min_dist = min(dists)
            features["invaderDistance"] = min_dist / (walls.width + walls.height)

            # Very close to invader (good for catching)
            if min_dist <= 1:
                features["about-to-catch"] = 1.0
            else:
                features["about-to-catch"] = 0.0
        else:
            features["invaderDistance"] = 1.0  # No invaders = max distance
            features["about-to-catch"] = 0.0

        # Invader carrying food (priority target)
        invader_carrying_food = any(
            inv_state.numCarrying > 0 for _, _, inv_state in invaders_with_pos
        )
        features["invader-has-food"] = 1.0 if invader_carrying_food else 0.0

        # Distance to threatened food (food close to invaders)
        our_food = self.getFoodYouAreDefending(successor).asList()
        if len(invaders_with_pos) > 0 and len(our_food) > 0:
            # Find the food that invaders are ACTUALLY targeting (closest to them)
            closest_threatened_food = None
            min_invader_food_dist = float("inf")

            for _, inv_pos, _ in invaders_with_pos:
                for food in our_food:
                    dist = self.getMazeDistance(inv_pos, food)
                    if dist < min_invader_food_dist:
                        min_invader_food_dist = dist
                        closest_threatened_food = food

            # Now compare: are WE closer to that SAME food than the invader?
            if closest_threatened_food and min_invader_food_dist <= 5:
                my_dist_to_target = self.getMazeDistance(myPos, closest_threatened_food)
                features["distance-to-threatened-food"] = my_dist_to_target / (
                    walls.width + walls.height
                )

                # Key feature: am I closer to the food the enemy is targeting?
                if my_dist_to_target < min_invader_food_dist:
                    features["intercepting-food-threat"] = 1.0
                else:
                    features["intercepting-food-threat"] = 0.0
            else:
                features["distance-to-threatened-food"] = 0.0
                features["intercepting-food-threat"] = 0.0
        else:
            features["distance-to-threatened-food"] = 0.0
            features["intercepting-food-threat"] = 0.0

        # Capsule defense (CRITICAL - capsules give huge advantage)
        our_capsules = self.getCapsulesYouAreDefending(successor)
        if len(our_capsules) > 0 and len(invaders_with_pos) > 0:
            # Find the capsule that invaders are ACTUALLY targeting (closest to them)
            closest_threatened_capsule = None
            min_invader_capsule_dist = float("inf")

            for _, inv_pos, _ in invaders_with_pos:
                for cap in our_capsules:
                    dist = self.getMazeDistance(inv_pos, cap)
                    if dist < min_invader_capsule_dist:
                        min_invader_capsule_dist = dist
                        closest_threatened_capsule = cap

            # Now compare: are WE closer to that SAME capsule than the invader?
            if closest_threatened_capsule and min_invader_capsule_dist <= 4:
                my_dist_to_target_cap = self.getMazeDistance(
                    myPos, closest_threatened_capsule
                )
                features["distance-to-our-capsule"] = my_dist_to_target_cap / (
                    walls.width + walls.height
                )
                features["capsule-under-threat"] = 1.0

                # Key feature: am I closer to the capsule the enemy is targeting?
                if my_dist_to_target_cap < min_invader_capsule_dist:
                    features["protecting-capsule"] = 1.0
                else:
                    features["protecting-capsule"] = 0.0
            else:
                # No immediate capsule threat
                features["distance-to-our-capsule"] = 0.0
                features["capsule-under-threat"] = 0.0
                features["protecting-capsule"] = 0.0
        else:
            # No capsules or no invaders
            features["distance-to-our-capsule"] = 0.0
            features["capsule-under-threat"] = 0.0
            features["protecting-capsule"] = 0.0

        # Blocking escape route (between invader and boundary)
        if len(invaders_with_pos) > 0:
            boundary_x = (
                walls.width // 2 - 1
                if gameState.isOnRedTeam(self.index)
                else walls.width // 2
            )
            blocking_escape = False
            for _, inv_pos, _ in invaders_with_pos:
                if gameState.isOnRedTeam(self.index):
                    if myPos[0] <= inv_pos[0] and myPos[0] >= boundary_x:
                        blocking_escape = True
                else:
                    if myPos[0] >= inv_pos[0] and myPos[0] <= boundary_x:
                        blocking_escape = True
            features["blocking-escape"] = 1.0 if blocking_escape else 0.0
        else:
            features["blocking-escape"] = 0.0

        # Action penalties
        if action == Directions.STOP:
            features["stop"] = 1.0
        else:
            features["stop"] = 0.0

        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]
        if action == rev:
            features["reverse"] = 1.0
        else:
            features["reverse"] = 0.0

        return features

    def getDefensiveWeights(self):
        return MixedAgent.QLWeights["defensiveWeights"]

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
