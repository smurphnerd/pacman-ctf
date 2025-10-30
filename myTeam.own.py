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
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os
from capture import GameState, noisyDistance
from game import Directions, Actions, AgentState, Agent
from util import nearestPoint
import sys, os

# the folder of current file.
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

from lib_piglet.utils.pddl_solver import pddl_solver
from lib_piglet.domains.pddl import pddl_state
from lib_piglet.utils.pddl_parser import Action

import belief_tracking

CLOSE_DISTANCE = 4
MEDIUM_DISTANCE = 10
LONG_DISTANCE = 25

#################
# Team creation #
#################


def createTeam(
    firstIndex, secondIndex, isRed, first="SmurphAgent", second="SmurphAgent"
):
    # Create shared belief tracking dictionary that both agents will use
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########


class SmurphAgent(CaptureAgent):
    """
    This is an agent that use pddl to guide the high level actions of Pacman
    """

    # Shared opponent belief distributions (class variable shared between teammates)
    OPPONENT_BELIEFS = {}

    # Default weights for q learning, if no QLWeights.txt find, we use the following weights.
    # You should add your weights for new low level planner here as well.
    # weights are defined as class attribute here, so taht agents share same weights.
    QLWeights = {
        "offensiveWeights": {
            "closest-food": -1,
            "bias": 1,
            "#-of-ghosts-1-step-away": -100,
            "successorScore": 100,
            "chance-return-food": 10,
        },
        "defensiveWeights": {
            "numInvaders": -1000,
            "onDefense": 100,
            "teamDistance": 2,
            "invaderDistance": -10,
            "stop": -100,
            "reverse": -2,
        },
        "escapeWeights": {
            "onDefense": 1000,
            "enemyDistance": 30,
            "stop": -100,
            "distanceToHome": -20,
        },
    }
    QLWeightsFile = BASE_FOLDER + "/QLWeightsMyTeam.txt"

    # Also can use class variable to exchange information between agents.
    CURRENT_ACTION = {}

    def __init__(self, index):
        """Initialize agent with shared belief tracking."""
        super().__init__(index)

    def registerInitialState(self, gameState: GameState):
        self.pddl_solver = pddl_solver(BASE_FOLDER + "/smurphTeam.pddl")
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

        # Use a dictionary to save information about current agent.
        SmurphAgent.CURRENT_ACTION[self.index] = {}

        # Initialize shared beliefs using centralized function
        # Only initialize once per team (both teammates share the same dict)
        if not SmurphAgent.OPPONENT_BELIEFS:
            beliefs = belief_tracking.initialize_beliefs(gameState)
            # Only store beliefs for our opponents
            opponents = self.getOpponents(gameState)
            for opp_idx in opponents:
                SmurphAgent.OPPONENT_BELIEFS[opp_idx] = beliefs[opp_idx]

    def final(self, gameState: GameState):
        pass

    def chooseAction(self, gameState: GameState):
        """
        This is the action entry point for the agent.
        In the game, this function is called when its current agent's turn to move.

        We first pick a high-level action.
        Then generate low-level action ("North", "South", "East", "West", "Stop") to achieve the high-level action.
        """

        # Update belief tracking for opponents
        updated_beliefs = belief_tracking.update_all_beliefs(
            SmurphAgent.OPPONENT_BELIEFS,
            gameState,
            self.index,
        )

        # Update the shared dict (teammate will see updated beliefs)
        opponents = self.getOpponents(gameState)
        for opp_idx in opponents:
            SmurphAgent.OPPONENT_BELIEFS[opp_idx] = updated_beliefs[opp_idx]

        # -------------High Level Plan Section-------------------
        # Get high level action from a pddl plan.

        # Collect objects and init states from gameState
        objects, initState = self.get_pddl_state(gameState)
        positiveGoal, negtiveGoal = self.getGoals(objects, initState)

        # Check if we can stick to current plan
        if not self.stateSatisfyCurrentPlan(initState, positiveGoal, negtiveGoal):
            # Cannot stick to current plan, prepare goals and replan
            print("Agnet:", self.index, "compute plan:")
            print(
                "\tOBJ:" + str(objects),
                "\tINIT:" + str(initState),
                "\tPOSITIVE_GOAL:" + str(positiveGoal),
                "\tNEGTIVE_GOAL:" + str(negtiveGoal),
                sep="\n",
            )
            self.highLevelPlan: List[Tuple[Action, pddl_state]] = self.getHighLevelPlan(
                objects, initState, positiveGoal, negtiveGoal
            )  # Plan is a list Action and pddl_state
            self.currentActionIndex = 0
            self.lowLevelPlan = []  # reset low level plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            print("\tPLAN:", self.highLevelPlan)
        if len(self.highLevelPlan) == 0:
            raise Exception(
                "Solver retuned empty plan, you need to think how you handle this situation or how you modify your model "
            )

        # Get next action from the plan
        highLevelActionObj = self.highLevelPlan[self.currentActionIndex][0]
        highLevelAction = highLevelActionObj.name
        SmurphAgent.CURRENT_ACTION[self.index] = highLevelActionObj

        print("Agent:", self.index, highLevelAction)

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
        print("\tAgent:", self.index, lowLevelAction)
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

        myPos = gameState.getAgentPosition(self.index)
        myObj = "a{}".format(self.index)

        # Collect winning states
        currentScore = gameState.data.score
        winning = False
        if gameState.isOnRedTeam(self.index):
            if currentScore > 0:
                states.append(("winning",))
                winning = True
        else:
            if currentScore < 0:
                states.append(("winning",))
                winning = True

        # Collect team agents states
        agents: List[Tuple[int, AgentState]] = [
            (i, gameState.getAgentState(i)) for i in self.getTeam(gameState)
        ]

        # Get teammate info
        teammate_idx = (self.index + 2) % 4

        for agent_index, agent_state in agents:
            agent_object = "a{}".format(agent_index)
            agent_type = "current_agent" if agent_index == self.index else "ally"
            objects += [(agent_object, agent_type)]

            if agent_state.scaredTimer > 0:
                states.append(("is_scared", agent_object))

            if agent_state.isPacman:
                states.append(("is_pacman", agent_object))

                # Check if carrying enough to win
                food_needed = 0 if winning else abs(currentScore) + 1
                if agent_state.numCarrying >= food_needed:
                    states.append(("can_win_with_backpack", agent_object))

        # Collect enemy agents states
        enemies: List[Tuple[int, AgentState]] = [
            (i, gameState.getAgentState(i)) for i in self.getOpponents(gameState)
        ]
        noisyDistance = gameState.getAgentDistances()
        typeIndex = 1

        # Get beliefs for better enemy tracking
        enemy_beliefs = {}
        for enemy_index, enemy_state in enemies:
            if enemy_index in SmurphAgent.OPPONENT_BELIEFS:
                enemy_beliefs[enemy_index] = SmurphAgent.OPPONENT_BELIEFS[enemy_index]

        for enemy_index, enemy_state in enemies:
            enemy_position = enemy_state.getPosition()
            enemy_object = "e{}".format(enemy_index)
            objects += [(enemy_object, "enemy{}".format(typeIndex))]

            if enemy_state.scaredTimer > 0:
                states.append(("is_scared", enemy_object))

            if enemy_state.isPacman:
                states.append(("is_pacman", enemy_object))

                # Check if enemy has enough food to win
                food_needed = 0 if not winning else abs(currentScore) + 1
                if enemy_state.numCarrying >= food_needed:
                    states.append(("can_win_with_backpack", enemy_object))

            # Check enemy_around for each teammate
            if enemy_position != None:
                for agent_index, agent_state in agents:
                    agent_pos = agent_state.getPosition()
                    if (
                        agent_pos
                        and self.getMazeDistance(agent_pos, enemy_position)
                        <= CLOSE_DISTANCE
                    ):
                        states.append(
                            ("enemy_around", enemy_object, "a{}".format(agent_index))
                        )

            # Check if enemy passed us using belief distribution
            if enemy_state.isPacman and enemy_index in SmurphAgent.OPPONENT_BELIEFS:
                enemy_belief = SmurphAgent.OPPONENT_BELIEFS[enemy_index]
                # Enemy has passed if belief sum behind us (deeper in territory) >= 0.5
                if self.hasEnemyPassed(myPos, enemy_belief, gameState, threshold=0.5):
                    states.append(("enemy_passed", enemy_object, myObj))

            typeIndex += 1

        # Check cooperative predicates
        if teammate_idx is not None:
            teammate_state = gameState.getAgentState(teammate_idx)
            teammate_obj = "a{}".format(teammate_idx)
            teammate_pos = teammate_state.getPosition()

            # Use belief-based comparison to determine who has more enemies around
            if teammate_pos and SmurphAgent.OPPONENT_BELIEFS:
                walls = gameState.getWalls()
                teammate_has_more = self.compareEnemyProximity(
                    myPos, teammate_pos, SmurphAgent.OPPONENT_BELIEFS, walls
                )
                if teammate_has_more:
                    states.append(("more_enemies_around", teammate_obj))

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

    def getGoals(
        self, objects: List[Tuple], initState: List[Tuple]
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        This function is the strategic brain. It checks for situations
        in a priority order and sets the NEXT LOGICAL PDDL GOAL.

        Returns:
            Tuple[List[Tuple], List[Tuple]]: (positiveGoalList, negativeGoalList)
        """

        # --- 1. Helper variables and state parsing ---
        myObj = next((obj[0] for obj in objects if obj[1] == "current_agent"), None)
        teammateObj = next((obj[0] for obj in objects if obj[1] == "ally"), None)
        teammateIdx = (self.index + 2) % 4
        enemies = [obj[0] for obj in objects if "enemy" in obj[0]]

        assert myObj is not None, "Should not happen, but good to be safe"
        assert len(enemies) == 2, "Should not happen, but good to be safe"

        # Use a set for fast, easy lookups of the current state
        initStateSet = set(initState)

        # Check current status
        is_winning = ("winning",) in initStateSet
        am_i_pacman = ("is_pacman", myObj) in initStateSet

        if is_winning:
            critical_threats = []
            normal_threats = []
            for e_obj in enemies:
                # Skip non-invaders
                if ("is_pacman", e_obj) not in initStateSet:
                    continue

                if ("can_win_with_backpack", e_obj) in initStateSet:
                    critical_threats.append(e_obj)
                else:
                    normal_threats.append(e_obj)

            all_threats = critical_threats + normal_threats

            for threat_obj in all_threats:
                currentAction = SmurphAgent.CURRENT_ACTION.get(teammateIdx)
                if not (
                    currentAction
                    and currentAction.name == "prevent_escape"
                    and currentAction.parameters[1] == threat_obj
                ):
                    return [], [("is_pacman", threat_obj)]

            negativeGoals = [("is_pacman", e_obj) for e_obj in enemies]
            return [("defend_foods",)], negativeGoals

        else:

            if am_i_pacman or ("more_enemies_around", teammateObj) in initStateSet:
                negative_goals = []
                for e_obj in enemies:
                    negative_goals.append([("enemy_around", e_obj, myObj)])
                return [("winning",)], negative_goals
            else:
                negativeGoals = []
                for e_obj in enemies:
                    if ("is_pacman", e_obj) in initStateSet:
                        if ("can_win_with_backpack", e_obj) in initStateSet:
                            return [], [("is_pacman", e_obj)]

                        negativeGoals.append([("is_pacman", e_obj)])
                if negativeGoals:
                    return [("defend_foods",)], negativeGoals
                else:
                    return [("winning",)], []

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
        # The following classification of high level actions is only a example.
        # You should think and use your own way to design low level planner.
        ##########
        if highLevelAction == "invade":
            # The q learning process for offensive actions are complete,
            # you can improve getOffensiveFeatures to collect more useful feature to pass more information to Q learning model
            # you can improve the getOffensiveReward function to give reward for new features and improve the trainning process .
            rewardFunction = self.getOffensiveReward
            featureFunction = self.getOffensiveFeatures
            weights = self.getOffensiveWeights()
            learningRate = self.alpha
        elif highLevelAction == "go_home":
            # The q learning process for escape actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getEscapeReward
            featureFunction = self.getEscapeFeatures
            weights = self.getEscapeWeights()
            learningRate = 0  # learning rate set to 0 as reward function not implemented for this action, do not do q update,
        else:
            # The q learning process for defensive actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getDefensiveReward
            featureFunction = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()
            learningRate = 0  # learning rate set to 0 as reward function not implemented for this action, do not do q update

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
                    values.append(
                        (
                            self.getQValue(featureFunction(gameState, action), weights),
                            action,
                        )
                    )
                action = max(values)[1]
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

    def getSpacingOutPenalty(self, gameState: GameState, nextState: GameState):
        # Reward to encourage not stacking on top of each other
        # This reward shouldn't be linear, we just don't want to be within CLOSE_DISTANCE
        # too much
        pass

    def getAroundEnemyPenalty(self, gameState: GameState, nextState: GameState):
        # Negative reward for being too close to an enemy (within CLOSE_DISTANCE)
        pass

    def getInvadeReward(self, gameState: GameState, nextState: GameState):
        getSpacingOutPenalty = self.getSpacingOutPenalty(gameState, nextState)

        # Reward for distance to enemy territory

        # We want them to enter the enemy territory with space to maneuver
        aroundEnemyPenalty = self.getAroundEnemyPenalty(gameState, nextState)

        pass

    def getEatFoodUntilWinningReward(self, gameState: GameState, nextState: GameState):
        # Reward for distance to closest food

        # Reward for eating food
        pass


    def getTakeLeadReward(self, gameState: GameState, nextState: GameState):
        # Reward for distance to home territory
        pass

    def getEscapeEnemyReward(self, gameState: GameState, nextState: GameState):
        # Penalty for close to enemy
        aroundEnemyPenalty = self.getAroundEnemyPenalty(gameState, nextState)

        # Reward for distance to home territory (we want to go back home)

    def getGoHomeReward(self, gameState: GameState, nextState: GameState):
        # Reward for distance to home territory
        pass


    def getOffensiveReward(self, gameState: GameState, nextState: GameState):
        # Calculate the reward.
        currentAgentState: AgentState = gameState.getAgentState(self.index)
        nextAgentState: AgentState = nextState.getAgentState(self.index)

        ghosts = self.getGhostLocs(gameState)
        ghost_1_step = sum(
            nextAgentState.getPosition()
            in Actions.getLegalNeighbors(g, gameState.getWalls())
            for g in ghosts
        )

        base_reward = -50 + nextAgentState.numReturned + nextAgentState.numCarrying
        new_food_returned = nextAgentState.numReturned - currentAgentState.numReturned
        score = self.getScore(nextState)

        if ghost_1_step > 0:
            base_reward -= 5
        if score < 0:
            base_reward += score
        if new_food_returned > 0:
            # return home with food get reward score
            base_reward += new_food_returned * 10

        print("Agent ", self.index, " reward ", base_reward)
        return base_reward

    def getDefensiveReward(self, gameState, nextState):
        print(
            "Warnning: DefensiveReward not implemented yet, and learnning rate is 0 for defensive ",
            file=sys.stderr,
        )
        return 0

    def getEscapeReward(self, gameState, nextState):
        print(
            "Warnning: EscapeReward not implemented yet, and learnning rate is 0 for escape",
            file=sys.stderr,
        )
        return 0

    # ------------------------------- Feature Related Action Functions -------------------------------

    def getOffensiveFeatures(self, gameState: GameState, action):
        food = self.getFood(gameState)
        currAgentState = gameState.getAgentState(self.index)

        walls = gameState.getWalls()
        ghosts = self.getGhostLocs(gameState)

        # Initialize features
        features = util.Counter()
        nextState = self.getSuccessor(gameState, action)

        # Successor Score
        features["successorScore"] = (
            self.getScore(nextState) / (walls.width + walls.height) * 10
        )

        # Bias
        features["bias"] = 1.0

        # Get the location of pacman after he takes the action
        next_x, next_y = nextState.getAgentPosition(self.index)

        # Number of Ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts
        )

        dist_home = (
            self.getMazeDistance(
                (next_x, next_y), gameState.getInitialAgentPosition(self.index)
            )
            + 1
        )

        features["chance-return-food"] = (currAgentState.numCarrying) * (
            1 - dist_home / (walls.width + walls.height)
        )  # The closer to home, the larger food carried, more chance return food

        # Closest food
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = dist / (walls.width + walls.height)
        else:
            features["closest-food"] = 0

        return features

    def getOffensiveWeights(self):
        return SmurphAgent.QLWeights["offensiveWeights"]

    def getEscapeFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features["onDefense"] = 1
        if myState.isPacman:
            features["onDefense"] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemiesAround = [
            a for a in enemies if not a.isPacman and a.getPosition() != None
        ]
        if len(enemiesAround) > 0:
            dists = [
                self.getMazeDistance(myPos, a.getPosition()) for a in enemiesAround
            ]
            features["enemyDistance"] = min(dists)

        if action == Directions.STOP:
            features["stop"] = 1
        features["distanceToHome"] = self.getMazeDistance(myPos, self.startPosition)

        return features

    def getEscapeWeights(self):
        return MixedAgent.QLWeights["escapeWeights"]

    def getDefensiveFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features["onDefense"] = 1
        if myState.isPacman:
            features["onDefense"] = 0

        team = [successor.getAgentState(i) for i in self.getTeam(successor)]
        team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())
        features["teamDistance"] = team_dist

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features["numInvaders"] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features["invaderDistance"] = min(dists)

        if action == Directions.STOP:
            features["stop"] = 1
        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction
        ]
        if action == rev:
            features["reverse"] = 1

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

    def hasEnemyPassed(self, my_pos, enemy_belief, gameState, threshold=0.5):
        """
        Check if an enemy has passed us based on their belief distribution.
        Enemy has passed if the sum of beliefs behind us (deeper in our territory) exceeds threshold.

        Args:
            my_pos: Current agent position (x, y)
            enemy_belief: 2D belief distribution for the enemy
            gameState: Current game state
            threshold: Minimum belief sum to consider enemy as "passed" (default 0.5)

        Returns:
            True if enemy belief sum behind us >= threshold
        """
        belief_sum_behind = 0.0
        width = len(enemy_belief)
        height = len(enemy_belief[0]) if width > 0 else 0

        if gameState.isOnRedTeam(self.index):
            # We're red, our territory is left side (x < width/2)
            # Enemy passed if they're more left than us (smaller x)
            for x in range(width):
                if x < my_pos[0]:  # Behind us (deeper in our territory)
                    for y in range(height):
                        belief_sum_behind += enemy_belief[x][y]
        else:
            # We're blue, our territory is right side (x >= width/2)
            # Enemy passed if they're more right than us (larger x)
            for x in range(width):
                if x > my_pos[0]:  # Behind us (deeper in our territory)
                    for y in range(height):
                        belief_sum_behind += enemy_belief[x][y]

        return belief_sum_behind >= threshold

    def compareEnemyProximity(self, pos1, pos2, opponent_beliefs, walls):
        """
        Compare which position has more enemies nearby using belief distributions.
        For each opponent, uses synchronized BFS to determine which position is closer.

        Args:
            pos1: First agent position (x, y)
            pos2: Second agent position (x, y)
            opponent_beliefs: Dict mapping opponent_idx -> belief distribution
            walls: Wall grid

        Returns:
            True if pos2 has more enemies around than pos1
            False otherwise (including ties)
        """
        pos1_score = 0
        pos2_score = 0

        # Compare proximity for each opponent
        for opponent_idx, belief_dist in opponent_beliefs.items():
            # Synchronized BFS for this opponent's belief distribution
            fringe1 = [(pos1[0], pos1[1], 0)]
            fringe2 = [(pos2[0], pos2[1], 0)]
            expanded1 = set()
            expanded2 = set()

            sum1 = 0.0
            sum2 = 0.0
            current_dist = 0

            winner = None  # Will be 1 or 2 when someone wins

            while (fringe1 or fringe2) and winner is None:
                # Process all nodes at current distance for agent 1
                next_fringe1 = []
                while fringe1 and fringe1[0][2] == current_dist:
                    pos_x, pos_y, dist = fringe1.pop(0)

                    if (pos_x, pos_y) in expanded1:
                        continue
                    expanded1.add((pos_x, pos_y))

                    # Add belief probability at this position
                    if 0 <= pos_x < len(belief_dist) and 0 <= pos_y < len(
                        belief_dist[0]
                    ):
                        sum1 += belief_dist[pos_x][pos_y]

                    # Expand to neighbors
                    nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
                    for nbr_x, nbr_y in nbrs:
                        next_fringe1.append((nbr_x, nbr_y, dist + 1))

                # Process all nodes at current distance for agent 2
                next_fringe2 = []
                while fringe2 and fringe2[0][2] == current_dist:
                    pos_x, pos_y, dist = fringe2.pop(0)

                    if (pos_x, pos_y) in expanded2:
                        continue
                    expanded2.add((pos_x, pos_y))

                    # Add belief probability at this position
                    if 0 <= pos_x < len(belief_dist) and 0 <= pos_y < len(
                        belief_dist[0]
                    ):
                        sum2 += belief_dist[pos_x][pos_y]

                    # Expand to neighbors
                    nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
                    for nbr_x, nbr_y in nbrs:
                        next_fringe2.append((nbr_x, nbr_y, dist + 1))

                # After processing this distance level, check if either sum > 1.0
                if sum1 > 0.5 or sum2 > 0.5:
                    # Both exceed on same iteration - compare sums
                    if sum1 > 0.5 and sum2 > 0.5:
                        if sum2 > sum1:
                            winner = 2
                        elif sum1 > sum2:
                            winner = 1
                        # If exactly equal, winner stays None (tie)
                    # Only one exceeds - they win
                    elif sum2 > 0.5:
                        winner = 2
                    else:  # sum1 > 1.0
                        winner = 1

                # Move to next distance level
                fringe1 = next_fringe1
                fringe2 = next_fringe2
                current_dist += 1

            # Update scores based on winner for this opponent
            if winner == 1:
                pos1_score += 1
            elif winner == 2:
                pos2_score += 1
            # Ties don't add to either score

        # pos2 has more enemies around if they won more comparisons
        return pos2_score > pos1_score

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
