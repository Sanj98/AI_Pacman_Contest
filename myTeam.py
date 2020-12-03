# baselineTeam.py
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint, Counter, Stack

# import numpy as np
# from matplotlib import pyplot as plt
# import matplotlib.patches as patches


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, 
    first = 'OffenceAgent', second = 'DefenceAgent'):
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

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
 
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.mode = 0  # 0-eat pellet, 1-return, 2-escape, 3-defende, 4-penetrate
        self.preferY = 0 # prefered food position, upward of downward 
        self.discount = 0.8 # discount value 
        self.ghosts = [] # enemy normal ghosts
        self.scaredGho = [] # enemy scared ghosts 
        self.oppoPacman = [] # enemy pacman 
        self.caps = [] # position of capsules 
        self.foods = []
        self.lastSpotEnemy = [] # last place spotted enemy
        self.numFoods = len(self.getFood(gameState).asList()) # total number of foods 
        self.border = [] # mid border position
        self.xMid = [] # middle of the map
        self.penetrateCount = 0 
        self.track = {}
        self.trackCount = 0
        self.defenceCount = 0
        self.penetrateGoal = []
        self.mySpace = []
        self.teamMate = {}
        self.teamMate['index'] = ([a for a in self.getTeam(gameState) if not a ==self.index])
        self.walls = gameState.getWalls().asList()
        if self.index%2 == 0:
            borderX = int(gameState.data.layout.width/2-1)
            for i in range(1,borderX+1):
                for j in range(1,gameState.data.layout.height):
                    if (i,j) not in self.walls:
                        self.mySpace.append((i,j))
        else:
            borderX = gameState.data.layout.width/2
            for i in range(int(borderX),int(gameState.data.layout.width - borderX)):
                for j in range(1,gameState.data.layout.height):
                    if (i,j) not in self.walls:
                        self.mySpace.append((i,j))
        self.xMid = borderX
        
        for i in range(1,gameState.data.layout.height):
            if (borderX, i ) not in self.walls:
                self.border.append((borderX,i))
        # print(self.border, self.mySpace)
        # self.border = ([(borderX, i)] for i in range(gameState.data.layout.height) 
            # and not (i in gameState.getWal?ls())) 

    def getSuccessor(self, gameState, action):
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


class CombineAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    
        # ReflexCaptureAgent.registerInitialState(self, gameState)

    def getEnemies(self, gameState):
        enemies = ([gameState.getAgentState(i) for i in self.getOpponents(gameState)])
        ghosts = ([a for a in enemies if not a.isPacman and a.getPosition()])
        scaredGho = ([a for a in enemies if not a.isPacman and a.getPosition() and a.scaredTimer > 0])
        oppoPacman = ([a for a in enemies if a.isPacman and a.getPosition()])
        self.ghosts = ghosts 
        self.scaredGho = scaredGho
        self.oppoPacman = oppoPacman
    
    def getTeamMate(self, gameState):
        teamMatePos = gameState.getAgentState(self.teamMate['index'][0]).getPosition()
        self.teamMate['pos'] = teamMatePos
        self.teamMate['distance'] = self.getMazeDistance(teamMatePos, gameState.getAgentState(self.index).getPosition())
        self.teamMate['pacman'] = gameState.getAgentState(self.teamMate['index'][0]).isPacman

    def getCaps(self, gameState):
        caps = self.getCapsules(gameState)
        self.caps = caps

    def chooseMode(self, gameState):
        self.getEnemies(gameState)
        self.getTeamMate(gameState)
        self.getCaps(gameState)
        self.spotEnemy(gameState)
        numCarrying = gameState.getAgentState(self.index).numCarrying
        self.mode = 0 #initialize mode 
        # number of foods ate by enemies 
        enemyFoodNum = self.numFoods - len(self.getFoodYouAreDefending(gameState).asList()) 
        teamFoodNum = self.numFoods - len(self.getFood(gameState).asList()) 
        difference = enemyFoodNum - teamFoodNum 
        selfState = gameState.getAgentState(self.index)
        if len(self.oppoPacman)>0 and not selfState.isPacman: 
            # self.debugDraw([a.getPosition() for a in self.oppoPacman],[1,0,0])
            # time.sleep(10)
            if self.teamMate['pacman']: # teammate is pacman
                self.mode = 3
                self.penetrateCount = 0
                self.penetrateGoal = []
                return
            else: # teammate and self both ghost 
                if len(self.oppoPacman)==1:
                    distSelf = self.getMazeDistance(self.oppoPacman[0].getPosition(), selfState.getPosition())
                    distTeamMate = self.getMazeDistance(self.teamMate['pos'],self.oppoPacman[0].getPosition())
                    if distSelf < distTeamMate:
                        self.mode = 3
                        self.penetrateCount = 0
                        self.penetrateGoal = []
                        return
                    elif distSelf == distTeamMate:
                        if self.index<max(self.getTeam(gameState)):
                            self.mode = 3
                            self.penetrateCount = 0
                            self.penetrateGoal = []
                            return
                else: 
                    if self.index<max(self.getTeam(gameState)):
                            self.mode = 3
                            self.penetrateCount = 0
                            self.penetrateGoal = []
                            return
                         
        
        if self.penetrateCount > 0 :
            self.mode = 4 
            self.penetrateCount -= 1
        # elif difference > 3: #and self.getScore(gameState)<0:
        #     if self.index<max(self.getTeam(gameState)):
        #         self.mode = 3
        
        elif (len(self.getFood(gameState).asList()) <=  2):
            if selfState.isPacman:
                self.mode = 1 # if only 2 food left, go back 
            else:
                self.mode = 3
        elif len(self.ghosts)>0:    # has ghost nearby 
            pacman = selfState.isPacman
            distToEnemy = []
            for enemy in self.ghosts:
                if enemy.getPosition():
                    distToEnemy.append([self.getMazeDistance
                        (selfState.getPosition(),enemy.getPosition())])
                minDist = min(distToEnemy) # distance to the closest ghost
                if minDist[0] < 5:
                    # if (selfState.getPosition()[0]<=self.border[0][0]+1 and
                    #      selfState.getPosition()[0]>=self.border[0][0]-1 and selfState.numCarrying==0):
                    #      self.mode = 4
                    #     #  time.sleep(3)
                    #      self.penetrateCount = 5
                    if pacman:
                        self.mode = 2 #escape mode
                    else:
                        self.mode = 4
                        self.penetrateCount = 5#gameState.data.layout.width/2 + gameState.data.layout.height/2 
    
    def closerToGoal(self, gameState, pos, goal):
        goalPos = list([goal])
        action = self.aStar(gameState, goalPos)
        if action and not action == 0:
            return action 
        else:
            return random.choice(gameState.getLegalActions(self.index))
    
    def getHuristic2(self, successorState, goalState):
        heuristic = 0
        pos = successorState.getAgentState(self.index).getPosition()
        heuristic += self.getMazeDistance(pos, goalState[0])*(10)

        enemies = ([a.getPosition() for a in self.ghosts])
        if len(enemies)>0:
            for enemy in enemies:
                heuristic += self.getMazeDistance(pos, enemy)*(-20)
        corner = self.deadCorner(successorState, pos)
        heuristic += corner*(10)
        if pos == self.start:
            heuristic += 9999

        return heuristic

    def returnBorder(self, gameState):
        pos = gameState.getAgentState(self.index).getPosition()
        border = self.border 
        distance = ([self.getMazeDistance(pos, item) for item in border])
        # print(pos,border,distance,'--------------')
        minDist = min(distance)
        if minDist == 0:
            goal = ([self.lastSpotEnemy])
        else:
            goal = [a for a,v in zip(border, distance) if v == minDist] # find the closest border 
        action = self.closerToGoal(gameState, pos, goal[0])
        if action:
            return action
        else:
            actions = gameState.getLegalActions(self.index)
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                if not successor.getAgentState(self.index).isPacman:
                    return action
            return random.choice(gameState.getLegalActions(self.index))

    def toClosestFood(self, gameState):
        # if gameState.getAgentState(self.index).numCarrying >= self.numFoods/4:
        #     return self.returnBorder(gameState)
        if len(self.scaredGho)>0:
            if self.scaredGho[0].scaredTimer < 5:
                return self.escapePath(gameState)
        pos = gameState.getAgentState(self.index).getPosition()
        foodList = self.foods
        if foodList:
            distance = ([(self.getMazeDistance(pos,food) + abs(food[1]-self.preferY)) for food in foodList])
            minDist = min(distance)
            goal = [a for a,v in zip(foodList, distance) if v == minDist]
            action = self.closerToGoal(gameState, pos, goal[0])
            if action:
                return action 
            else: 
                return random.choice(gameState.getLegalActions(self.index))
        else:
            return self.returnBorder(gameState)

    # def drawMDP(self, gameState, valueDic):
    #     data = np.zeros((gameState.data.layout.height, gameState.data.layout.width))
    #     nonzero_idxs = np.where(data == 0)[0]
    #     data[nonzero_idxs] = -1000
    #     for item in valueDic.keys():
    #         data[gameState.data.layout.height-1-item[1]][item[0]]= valueDic[item]
    #     fig, ax = plt.subplots()
    #     pos = gameState.getAgentState(self.index).getPosition()
    #     data[gameState.data.layout.height-1-int(pos[1])][int(pos[0])]= 0

    #     im = ax.imshow(data,vmin=-200, vmax=200)
    #     for i in range(gameState.data.layout.height):
    #         for j in range(gameState.data.layout.width):
    #             if (j,gameState.data.layout.height-1-i) in self.walls:
    #                 continue
    #             text = ax.text(j, i, "{:.2f}".format(data[i, j]),
    #                         ha="center", va="center")

    #     fig.tight_layout()
    #     plt.show()



    def escapePath(self, gameState):
        timer = time.time() 
        valueDic, positionNeedCal = self.getMap(gameState)
        iteration = 100 #set iteration time 
        while iteration > 0: 
            oldValueDic = valueDic.copy()
            for position in positionNeedCal:
                valueDic[position] = self.trasition(position, 0, oldValueDic)
            iteration -= 1 
        # choose best direction 
        # self.drawMDP(gameState,valueDic)
        # for item in valueDic.keys():
        #     if valueDic[item]>0:
        #         self.debugDraw(item,[0,(200-valueDic[item])/200,0],False)
        #     else:
        #         self.debugDraw(item,[(200+valueDic[item])/200,0,0],False)
        pos = gameState.getAgentState(self.index).getPosition()
        north = (pos[0],pos[1]+1)
        south = (pos[0],pos[1]-1)
        west = (pos[0]-1,pos[1])
        east = (pos[0]+1, pos[1])
        posValDic = {}
        if north in valueDic.keys():
            posValDic[Directions.NORTH] = valueDic[north]
            # print("north", valueDic[north])
        if south in valueDic.keys():
            posValDic[Directions.SOUTH] = valueDic[south]
            # print("south", valueDic[south])
        if west in valueDic.keys(): 
            posValDic[Directions.WEST] = valueDic[west]
            # print('west', valueDic[west])
        if east in valueDic.keys():
            posValDic[Directions.EAST] = valueDic[east]
            # print('east'), valueDic[east]
        # print(valueDic, pos)
        optvalue = max(posValDic.values())
        optAction = max(posValDic, key=lambda k: posValDic[k])
        return optAction

    def trasition(self, pos, reward, valueDic):
        north = (pos[0],pos[1]+1)
        south = (pos[0],pos[1]-1)
        west = (pos[0]-1,pos[1])
        east = (pos[0]+1, pos[1])
        posValDic = {}
        noise= 0.2

        # to north 
        if north in valueDic.keys():
            posValN = (1-noise)*(reward + self.discount*valueDic[north])
        else:
            posValN = (1-noise)*(reward + self.discount*valueDic[pos])
        if west in valueDic.keys(): 
            posValN += noise/3*(reward + self.discount*valueDic[west])
        else: 
            posValN += noise/3*(reward + self.discount*valueDic[pos])
        if east in valueDic.keys(): 
            posValN += noise/3*(reward + self.discount*valueDic[east])
        else: 
            posValN += noise/3*(reward + self.discount*valueDic[pos])
        if south in valueDic.keys(): 
            posValN += noise/3*(reward + self.discount*valueDic[south])
        else: 
            posValN += noise/3*(reward + self.discount*valueDic[pos])

        posValDic['North'] = posValN

        # to south
        if south in valueDic.keys():
            posValS = (1-noise)*(reward + self.discount*valueDic[south])
        else:
            posValS = (1-noise)*(reward + self.discount*valueDic[pos])
        if west in valueDic.keys(): 
            posValS += noise/3*(reward + self.discount*valueDic[west])
        else: 
            posValS += noise/3*(reward + self.discount*valueDic[pos])
        if east in valueDic.keys(): 
            posValS += noise/3*(reward + self.discount*valueDic[east])
        else: 
            posValS += noise/3*(reward + self.discount*valueDic[pos])
        if north in valueDic.keys(): 
            posValS += noise/3*(reward + self.discount*valueDic[north])
        else: 
            posValS += noise/3*(reward + self.discount*valueDic[pos])

        posValDic['South'] = posValS

        # to west 
        if west in valueDic.keys():
            posValW = (1-noise)*(reward + self.discount*valueDic[west])
        else:
            posValW = (1-noise)*(reward + self.discount*valueDic[pos])
        if north in valueDic.keys(): 
            posValW += noise/3*(reward + self.discount*valueDic[north])
        else: 
            posValW += noise/3*(reward + self.discount*valueDic[pos])
        if south in valueDic.keys(): 
            posValW += noise/3*(reward + self.discount*valueDic[south])
        else: 
            posValW += noise/3*(reward + self.discount*valueDic[pos])
        if east in valueDic.keys(): 
            posValW += noise/3*(reward + self.discount*valueDic[east])
        else: 
            posValW += noise/3*(reward + self.discount*valueDic[pos])

        posValDic['West'] = posValW

        # to east
        if east in valueDic.keys():
            posValE = (1-noise)*(reward + self.discount*valueDic[east])
        else:
            posValE = (1-noise)*(reward + self.discount*valueDic[pos])
        if north in valueDic.keys(): 
            posValE += noise/3*(reward + self.discount*valueDic[north])
        else: 
            posValE += noise/3*(reward + self.discount*valueDic[pos])
        if south in valueDic.keys(): 
            posValE += noise/3*(reward + self.discount*valueDic[south])
        else: 
            posValE += noise/3*(reward + self.discount*valueDic[pos])
        if west in valueDic.keys(): 
            posValE += noise/3*(reward + self.discount*valueDic[west])
        else: 
            posValE += noise/3*(reward + self.discount*valueDic[pos])

        posValDic['East'] = posValE

        return max(posValDic.values())

    def getMap(self, gameState):
        valueDic = {}
        positionNeedCal = []
        pos = gameState.getAgentState(self.index).getPosition()
        for i in range(1,gameState.data.layout.width-1):
            for j in range(1,gameState.data.layout.height-1):
                if ((not gameState.hasWall(i,j)) and (not (i,j)==pos)):
                    value = self.getValue(gameState, (i,j))
                    if value == 0:
                        positionNeedCal.append((i,j))
                    valueDic[(i,j)] = value 
        return valueDic, positionNeedCal
    
    def getValue(self, gameState, pos):
        ghosts = ([a.getPosition() for a in self.ghosts])
        ghostArea = []
        for ghost in ghosts:
            if ghost not in [a.getPosition() for a in self.scaredGho]:
                ghostArea.append(ghost)
                ghostArea.append ((ghost[0],ghost[1]+1))
                ghostArea.append ((ghost[0],ghost[1]-1))
                ghostArea.append ((ghost[0]-1,ghost[1]))
                ghostArea.append ((ghost[0]+1, ghost[1]))
        if pos in ghostArea:
            return -100
        if pos in [a.getPosition() for a in self.scaredGho]:
            return 40
        if self.deadCorner(gameState, pos)==2:
            return -100
        if pos in self.border:
            return 100
        # if self.deadCorner(gameState, pos)==1:
        #     return -50

        if pos in self.foods:
            return 0
        if pos in self.caps:
            return 100

        return 0
    
    def deadCorner(self, gameState, pos):
        corner = []
        if gameState.hasWall(int(pos[0]),int(pos[1]+1)):
            corner += 'N'
        if gameState.hasWall(int(pos[0]),int(pos[1]-1)):
            corner += 'S'
        if gameState.hasWall(int(pos[0]-1),int(pos[1])):
            corner += 'W'
        if gameState.hasWall(int(pos[0]+1), int(pos[1])):
            corner += 'E'
        deadCorner = [ ['N','W'],['N','E'],['S','W'],['S','E']]
        if len(corner)<2:
            return 0
        elif len(corner)==2:
            for item in deadCorner:
                if set(set(item)).issubset(set(corner)):
                    return 1
            return 0
        elif len(corner)>2:
            return 2

    def defence(self, gameState):
        actions = gameState.getLegalActions(self.index)
        optAction = random.choice(actions)
        pos = gameState.getAgentState(self.index).getPosition()
        scaredTimer = gameState.getAgentState(self.index).scaredTimer
        if scaredTimer > 10: 
            return self.toClosestFood(gameState)

        elif gameState.getAgentState(self.index).isPacman:
            optAction = self.returnBorder(gameState)
        else:
            if len(self.oppoPacman)>0: # if spot opponent pacman
                oppoPacmans = self.oppoPacman
                distToEnemies = ([self.getMazeDistance(pos, enemy.getPosition()) for enemy in oppoPacmans])
                minDist = min(distToEnemies)
                closestEnemy = ([a.getPosition() for a,v in zip(oppoPacmans, distToEnemies) if v==minDist])[0]     
                if minDist <= 2:
                    if gameState.getAgentState(self.index).scaredTimer>0: # spot enemy, but scared 
                        optAction = self.maintainDistance(gameState, pos, closestEnemy)
                    else:
                        enemyNextMove = self.enemyNextMove(gameState, closestEnemy)
                        optAction = self.chaseEnemy(gameState, pos, random.choice(enemyNextMove))
                else:
                    optAction = self.chaseEnemy(gameState,pos,closestEnemy)
            else: # no spotted enemies 
                # print('no spotted enemmies............',self.lastSpotEnemy)
                if len(self.lastSpotEnemy)>0 :
                    goal = self.lastSpotEnemy[0]
                    optAction = self.chaseEnemy(gameState, pos, goal)
                    # print('chase enemy', optAction,'---------------')
                elif len(self.getFoodYouAreDefending(gameState).asList())>0:
                    optAction = self.toMyClosestFood(gameState)
                else:
                    optAction = random.choice(actions)
        return optAction

    def maintainDistance(self, gameState, pos, enemy):
        currentDist = self.getMazeDistance(pos, enemy)
        actions = gameState.getLegalActions(self.index)
        possibleAction = []
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            newPos = successor.getAgentState(self.index).getPosition()
            newDist = self.getMazeDistance(newPos, enemy)
            if newDist > currentDist:
                possibleAction.append(action)
        if len(possibleAction)>0:
            return random.choice(possibleAction)
        else:
            return random.choice(actions)

        



    def toMyClosestFood(self, gameState):  # defence mode, to my team's closest food
        pos = gameState.getAgentState(self.index).getPosition()
        foods = self.getFoodYouAreDefending(gameState).asList()
        minDist = 0
        nearestFood = None
        for food in foods:
            distance = self.getMazeDistance(food,pos)
            if distance > minDist:
                minDist = distance
                nearestFood = food
        action = self.closerToGoal(gameState,pos,nearestFood)
        return action


    def enemyNextMove(self, gameState, pos):
        north = (pos[0],pos[1]+1)
        south = (pos[0],pos[1]-1)
        west = (pos[0]-1,pos[1])
        east = (pos[0]+1, pos[1])
        posiblePos = []
        posiblePos.append(pos)
        for a in [north, south, west, east]:
            if a in self.mySpace and a not in self.walls:
                posiblePos.append(a)
        return posiblePos

    def spotEnemy(self, gameState):
        previousState = self.getPreviousObservation()
        if previousState:
            preFoods = self.getFoodYouAreDefending(previousState).asList()
            curFoods = self.getFoodYouAreDefending(gameState).asList()
            missingPellet = ([x for x in preFoods if x not in curFoods])
            if len(missingPellet)==1:
                self.lastSpotEnemy = ([missingPellet[0]])
    
    def chaseEnemy(self, gameState, pos, enemyPos):
        actions = gameState.getLegalActions(self.index)
        currentDist = self.getMazeDistance(pos, enemyPos)
        for action in actions: 
            successor = self.getSuccessor(gameState, action)
            newDist = self.getMazeDistance(successor.getAgentState(self.index).getPosition(), enemyPos)
            if newDist<currentDist:
                return action
        return random.choice(actions)

    def penetrate(self, gameState):
        goalPos = (self.border).copy()
        for item in goalPos:
            if self.deadCorner(gameState,item)==2:
                goalPos.remove(item)
        if len(self.ghosts)>0:
            enemy = self.ghosts[0].getPosition()
            temp = goalPos.copy()
            for goal in goalPos:
                if abs(goal[1]-enemy[1])<=3:
                    temp.remove(goal)
            goalPos = temp
            if len(goalPos)>0:
                action = self.aStar(gameState, goalPos)
                self.penetrateGoal = goalPos
            else:
                action = self.defence(gameState)
        elif len(self.penetrateGoal)>0:
            action = self.aStar(gameState, self.penetrateGoal)
        else:
            action = self.defence(gameState)

        if action and not action==0:
            return action 
        elif action == 0:
            action = self.toClosestFood(gameState)
            self.penetrateCount = 0
            self.penetrateGoal = []
            return action
        else:
            return random.choice(gameState.getLegalActions(self.index))
        
    def aStar(self, gameState, goalState):
        # self.debugDraw(goalState,(1,1,1)) # DEBUG
        queue = util.PriorityQueue()
        queue.push((gameState,[],0),0)
        closedList = []
        fVal = {}
        while not queue.isEmpty():
            currentState, path, cost = queue.pop()
            pos = currentState.getAgentState(self.index).getPosition()
            if (not pos in closedList):
                closedList.append(pos)
                if pos in goalState:
                    if len(path)>0:
                        return path[0]
                    else:
                        return 0
                actions = currentState.getLegalActions(self.index)
                for action in actions:
                    successorState = self.getSuccessor(currentState, action)
                    newPos = successorState.getAgentState(self.index).getPosition()
                    if self.mode == 4:
                        heuristic = cost + 1 + self.getHuristic(successorState, goalState)
                    else:
                        heuristic = cost + 1 + self.getHuristic2(successorState, goalState)
                    if (newPos in fVal.keys()) and (heuristic>= fVal[newPos]):
                        continue 
                    fVal[newPos]= heuristic
                    queue.push((successorState,path+[action], cost+1), heuristic)

    def getHuristic(self, gameState, goalState):
        heuristic = 0 
        distToGoal = 9999
        pos = gameState.getAgentState(self.index).getPosition()
        if pos == self.start:
            heuristic += 9999
        for goal in goalState:
            dist = self.getMazeDistance(goal,pos)
            if dist < distToGoal:
                distToGoal = dist
        heuristic += distToGoal*10
        distToEnemy = 9999 
        distToPacman = 9999
        selfState = gameState.getAgentState(self.index)
        if ((selfState.scaredTimer>0) and (not selfState.isPacman)):
            for item in self.oppoPacman:
                dist= self.getMazeDistance(pos, item.getPosition())
                if dist < distToPacman:
                    distToPacman = dist
            heuristic += distToPacman*(-10)
        if len(self.ghosts)>0:          
            for item in self.ghosts:
                dist = self.getMazeDistance(pos, item.getPosition())
                if dist < distToEnemy:
                    distToEnemy = dist
            heuristic += distToEnemy*(-10)
        else:
            heuristic -= 50
        if pos not in self.mySpace:
            heuristic += 100    
        return heuristic




    def chooseAction(self, gameState):
        # self.debugClear()  # DEBUG
        self.chooseMode(gameState)
        self.foods = self.getFood(gameState).asList()
        optAction = None
        # if (gameState.getAgentState(self.index).getPosition() == self.start):
        #     print('agent',self.index,'die')
        # print('Agent:',self.index,'=======', "mode:", self.mode, gameState.getAgentState(self.index).getPosition())
        if self.mode == 0: #eat food 
            optAction = self.toClosestFood(gameState)
        elif self.mode == 1: #return 
            optAction = self.returnBorder(gameState)
        elif self.mode == 2: #escape
            optAction = self.escapePath(gameState)
            # time.sleep(1)
        elif self.mode == 3: #defence 
            optAction = self.defence(gameState)
            # time.sleep(1)
        elif self.mode == 4: #penetrate 
            optAction = self.penetrate(gameState)
            # time.sleep(3)
        if self.trackCount<4:
            self.track[self.trackCount]= gameState.getAgentState(self.index).getPosition()
            self.trackCount += 1
        else:
            self.trackCount = 0
            self.track[self.trackCount]= gameState.getAgentState(self.index).getPosition()
        if len(self.track)==4:
            if self.track[0]==self.track[2] and self.track[1]==self.track[3]:
                return(random.choice(gameState.getLegalActions(self.index)))
        return optAction
    
class OffenceAgent(CombineAgent):
  def registerInitialState(self, gameState):
    ReflexCaptureAgent.registerInitialState(self, gameState)
    self.preferY = gameState.data.layout.height

class DefenceAgent(CombineAgent):
    def registerInitialState(self, gameState):
        ReflexCaptureAgent.registerInitialState(self, gameState)
        self.preferY = 0