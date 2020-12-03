# Pacman Capture the Flag Contest

This project was done as part of my master's degree in IT in the subject - "AI Planning for Autonomy" with 2 other classmates. The details can be found in TEAM.md 

The purpose of this project is to implement a Pacman Autonomous Agent that can play and compete in the UoM Pacman Capture the Flag tournament.

## Techniques used 

1. Q-learning
2. Monte Carlo Tree Search with Score Maxmising
3. MDP with A* search

For the final submission, MDP with A* was submitted due its high performance as compared to the other two techniques. The code for this can be found in *myteam.py*.

## Application of each technique

### I. Q-Learning(Offense)

In this technique, we build the offensive agent by following the basic rule:

If there is no enemy, the agent will find and eat the closest pellets.
If there is an enemy nearby, the agent will choose to eat capsule if there is any, run away, or go back to home.
If the number of pellets less than two, the agent will go back directly or stay at home.
Following are the features according to our rules.

**eatFo**: when there is no enemy nearby, the agent should move to the nearest pellets. this feature will be the distance to the closest pellets.

**eatCap**: when there is an enemy nearby, the agent should try to eat capsule. This feature will be the distance to the closest capsules if capsules are not been eaten.

**runAway**: when there is an enemy nearby, the agent should move away from the enemy. This feature will be the distance to the closest enemy.

**goBack**: when there is an enemy nearby or the number of pellets less than two, the agent should go back to home. This feature will be the distance to closest border position. And the closest border position will not change until the agent goes back.

**returnFo**: if the agent carries pellets go back to home in next step, the feature will be one. Otherwise, it will be zero. This feature aims to encourage agent go back with pellets.

**returnNo**: if the agent does not carry pellets and go back to home in next step, the feature will be one. Otherwise, it will be zero. This feature aims to discourage agent go back home without carrying pellets.

**deadEnd**: if agent surrounded by three walls in next step, the feature will be one, otherwise, it will be zero. This feature aims to discourage agent to a dead end.

**stop**: if next action is 'Stop', the feature will be one. Otherwise, it will be zero. We don't want the agent stop. Therefore, we set this feature.

**eatenByGho**: if agent eats by a ghost in next step, the feature will be one. Otherwise, it will be zero.

To prevent weights change rapidly, all features are normalized by dividing them by the product of 100 and the area of the map.

We also define rewards, which are shown below:

Success Score increase: +100
Eat Capsule: +50
Eaten by ghost: -100
Away from ghost: +1
Carry pellets increased: +1
In dead end: -3
Overall, this approach performs well in combating baseline team with 80% winning rate. However, the winning rate is quite low when combat with the agent that use MDP and A* approach. Therefore, we choose to improve MDP with A* approach in the end.

### II. Monte Carlo Tree Search(Offense)

It begins with estimating parameters such as distance to the food and distance to the ghosts in the maze. Following this, a weight is set for each parameter which together with the moves is fed into a Monte Carlo Tree Search simulation. This simulation returns the best possible action as an outcome. The agent also tries to estimate if a capsule is available and heads for the same. Additionally, it also includes the functionality of returning home if the agent has been able to score some food considering the main objective of an offensive agent is to bring maximum food home by avoiding the ghosts. Hence, it calculates distance to a closest ghost as mentioned previously to avoid it whenever and wherever possible.

### III. Score Maximising(Defense)

Our core strategy here is to be able to locate our opponents by calculating the distance to them and choose the best action such that our agent chases these opponents and ultimately, eats them. Another aspect that the agent takes into consideration is the location of the food spread across the maze as every opponent is heading for the food. Moreover, the agent is also able to realise if the food has been eaten by the opponent and identify the location of the next food such that it can go after the opponent and get in its way. In conclusion, the defensive agent is simply chasing the opponents depending on their location as well as the location of the next food. Consequently, also choosing the best action depending on the cost and above calculations.

### IV. MDP with A*(Both offense and defense)

**A** - In this method, the A* is built with a priority queue. With given goal states and current state, it will calculate the f(n) for each state. Then push the state in the priority queue with priority f(n). The queue will pop out one state every loop. The A* function will loop until the path to the goal was found or the priority queue is empty. It will return the first action of the found path or return random possible action if no possible path to the goal was found. The offence, defense, return and penetrate mode are using the A* searching function. But the heuristic function they use is different which are as follows - 

1. **Offense mode**: The offense mode is the initial mode for agents. In the offense mode, the agent finds the closest pellet and heading to it. To avoid two agents target on the same pellet, each agent was assigned a preferred Y-axis value. As shown in the above image, the two white blocks are the target of two agents respectively. One prefers pellets with higher Y while another agent prefer lower Y. An important thing to apply A* search is the heuristic function. For this mode, the distance to enemy ghost and the distance to the pellet are considered when calculating the heuristic value. The heuristic is calculate by: H = Dist(Pos, Goal)*10 +Dist(Pos, Ghost)*(-10).

2. **Defense mode**: Similar to offence mode, the defense mode is first to allocate position the enemy Pacman and then using A* to find the path. To locate the enemy Pacman, it first will try to use the getPosition() function. If can't get the enemy's position, it will set the last disappeared pellet as the goal. The heuristic is calculated by: H = Dist(Pos, Goal)*10.

3. **Penetrate mode**: The penetrate mode is used to penetrate the enemy's border while there is an enemy guarding their border. As shown in the above image, the white blocks are the goals. The heuristic is similar to the offence mode. Only add one more parameter: whether the path is within the self border. If not, add a large value to the heuristic because it will attract the enemy ghost which is unwanted.

4. **Return mode**: The return mode is used when only two pellets left. The agent will set the closest mid-border block as the goal position, then applying the A* to find the path. The heuristic function is as the same as offence mode.

**MDP** - The value iteration MDP was used in this method for escaping. As shown in the below image, the value iteration is conducted for the whole map every time. The position of the enemy and its adjacent blocks are set as -100. The mid-border blocks and capsule position are set as 100. Additionally, the blocks have walls in three out of four directions also has value -100. It is because we want the agent to avoid entering the dead-end alley while escaping. The pellet is not considered while escaping. More specifically, the discount of the MDP is set as '0.8', the noise is set as '0.2', and the reward for each step is set as 0. The value map below shows the value of each block after 100 iterations. The Q value of each state is calculated by: V = Sigma(P(s,s')*(R(s,s')+discount*V(s')).

### Acknowledgements

This is [Pacman Capture the Flag Contest](http://ai.berkeley.edu/contest.html) from the set of [UC Pacman Projects](http://ai.berkeley.edu/project_overview.html). 
