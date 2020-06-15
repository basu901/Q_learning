import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') #used to import cv2
import cv2 
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import time
from matplotlib import style

style.use("ggplot")

SIZE = 10

HM_EPISODES = 25000

#Parameters for learning
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.5
EPS_DECAY = 0.999
SHOW_EVERY = 1000

start_q_table = None #If a Q table is loaded from pickle we can use it here

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {1:(255,175,0), #blueish color for PLAYER_N
     2:(0,255,0), #green color for FOOD_N
     3:(0,0,255)} #red color for ENEMY_N

#Creating a Blob object
class Blob:
    # Although we are generating the blobs on the grid randomly,
    #it might happen that 2 blocks end up on the same position
    #You can fix that later
    def __init__(self):
        self.x = np.random.randint(0,SIZE)
        self.y = np.random.randint(0,SIZE)

    def __str__(self):
        return "{,}".format(self.x,self.y)

    #Performing operator overloading
    def __sub__(self,other):
        return (self.x-other.x,self.y-other.y)

    def action(self,choice):
        if choice == 0:
            self.move(x = 1, y = 1)
        elif choice == 1:
            self.move(x = -1, y = -1)
        elif choice == 2:
            self.move(x = -1, y = 1)
        elif choice == 3:
            self.move(x = -1, y = -1)

    def move(self,x = False, y = False):
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y


        #Fixing out of bound issues
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


if start_q_table is None:
    q_table = {}
    for i in range(-SIZE+1,SIZE):
        for ii in range(-SIZE+1,SIZE):
            for iii in range(-SIZE+1,SIZE):
                for iiii in range(-SIZE+1,SIZE):
                    q_table[((i,ii),(iii,iiii))] = [np.random.uniform(-5,0) for i in range(4)]
else:
    with open(start_q_table,"rb") as f:
        q_table = pickle.load(f)


episode_rewards = []

for episode in range(HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY==0:
        print("on #{}, epsilon is {}".format(episode,epsilon))
        print("{} ep mean :".format(SHOW_EVERY,np.mean(episode_rewards[-SHOW_EVERY:])))
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)

        if np.random.random()>epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)

        player.action(action)

        if player.x == enemy.x and player.y == enemy.y:
            reward = - ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        new_obs = (player-food,player-enemy) #new observation or start_q_table
        max_future_q = np.max(q_table[new_obs]) #max q_value for the next observation
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)


        if show:
            env = np.zeros((SIZE,SIZE,3), dtype = np.uint8) #starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N] #sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N] #sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N] #sets the enemy location to red
            img = Image.fromarray(env,'RGB') #reading to rgb. Apparently. Even though color definitions are bgr.
            img = img.resize((300,300)) #resizing the image
            cv2.imshow("image",np.array(img)) #display the image

            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY: #code to hang at the end if we reach abrupt end for good reasons or not
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        episode_reward += reward

        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    episode_rewards.append(episode_reward)

    epsilon *= EPS_DECAY



moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY,mode = 'valid')

plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel("Reward {}ma".format(SHOW_EVERY))
plt.xlabel("episode #")
plt.show()

with open("qtable-{}.pickle".format(int(time.time())),"wb") as f:
    pickle.dump(q_table,f)
