# Load in the gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

# Reset the environment to begin playing/training
state = env.reset()

# Load in some libraries that we will need
import copy
import numpy as np
import tensorflow as tf
from ModelPPO import PPO

# We are defining the function to get the Generalized Advantage Estimation
def get_gaes(rewards, state_values, next_state_values, GAMMA, LAMBDA):
    deltas = [r_t + GAMMA * next_v - v for r_t, next_v, v in zip(rewards, next_state_values, state_values)]
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):
        gaes[t] = gaes[t] + LAMBDA * GAMMA * gaes[t + 1]
    return gaes, deltas

# We are getting some dimensions to define the shape of our placeholders for our computation graph
state_shape = state.shape
frame_height = state.shape[0]
frame_width = state.shape[1]
channels = state.shape[2]
n_actions = len(SIMPLE_MOVEMENT)

# This refers to the number of actions we want to look back at (refer to the end of my blog)
timesteps = 4

# Some hyperparameters for the optimization
epsilon = 0.2
learning_rate = 5 * 10e-5
GAMMA = 0.99
LAMBDA = 0.95

# Initialize the model
tf.reset_default_graph()
sess = tf.Session()
model = PPO(sess, frame_height, frame_width, channels, n_actions, timesteps, epsilon, learning_rate, GAMMA, LAMBDA)
sess.run(tf.global_variables_initializer())

# Define how many episodes we want to run it for
total_episodes = 200
episode_counter = 0
done = False

# Reshaping the data to be the input to our model (we divide by 255 to put the pixel value between 0 and 1)
current_state = state.reshape(1, frame_height, frame_width, channels) / 255

# Initialize the empty lists, which we will use for mini-batch gradient descent
states = []
prev_actions_list = []
actions = []
state_values = []
rewards = []

# You should probably increase the number of epochs
epochs = 1
batch_size = 64

# Some performance stats to keep track of
performance = []
stage_max_position = 0
true_max_position = 0
last_level = 1
max_position_tracker = []

while episode_counter < total_episodes:
    if done:
        # Total distance traveled since you started
        score = true_max_position + max_position_tracker[-1]
        
        next_state_values = state_values[1:] + [0]        
        GAEs, deltas = get_gaes(rewards, state_values, next_state_values, GAMMA, LAMBDA)
        
        # Convert Lists to Numpy Arrays for Training
        states = np.reshape(states, newshape = [-1] + list(state.shape))
        prev_actions_list = np.array(prev_actions_list)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_state_values = np.array(next_state_values)        
        GAEs = np.array(GAEs)
        GAEs = (GAEs - GAEs.mean()) / GAEs.std()
        
        # Update the old model parameters to start off training (since we need to calculate the ratio between new and old)
        model.update()
        
        # Train the Model
        model.train(states, actions, GAEs, rewards, next_state_values, prev_actions_list, epochs, batch_size)

        # Append most recent score to the performance tracker and reinitialize everything for the next episode
        performance.append(score)
        stage_max_position = 0
        true_max_position = 0
        last_level = 1
        max_position_tracker = []
        state = env.reset()
        current_state = state.reshape(1, frame_height, frame_width, channels) / 255

        states = []
        prev_actions_list = []
        actions = []
        state_values = []
        rewards = []
        
        # Print out a few numbers to keep track of things
        print("Episode: {}, Cumulative Reward: {}".format(episode_counter, performance[-1]))
        episode_counter += 1

    # Get your inputs ready for the model
    prev_state = np.copy(state)
    prev_actions = np.array([0]*(timesteps - min(len(actions), timesteps)) + actions[-timesteps:])
    prev_actions = np.hstack((np.eye(n_actions)[prev_actions[0]], np.eye(n_actions)[prev_actions[1]], 
                              np.eye(n_actions)[prev_actions[2]], np.eye(n_actions)[prev_actions[3]]))
                              
    # Extract the policy and state-value from the model
    policy, value = sess.run([model.policy, model.value], {model.inputs: current_state, 
                             model.previous_actions: np.expand_dims(prev_actions, axis = 0), model.training: False})
    
    # Policy is a probabilistic output, so we sample from it to get our action
    action = np.random.choice(np.arange(n_actions), 1, p = policy.reshape(-1))[0]
    
    # We take the action in the environment and receive the next state, reward, and some other useful info
    state, reward, done, info = env.step(action)    
    
    # Reformat our state to use as input to get the next action
    current_state = state.reshape(1, frame_height, frame_width, channels) / 255
    
    # The max absolute reward we can get is 15, so we make sure the reward is between -1 and 1
    reward /= 15
    
    # Seeing if we made it to a new level to start a new counter for mario's max position
    if info['stage'] != last_level:
        last_level = info['stage']
        true_max_position += stage_max_position
        stage_max_position = 0
    
    # Appending the data to be used in training
    states.append(prev_state)
    prev_actions_list.append(prev_actions)
    actions.append(action)
    state_values.append(value[0,0])
    rewards.append(reward)        
    
    # Tracking the max position in the level
    if info['x_pos'] > stage_max_position:
        stage_max_position = info['x_pos']
    max_position_tracker.append(stage_max_position)    

    # This is to render the actual game (Comment this out for a speed boost in training)
    env.render()

env.close()
