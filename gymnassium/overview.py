# import the gymnasium package
import gymnasium as gym
import panda_gym
import time

# setup the environment and the observation and info
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

print("The action space is ", env.action_space)

for _ in range(10000):
    action = env.action_space.sample() # the policy to specify the action will be placed here
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
        
        
env.close()
    

