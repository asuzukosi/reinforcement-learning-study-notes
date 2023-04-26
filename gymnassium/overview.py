# import the gymnasium package
import gymnasium as gym
import panda_gym
import time

# setup the environment and the observation and info
env = gym.make("PandaStack-v3", render_mode="human", render_width=200,render_height=200,)
observation, info = env.reset(seed=42)

print("The action space is ", env.action_space)

for _ in range(10000):
    action = env.action_space.sample() # the policy to specify the action will be placed here
    time.sleep(0.5)
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
        
        
env.close()
    

