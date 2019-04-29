import numpy as np
import sys
sys.path.insert(0, '../gym')  # Environment
sys.path.insert(0, '../')  # Baselines
import gym

env_names = [
    # Jaco primitive skills
    'JacoCatch-v1', 'JacoPick-v1', 'JacoToss-v1', 'JacoHit-v1',
    # Jaco complex tasks
    'JacoKeepCatch-v1', 'JacoKeepPick-v1', 'JacoServe-v1',
    # Walker primitive skills
    'Walker2dForward-v1', 'Walker2dBackward-v1', 'Walker2dBalance-v1', 
    'Walker2dJump-v1', 'Walker2dCrawl-v1', 
    # Walker complex tasks
    'Walker2dPatrol-v1', 'Walker2dHurdle-v1', 'Walker2dObstacleCourse-v1',
]

timesteps = 10

for env_name in env_names:
    env = gym.make(env_name)
    print(env_name, env.observation_space, env.action_space)
    env.reset()
    for _ in range(timesteps):
        ob, reward, done, info = env.step(env.action_space.sample())
        print(reward)
        if done:
            env.reset()
