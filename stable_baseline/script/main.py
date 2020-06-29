import gym
import numpy as np

from stable_baselines.common.env_checker import check_env
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

from env.blimp import BlimpEnv

env = BlimpEnv()
print("Observation space:", env.observation_space)
print("Shape:", env.observation_space.shape)
print("Action space:", env.action_space)
# check_env(env, warn=True)

model = SAC(MlpPolicy, env, verbose=0)
model.learn(total_timesteps=120*60*4, log_interval=10)
model.save("sac_blimp")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_blimp")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    print("action=",action)
    obs, rewards, dones, info = env.step(action[0])
