import gym
import numpy as np

from stable_baselines.common.env_checker import check_env
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

from env.blimp import BlimpEnv

env = BlimpEnv()
# check_env(env, warn=True)

model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=120*60*8, log_interval=10)
model.save("sac_blimp")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_blimp")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action[0])
