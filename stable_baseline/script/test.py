from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
from env.blimp import BlimpEnv

model_path = '/home/yliu_local/blimpRL_ws/src/RL_log/sac_log/2020-06-30--20:35:30/final_model.zip'

model = SAC.load(model_path)

SLEEP_RATE = 100
EPISODE_LENGTH = SLEEP_RATE*60 #60 sec

env = BlimpEnv(SLEEP_RATE, EPISODE_LENGTH)
obs = env.reset()

reward_sum = 0.0
for i in range(EPISODE_LENGTH):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    reward_sum += rewards

print("reward_sum=",reward_sum/SLEEP_RATE)