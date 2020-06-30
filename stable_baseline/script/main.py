from time import time, localtime, strftime
import os

import gym
import numpy as np

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

from env.blimp import BlimpEnv


def main(logdir):
	# params
	SLEEP_RATE = 100
	SAVE_FREQ = 10

	# logdir
	tensorboard_logdir = logdir+'/tensorboard'
	logdir = os.path.join(logdir, strftime("%Y-%m-%d--%H:%M:%S", localtime()))	

	# env
	env = BlimpEnv(SLEEP_RATE)
	env = make_vec_env(lambda: env, n_envs=1)
	print("Observation space:", env.observation_space)
	print("Shape:", env.observation_space.shape)
	print("Action space:", env.action_space)

	# callback
	checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=logdir+'/model',
	                                         name_prefix='sac_model')

	eval_callback = EvalCallback(env, best_model_save_path=logdir+'/best_model',
                             log_path=logdir+'/results', eval_freq=10) 
	callback = CallbackList([checkpoint_callback, eval_callback])

	# agent
	One_Episode = SLEEP_RATE*30 #30 sec
	model = SAC(MlpPolicy, env, gamma=0.98, 
		learning_rate=0.0003, buffer_size=1000000, learning_starts=10000, 
		train_freq=1, batch_size=256, tau=0.01, ent_coef='auto', 
		target_update_interval=1, gradient_steps=1, target_entropy='auto', 
		action_noise=None, random_exploration=0.0, verbose=0, tensorboard_log=tensorboard_logdir, 
		_init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None)

	model.learn(total_timesteps=200000, log_interval=10, 
		tb_log_name="test_run",
		callback=callback)
	model.save("sac_blimp")

	# del model # remove to demonstrate saving and loading
	# model = SAC.load("sac_blimp")

	print("train finished")
	obs = env.reset()
	reward_sum = 0.0
	while 100*60:
	    action, _states = model.predict(obs)
	    obs, rewards, dones, info = env.step(action)
	    reward_sum += rewards

	print("reward_sum=",reward_sum)


if __name__ == "__main__":

	logdir = '/home/yliu_local/blimpRL_ws/src/RL_log/sac_log'

	# parser = argparse.ArgumentParser()
 #    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[])
	# parser.add_argument('-o', '--override', action='append', nargs=2, default=[])
 #    parser.add_argument('-logdir', type=str, default=logdir)
 #    args = parser.parse_args()
	
	main(logdir)