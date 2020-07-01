from time import time, localtime, strftime
import os

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from stable_baselines.bench import Monitor
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

from env.blimp import BlimpEnv


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

	        # Retrieve training reward
	        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
	        if len(x) > 0:
	            # Mean training reward over the last 100 episodes
	            mean_reward = np.mean(y[-100:])
	            if self.verbose > 0:
	              print("Num timesteps: {}".format(self.num_timesteps))
	              print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

	            # New best model, you could save the agent here
	            if mean_reward > self.best_mean_reward:
	                self.best_mean_reward = mean_reward
	                # Example for saving best model
	                if self.verbose > 0:
	                  print("Saving new best model at {} timesteps".format(x[-1]))
	                  print("Saving new best model to {}.zip".format(self.save_path))
	                self.model.save(self.save_path)

        return True

def main(logdir):
	# params
	SLEEP_RATE = 100 #100Hz  
	N_EPISODE = 10000 
	EPISODE_LENGTH = SLEEP_RATE*30 #30 sec
	TOTAL_TIMESTEPS = EPISODE_LENGTH * N_EPISODE

	# logdir
	logdir = os.path.join(logdir, strftime("%Y-%m-%d--%H:%M:%S", localtime()))
	os.makedirs(logdir)	
	checkpoint_path = os.path.join(logdir,'checkpoint')
	callback_path = logdir
	final_model_path = logdir+'/final_model'

	# env
	env = BlimpEnv(SLEEP_RATE)
	env = Monitor(env, logdir)
	# env = make_vec_env(lambda: env, n_envs=1, monitor_dir=logdir)
	print("Observation space:", env.observation_space)
	print("Shape:", env.observation_space.shape)
	print("Action space:", env.action_space)

	# callback
	SAVE_FREQ = EPISODE_LENGTH*1 # every 1 episode
	checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, 
											save_path=checkpoint_path,
	                                        name_prefix='sac_callback_model')
	save_on_best_training_reward_callback = SaveOnBestTrainingRewardCallback(
											check_freq=SAVE_FREQ, 
											log_dir = callback_path)	
	callback = CallbackList([
		checkpoint_callback, 
		save_on_best_training_reward_callback])

	# agent
	model = SAC(MlpPolicy, env, gamma=0.98, 
		learning_rate=0.0003, buffer_size=1000000, learning_starts=EPISODE_LENGTH*5, 
		train_freq=1, batch_size=256, tau=0.01, ent_coef='auto', 
		target_update_interval=1, gradient_steps=1, target_entropy='auto', 
		action_noise=None, verbose=1, tensorboard_log=logdir, 
		_init_setup_model=True)

	print("---------- Start Learing -----------")
	model.learn(total_timesteps=TOTAL_TIMESTEPS, 
				log_interval=SAVE_FREQ, 
				callback=callback)
	
	print("---------- Finish Learning ----------")
	model.save(final_model_path)
	del model # remove to demonstrate saving and loading
	model = SAC.load(final_model_path)

	results_plotter.plot_results([logdir], TOTAL_TIMESTEPS, results_plotter.X_TIMESTEPS, "SAC BLIMP")
	plt.show()

if __name__ == "__main__":

	logdir = '/home/yliu_local/blimpRL_ws/src/RL_log/sac_log'
	
	main(logdir)