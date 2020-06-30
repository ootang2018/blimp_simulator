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


class PlottingCallback(BaseCallback):
    """
    Callback for plotting the performance in realtime.

    :param verbose: (int)
    """
    def __init__(self, verbose=1):
        super(PlottingCallback, self).__init__(verbose)
        self._plot = None

    def _on_step(self) -> bool:
        # get the monitor's data
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if self._plot is None: # make the plot
            plt.ion()
            fig = plt.figure(figsize=(6,3))
            ax = fig.add_subplot(111)
            line, = ax.plot(x, y)
            self._plot = (line, ax, fig)
            plt.show()
        else: # update and rescale the plot
            self._plot[0].set_data(x, y)
            self._plot[-2].relim()
            self._plot[-2].set_xlim([self.locals["total_timesteps"] * -0.02, 
                                     self.locals["total_timesteps"] * 1.02])
            self._plot[-2].autoscale_view(True,True,True)
            self._plot[-1].canvas.draw()

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

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

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
	SLEEP_RATE = 100

	N_EPISODE = 5 
	EPISODE_LENGTH = SLEEP_RATE*60 #60 sec

	# logdir
	logdir = os.path.join(logdir, strftime("%Y-%m-%d--%H:%M:%S", localtime()))	
	final_model_path = logdir+'/final_model'
	checkpoint_model_path = logdir+'/model'
	best_model_path = logdir+'/best_model'

	# env
	env = BlimpEnv(SLEEP_RATE)
	env = Monitor(env, logdir)
	# env = make_vec_env(lambda: env, n_envs=1, monitor_dir=logdir)
	print("Observation space:", env.observation_space)
	print("Shape:", env.observation_space.shape)
	print("Action space:", env.action_space)

	# callback
	SAVE_FREQ = EPISODE_LENGTH*5 # every 5 episode
	checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, 
											save_path=checkpoint_model_path,
	                                        name_prefix='sac_model')
	save_on_best_training_reward_callback = SaveOnBestTrainingRewardCallback(
											check_freq=SAVE_FREQ, 
											log_dir = best_model_path)
	callback = CallbackList([
		checkpoint_callback, 
		save_on_best_training_reward_callback])

	# agent
	model = SAC(MlpPolicy, env, gamma=0.98, 
		learning_rate=0.0003, buffer_size=1000000, learning_starts=10000, 
		train_freq=1, batch_size=256, tau=0.01, ent_coef='auto', 
		target_update_interval=1, gradient_steps=1, target_entropy='auto', 
		action_noise=None, verbose=1, tensorboard_log=logdir, 
		_init_setup_model=True)

	print("---------- Start Learing -----------")
	model.learn(total_timesteps=N_EPISODE*EPISODE_LENGTH, 
				log_interval=SAVE_FREQ, 
				callback=callback)
	
	print("---------- Finish Learning ----------")
	model.save(final_model_path)

	del model # remove to demonstrate saving and loading
	model = SAC.load(final_model_path)

	obs = env.reset()
	reward_sum = 0.0
	for i in range(EPISODE_LENGTH):
	    action, _states = model.predict(obs)
	    obs, rewards, dones, info = env.step(action)
	    reward_sum += rewards

	print("reward_sum=",reward_sum/100)


if __name__ == "__main__":

	logdir = '/home/yliu_local/blimpRL_ws/src/RL_log/sac_log'

	# parser = argparse.ArgumentParser()
 #    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[])
	# parser.add_argument('-o', '--override', action='append', nargs=2, default=[])
 #    parser.add_argument('-logdir', type=str, default=logdir)
 #    args = parser.parse_args()
	
	main(logdir)