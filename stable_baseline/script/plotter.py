from stable_baselines import results_plotter
import matplotlib.pyplot as plt

log_dir = '/home/yliu_local/blimpRL_ws/src/RL_log/sac_log/2020-06-30--20:35:30/'
SLEEP_RATE = 100
N_EPISODE = 5 
EPISODE_LENGTH = SLEEP_RATE*60 #60 sec
TOTAL_TIMESTEPS = EPISODE_LENGTH * N_EPISODE

results_plotter.plot_results([log_dir], TOTAL_TIMESTEPS, results_plotter.X_TIMESTEPS, "SAC BLIMP")
plt.show()