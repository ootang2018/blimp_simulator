from stable_baselines import results_plotter


results_plotter.plot_results(["./log"], 10e6, results_plotter.X_TIMESTEPS, "Breakout")
