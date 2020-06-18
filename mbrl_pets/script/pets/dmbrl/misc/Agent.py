from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
# from gym.monitoring import VideoRecorder
from dotmap import DotMap
# from dmbrl.misc import logger

import time

class Agent:
    """An general class for RL agents.
    """
    def __init__(self, params):
        """Initializes an agent.

        Arguments:
            params:
                .env: The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the
                    action noise if params.noisy_actions is True.
        """
        self.env = params.env

        # load the imitation data if needed
        if hasattr(self.env, '_expert_data_loaded') and \
                (not self.env._expert_data_loaded):
            self.env.load_expert_data(
                params.params.misc.ctrl_cfg.il_cfg.expert_amc_dir
            )

        self.noise_stddev = params.noise_stddev if params.get("noisy_actions", False) else None

        if isinstance(self.env, DotMap):
            raise ValueError("Environment must be provided to the agent at initialization.")
        if (not isinstance(self.noise_stddev, float)) and params.get("noisy_actions", False):
            raise ValueError("Must provide standard deviation for noise for noisy actions.")

        if self.noise_stddev is not None:
            self.dU = self.env.action_space.shape[0]
        self._debug = 1

    def sample(self, horizon, policy, record_fname=None):
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.
            record_fname: (str/None) The name of the file to which a recording of the rollout
                will be saved. If None, the rollout will not be recorded.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
        times, rewards = [], []
        O, A, reward_sum, done = [self.env.reset()], [], 0, False

        policy.reset()
        for t in range(horizon):
            start = time.time()

            A.append(policy.act(O[t], t))
            times.append(time.time() - start)
            # print(self.action)

            if self.noise_stddev is None:
                # print(t, A[t], A[t+1])
                obs, reward, done, info = self.env.step(A[t])
            else:
                action = A[t] + np.random.normal(loc=0, scale=self.noise_stddev,
                                                 size=[self.dU])
                action = np.minimum(np.maximum(action,
                                               self.env.ac_lb),
                                    self.env.ac_ub)
                obs, reward, done, info = self.env.step(action)
            O.append(obs)  
            reward_sum += reward
            rewards.append(reward)
            if done:
                break

            self.env.RATE.sleep()

        self.record = False
        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }
