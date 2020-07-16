from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import numpy as np
import scipy.stats as stats
from dmbrl.misc import logger

from .optimizer import Optimizer


class CEMOptimizer(Optimizer):
    """A Tensorflow-compatible CEM optimizer.
    """

    def __init__(self, sol_dim, max_iters, popsize, num_elites, delay_step, tf_session=None,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25, dU=1,
                 params=None):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            tf_session (tf.Session): (optional) Session to be used for this optimizer. Defaults to None,
                in which case any functions passed in cannot be tf.Tensor-valued.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        from dmbrl.modeling.models import GT_dynamics
        self._gt_compile_cost = GT_dynamics.compile_cost
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites, self.dU, self.delay_step = \
            sol_dim, max_iters, popsize, num_elites, dU, delay_step
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha
        self.tf_sess = tf_session
        self.debug = False

        self._params = params

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        if self.tf_sess is not None:
            with self.tf_sess.graph.as_default():
                with tf.variable_scope("CEMSolver"):
                    self.init_mean = tf.placeholder(dtype=tf.float32, shape=[sol_dim])
                    self.init_var = tf.placeholder(dtype=tf.float32, shape=[sol_dim])
                    self.first_action = tf.placeholder(dtype=tf.float32, shape=[1, self.delay_step*dU])

        self.num_opt_iters, self.mean, self.var = None, None, None
        self.tf_compatible, self.cost_function = None, None
        

        if self._params.il_cfg.use_gt_dynamics:
            self._dynamics = GT_dynamics.GT(self._params)

    def setup(self, cost_function, tf_compatible):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        if tf_compatible and self.tf_sess is None:
            raise RuntimeError("Cannot pass in a tf.Tensor-valued cost function without passing in a TensorFlow "
                               "session into the constructor")

        self.tf_compatible = tf_compatible

        if not tf_compatible:
            self.cost_function = cost_function
        else:
            def continue_optimization(t, mean, var, best_val, best_sol):
                return tf.logical_and(tf.less(t, self.max_iters), tf.reduce_max(var) > self.epsilon)

            def iteration(t, mean, var, best_val, best_sol):
                lb_dist, ub_dist = mean - self.lb, self.ub - mean #mean might be wrong
                constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var)
                samples = tf.truncated_normal([self.popsize, self.sol_dim], mean, tf.sqrt(constrained_var))
                # samples = tf.Print(samples, [tf.shape(samples[1,:])])
                samples = tf.concat(axis=1, values=[tf.tile(self.first_action, [self.popsize, 1]), samples])
                # samples = tf.Print(samples, [tf.shape(samples[1,:])])
                # if first_action == None:
                #     samples = tf.concat(concat_dim=1,values=[tf.zeros([self.popsize, self.dU]), samples])
                # else:
                #     samples = tf.concat(concat_dim=1,values=[tf.tile(first_action, [self.popsize, 1]), samples])

                costs = cost_function(samples)
                values, indices = tf.nn.top_k(-costs, k=self.num_elites, sorted=True)

                best_val, best_sol = tf.cond(
                    tf.less(-values[0], best_val),
                    lambda: (-values[0], samples[indices[0]]),
                    lambda: (best_val, best_sol)
                )

                elites = tf.gather(samples, indices)
                # elites = tf.Print(elites, [elites[0,:]])
                elites = elites[:,self.dU*self.delay_step:]
                # elites = tf.Print(elites, [elites[0,:]])
                new_mean = tf.reduce_mean(elites, axis=0)
                new_var = tf.reduce_mean(tf.square(elites - new_mean), axis=0)
                # new_mean = new_mean[self.dU:]
                # new_var = new_var[self.dU:]

                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                var = self.alpha * var + (1 - self.alpha) * new_var

                return t + 1, mean, var, best_val, best_sol

            with self.tf_sess.graph.as_default():
                self.num_opt_iters, self.mean, self.var, self.best_val, self.best_sol = tf.while_loop(
                    cond=continue_optimization, body=iteration,
                    loop_vars=[0, self.init_mean, self.init_var, float("inf"), self.init_mean],
                    shape_invariants=[tf.TensorShape([]), tf.TensorShape([None,]), tf.TensorShape([None,]),tf.TensorShape([]), tf.TensorShape([None])]
                )

    def reset(self):
        pass

    def obtain_solution(self, last_action, init_mean, init_var, per, dU, obs=None):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        if self.tf_compatible:
            sol, solvar = self.tf_sess.run(
                [self.mean, self.var],
                feed_dict={self.init_mean: init_mean, self.init_var: init_var, self.first_action: last_action}
            )
        else:
            assert self._params.il_cfg.use_gt_dynamics
            mean, var, t = init_mean, init_var, 0
            X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

            cfg = {'plan_hor': self._params.opt_cfg.plan_hor,
                   'dU': self._params.env.action_space.shape[0]}
            while (t < self.max_iters) and np.max(var) > self.epsilon:
                lb_dist, ub_dist = mean - self.lb, self.ub - mean
                constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

                samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
                costs = self._gt_compile_cost(
                    obs, samples, cfg, self._dynamics,
                    self._dynamics._numpy_reward_function
                )
                costs = np.reshape(costs, [-1])
                elites = samples[np.argsort(costs)][:self.num_elites]

                new_mean = np.mean(elites, axis=0)
                new_var = np.var(elites, axis=0)

                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                var = self.alpha * var + (1 - self.alpha) * new_var
                logger.info('variance of elite: {}'.format(np.var(elites)))
                logger.info('Mean perforamnce: {}'.format(
                    np.mean(costs[np.argsort(costs)][:self.num_elites]))
                )

                t += 1
            sol, solvar = mean, var
        sol = np.reshape(sol, [-1])

        # prev_sol is going to be used next timestep
        prev_sol = self.update_prev_sol(per, dU, sol)
        return sol, prev_sol
