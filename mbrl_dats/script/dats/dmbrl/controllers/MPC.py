from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import tensorflow.compat.v1 as tf
import numpy as np
from scipy.io import savemat

from .Controller import Controller
from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.optimizers import RandomOptimizer, CEMOptimizer
from dmbrl.misc.optimizers import GBPRandomOptimizer, GBPCEMOptimizer
from dmbrl.misc.optimizers import POPLINPOptimizer
from dmbrl.misc.optimizers import POPLINAOptimizer
from dmbrl.misc import logger


class MPC(Controller):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer,
                  'GBPRandom': GBPRandomOptimizer, 'GBPCEM': GBPCEMOptimizer,
                  'POPLIN-A': POPLINAOptimizer, 'POPLIN-P': POPLINPOptimizer}

    def __init__(self, params):
        """Creates class instance.

        Arguments:
            params
                .env (gym.env): Environment for which this controller will be used.
                .update_fns (list<func>): A list of functions that will be invoked
                    (possibly with a tensorflow session) every time this controller is reset.
                .ac_ub (np.ndarray): (optional) An array of action upper bounds.
                    Defaults to environment action upper bounds.
                .ac_lb (np.ndarray): (optional) An array of action lower bounds.
                    Defaults to environment action lower bounds.
                .per (int): (optional) Determines how often the action sequence will be optimized.
                    Defaults to 1 (reoptimizes at every call to act()).
                .prop_cfg
                    .model_init_cfg (DotMap): A DotMap of initialization parameters for the model.
                        .model_constructor (func): A function which constructs an instance of this
                            model, given model_init_cfg.
                    .model_train_cfg (dict): (optional) A DotMap of training parameters that will be passed
                        into the model every time is is trained. Defaults to an empty dict.
                    .model_pretrained (bool): (optional) If True, assumes that the model
                        has been trained upon construction.
                    .mode (str): Propagation method. Choose between [E, DS, TSinf, TS1, MM].
                        See https://arxiv.org/abs/1805.12114 for details.
                    .npart (int): Number of particles used for DS, TSinf, TS1, and MM propagation methods.
                    .ign_var (bool): (optional) Determines whether or not variance output of the model
                        will be ignored. Defaults to False unless deterministic propagation is being used.
                    .obs_preproc (func): (optional) A function which modifies observations (in a 2D matrix)
                        before they are passed into the model. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc (func): (optional) A function which returns vectors calculated from
                        the previous observations and model predictions, which will then be passed into
                        the provided cost function on observations. Defaults to lambda obs, model_out: model_out.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc2 (func): (optional) A function which takes the vectors returned by
                        obs_postproc and (possibly) modifies it into the predicted observations for the
                        next time step. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .targ_proc (func): (optional) A function which takes current observations and next
                        observations and returns the array of targets (so that the model learns the mapping
                        obs -> targ_proc(obs, next_obs)). Defaults to lambda obs, next_obs: next_obs.
                        Note: Only needs to process NumPy arrays.
                .opt_cfg
                    .mode (str): Internal optimizer that will be used. Choose between [CEM, Random].
                    .cfg (DotMap): A map of optimizer initializer parameters.
                    .plan_hor (int): The planning horizon that will be used in optimization.
                    .obs_cost_fn (func): A function which computes the cost of every observation
                        in a 2D matrix.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .ac_cost_fn (func): A function which computes the cost of every action
                        in a 2D matrix.
                .log_cfg
                    .save_all_models (bool): (optional) If True, saves models at every iteration.
                        Defaults to False (only most recent model is saved).
                        Warning: Can be very memory-intensive.
                    .log_traj_preds (bool): (optional) If True, saves the mean and variance of predicted
                        particle trajectories. Defaults to False.
                    .log_particles (bool) (optional) If True, saves all predicted particles trajectories.
                        Defaults to False. Note: Takes precedence over log_traj_preds.
                        Warning: Can be very memory-intensive
        """
        super().__init__(params)
        self._params = params
        self.delay_step = params.prop_cfg.get("delay_step", 0)
        self.dO, self.dU = params.env.observation_space.shape[0], params.env.action_space.shape[0]
        self.ac_ub, self.ac_lb = params.env.action_space.high, params.env.action_space.low
        # assert np.max(self.ac_lb) == np.min(self.ac_lb)  # just to make sure ###
        self.ac_ub = np.minimum(self.ac_ub, params.get("ac_ub", self.ac_ub))
        self.ac_lb = np.maximum(self.ac_lb, params.get("ac_lb", self.ac_lb))
        self.update_fns = params.get("update_fns", [])
        self.per = params.get("per", 1)

        # self.model = get_required_argument(
        #     params.prop_cfg.model_init_cfg, "model_constructor", "Must provide a model constructor."
        # )(params.prop_cfg.model_init_cfg, misc=params)
        self.model = get_required_argument(
            params.prop_cfg.model_init_cfg, "model_constructor", "Must provide a model constructor."
        )(params.prop_cfg.model_init_cfg)
        self.model_train_cfg = params.prop_cfg.get("model_train_cfg", {})
        self.prop_mode = get_required_argument(params.prop_cfg, "mode", "Must provide propagation method.")
        self.ign_var = params.prop_cfg.get("ign_var", False) or self.prop_mode == "E"

        self.obs_preproc = params.prop_cfg.get("obs_preproc", lambda obs: obs)
        self.obs_postproc = params.prop_cfg.get("obs_postproc", lambda obs, model_out: model_out)
        self.obs_postproc2 = params.prop_cfg.get("obs_postproc2", lambda next_obs: next_obs)
        self.targ_proc = params.prop_cfg.get("targ_proc", lambda obs, next_obs: next_obs)
        self.npart = get_required_argument(params.prop_cfg, "npart", "Must provide number of particles.")

        self.opt_mode = get_required_argument(params.opt_cfg, "mode", "Must provide optimization method.")
        self.plan_hor = get_required_argument(params.opt_cfg, "plan_hor", "Must provide planning horizon.")
        self.obs_cost_fn = get_required_argument(params.opt_cfg, "obs_cost_fn", "Must provide cost on observations.")
        self.ac_cost_fn = get_required_argument(params.opt_cfg, "ac_cost_fn", "Must provide cost on actions.")
        self.obs_ac_cost_fn = params.prop_cfg.get("obs_ac_cost_fn", None)

        self.save_all_models = params.log_cfg.get("save_all_models", False)
        self.log_traj_preds = params.log_cfg.get("log_traj_preds", False)
        self.log_particles = params.log_cfg.get("log_particles", False)

        self.last_action = np.zeros([1, self.delay_step*self.dU])
        # Perform argument checks
        if self.prop_mode not in ["E", "DS", "MM", "TS1", "TSinf", "GT"]:
            raise ValueError("Invalid propagation method.")
        if self.prop_mode in ["TS1", "TSinf"] and self.npart % self.model.num_nets != 0:
            raise ValueError("Number of particles must be a multiple of the ensemble size.")
        if self.prop_mode == "E" and self.npart != 1:
            raise ValueError("Deterministic propagation methods only need one particle.")

        # Create action sequence optimizer
        opt_cfg = params.opt_cfg.get("cfg", {})
        self.optimizer = MPC.optimizers[params.opt_cfg.mode](
            sol_dim=(self.plan_hor-self.delay_step) * self.dU,
            delay_step = self.delay_step,
            lower_bound=np.tile(self.ac_lb, [self.plan_hor-self.delay_step]),
            upper_bound=np.tile(self.ac_ub, [self.plan_hor-self.delay_step]),
            tf_session=None if not self.model.is_tf_model else self.model.sess,
            params=params,
            dU=self.dU,
            **opt_cfg
        )
        self._policy_network = self.optimizer.get_policy_network()

        # Controller state variables
        self.has_been_trained = params.prop_cfg.get("model_pretrained", False)
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor-self.delay_step])
        self.init_var = np.tile(params.opt_cfg.init_var,
                                [self.ac_lb.shape[0] * (self.plan_hor - self.delay_step)])
        self.train_in = np.array([]).reshape(
            0, self.dU + self.obs_preproc(np.zeros([1, self.dO])).shape[-1]
        )
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dO]),
                              np.zeros([1, self.dO])).shape[-1]
        )
        if self.model.is_tf_model:
            assert not self._params.il_cfg.use_gt_dynamics
            self.sy_cur_obs = tf.Variable(np.zeros(self.dO), dtype=tf.float32)
            self.ac_seq = tf.placeholder(shape=[1, (self.plan_hor) * self.dU], dtype=tf.float32)
            self.pred_cost, self.pred_traj = self._compile_cost(self.ac_seq, get_pred_trajs=True)

            self.optimizer.set_sy_cur_obs(self.sy_cur_obs)
            # IT IS A HACK. only run when using POPLINA-INIT
            self.optimizer.forward_policy_propose(self._predict_next_obs,
                                                  self.sy_cur_obs)
            self.optimizer.setup(self._compile_cost, True)
            self.model.sess.run(tf.variables_initializer([self.sy_cur_obs]))
            self.prev_sol = self.optimizer.reset_prev_sol(self.prev_sol)  # hack
        else:
            assert self._params.il_cfg.use_gt_dynamics

        logger.info("Created an MPC controller, prop mode %s, %d particles. "
                    % (self.prop_mode, self.npart) + ("Ignoring variance." if self.ign_var else ""))

        if self.save_all_models:
            logger.info("Controller will save all models. (Note: This may be memory-intensive.")
        if self.log_particles:
            logger.info("Controller is logging particle predictions (Note: This may be memory-intensive).")
            self.pred_particles = []
        elif self.log_traj_preds:
            logger.info("Controller is logging trajectory prediction statistics (mean+var).")
            self.pred_means, self.pred_vars = [], []
        else:
            logger.info("Trajectory prediction logging is disabled.")

    def train(self, obs_trajs, acs_trajs, rews_trajs):
        """Trains the internal model of this controller. Once trained,
        this controller switches from applying random actions to using MPC.

        Arguments:
            obs_trajs: A list of observation matrices, observations in rows.
            acs_trajs: A list of action matrices, actions in rows.
            rews_trajs: A list of reward arrays.

        Returns: None.
        """
        if self.model.is_tf_model:
            # Construct new training points and add to training set
            new_train_in, new_train_targs = [], []
            for obs, acs in zip(obs_trajs, acs_trajs):
                new_train_in.append(np.concatenate([self.obs_preproc(obs[:-1]), acs], axis=-1))
                new_train_targs.append(self.targ_proc(obs[:-1], obs[1:]))
            self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
            self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

            # Train the model
            self.model_train_cfg['misc'] = self._params.mb_cfg
            self.model.train(self.train_in, self.train_targs, **self.model_train_cfg)

        if self.optimizer.train_policy_network() and self.has_been_trained:
            # train the policy network here
            self.optimizer.train(obs_trajs, acs_trajs, rews_trajs)

        self.has_been_trained = True

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor - self.delay_step])
        self.prev_sol = self.optimizer.reset_prev_sol(self.prev_sol)  # HACK
        self.optimizer.reset()
        if self.model.is_tf_model:
            for update_fn in self.update_fns:
                update_fn(self.model.sess)

    def act(self, obs, t, get_pred_cost=False, test_policy=False, average=False):
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        """
        if not self.has_been_trained:
            return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            # self.last_action = tf.Print(self.last_action, [self.last_action, action])
            if self.delay_step == 0:
                self.last_action = np.zeros([1,0])
            else:
                self.last_action = np.concatenate((self.last_action[:, self.dU:], action*np.ones([1,self.dU])), axis=1)
            return action

        if test_policy:
            self.sy_cur_obs.load(obs, self.model.sess)
            soln, self.prev_sol = self.optimizer.obtain_test_solution(
                self.prev_sol, self.init_var, self.per, self.dU, obs=obs, average=average
            )

        elif self.model.is_tf_model:
            
            self.sy_cur_obs.load(obs, self.model.sess)
            soln, self.prev_sol = self.optimizer.obtain_solution(
                self.last_action, self.prev_sol, self.init_var, self.per, self.dU
            )
        else:
            soln, self.prev_sol = self.optimizer.obtain_solution(
                self.last_action, self.prev_sol, self.init_var, self.per, self.dU, obs
            )

        # TODO: POPLIN-A prev_sol -- > noise. It is not very good
        assert self.per == 1  # the pwcem does not support per != 1
        # self.ac_buf = soln[:self.per * self.dU].reshape(-1, self.dU)
        self.ac_buf = soln[:self.per * self.dU].reshape(-1, self.dU)

        if get_pred_cost and not (self.log_traj_preds or self.log_particles):
            if self.model.is_tf_model:
                pred_cost = self.model.sess.run(
                    self.pred_cost,
                    feed_dict={self.ac_seq: soln[None]}
                )[0]
            else:
                raise NotImplementedError()
            return self.act(obs, t), pred_cost
        elif self.log_traj_preds or self.log_particles:
            pred_cost, pred_traj = self.model.sess.run(
                [self.pred_cost, self.pred_traj],
                feed_dict={self.ac_seq: soln[None]}
            )
            pred_cost, pred_traj = pred_cost[0], pred_traj[:, 0]
            if self.log_particles:
                self.pred_particles.append(pred_traj)
            else:
                self.pred_means.append(np.mean(pred_traj, axis=1))
                self.pred_vars.append(np.mean(np.square(pred_traj - self.pred_means[-1]), axis=1))
            if get_pred_cost:
                return self.act(obs, t), pred_cost
        return self.act(obs, t)

    def dump_logs(self, primary_logdir, iter_logdir):
        """Saves logs to either a primary log directory or another iteration-specific directory.
        See __init__ documentation to see what is being logged.

        Arguments:
            primary_logdir (str): A directory path. This controller assumes that this directory
                does not change every iteration.
            iter_logdir (str): A directory path. This controller assumes that this directory
                changes every time dump_logs is called.

        Returns: None
        """
        self.model.save(iter_logdir if self.save_all_models else primary_logdir)
        if self.log_particles:
            savemat(os.path.join(iter_logdir, "predictions.mat"), {"predictions": self.pred_particles})
            self.pred_particles = []
        elif self.log_traj_preds:
            savemat(
                os.path.join(iter_logdir, "predictions.mat"),
                {"means": self.pred_means, "vars": self.pred_vars}
            )
            self.pred_means, self.pred_vars = [], []

    def _compile_POPLINP_cost(self, weight_input, cem_type, tf_data_dict):
        """ @brief:
            The input is the noise of the weight space

            @weight_input: size [pop_size, plan_hor, weight_size]
        """

        policy_network = tf_data_dict['policy_network']
        # nopt is different solutions
        t, nopt = tf.constant(0), tf.shape(weight_input)[0]
        init_costs = tf.zeros([nopt, self.npart])
        init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.npart, 1])

        weight_input = tf.reshape(tf.tile(
            tf.transpose(weight_input, [1, 0, 2])[:, :, None],
            [1, 1, self.npart, 1]  # hor, popsize, npart, dU
        ), [self.plan_hor, -1, tf_data_dict['weight_size']])

        def limit_action(action):
            return tf.minimum(tf.maximum(action, self.ac_lb[0]), self.ac_ub[0])

        if cem_type in ['POPLINP-SEP', 'POPLINP-UNI']:

            # step 2: cem on top of the @proposed_act_seqs
            def iteration(t, total_cost, cur_obs):
                cur_acs = \
                    policy_network.forward_network(cur_obs, weight_input[t])
                cur_acs = limit_action(cur_acs)
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                if self.obs_ac_cost_fn is not None:
                    delta_cost = tf.reshape(
                        self.obs_ac_cost_fn(next_obs, cur_acs),
                        [-1, self.npart]
                    )
                else:
                    delta_cost = tf.reshape(
                        self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs),
                        [-1, self.npart]
                    )
                return t + 1, total_cost + delta_cost, \
                    self.obs_postproc2(next_obs), cur_acs
            pass
        else:
            raise NotImplementedError

        total_cost, cur_obs = init_costs, init_obs
        for t in range(self.plan_hor):
            _, total_cost, cur_obs, cur_acs = iteration(t, total_cost, cur_obs)

        costs = total_cost

        # replace nan costs with very high cost
        return tf.reduce_mean(tf.where(tf.is_nan(costs),
                              1e6 * tf.ones_like(costs), costs), axis=1)

    def _compile_POPLINA_cost(self, ac_noise_seqs, cem_type, tf_data_dict):
        """ @brief:
            'POPLINA-INIT': first get the mean actions
            'POPLINA-REPLAN': replan with the actions
        """
        policy_network = tf_data_dict['policy_network']
        # nopt is different solutions
        t, nopt = tf.constant(0), tf.shape(ac_noise_seqs)[0]
        init_costs = tf.zeros([nopt, self.npart])
        init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.npart, 1])
        ac_noise_seqs = tf.reshape(ac_noise_seqs,  # [popsize, sol_dim]
                                   [-1, self.plan_hor, self.dU])
        ac_noise_seqs = tf.reshape(tf.tile(
            tf.transpose(ac_noise_seqs, [1, 0, 2])[:, :, None],
            [1, 1, self.npart, 1]  # hor, popsize, npart, dU
        ), [self.plan_hor, -1, self.dU])  # [hor, popsize * npart, dU]

        def limit_action(action):
            return tf.minimum(tf.maximum(action, self.ac_lb[0]), self.ac_ub[0])

        if cem_type == 'POPLINA-INIT':
            proposed_act_seqs = tf_data_dict['proposed_act_seqs']

            # step 2: cem on top of the @proposed_act_seqs
            def iteration(t, total_cost, cur_obs):
                cur_acs = ac_noise_seqs[t] + proposed_act_seqs[t]
                cur_acs = limit_action(cur_acs)
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                if self.obs_ac_cost_fn is not None:
                    delta_cost = tf.reshape(
                        self.obs_ac_cost_fn(next_obs, cur_acs),
                        [-1, self.npart]
                    )
                else:
                    delta_cost = tf.reshape(
                        self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs),
                        [-1, self.npart]
                    )
                return t + 1, total_cost + delta_cost, \
                    self.obs_postproc2(next_obs), cur_acs

        elif cem_type == 'POPLINA-REPLAN':

            # step 2: cem on top of the @proposed_act_seqs
            def iteration(t, total_cost, cur_obs):
                cur_acs = \
                    policy_network.forward_network(cur_obs) + ac_noise_seqs[t]
                cur_acs = limit_action(cur_acs)
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                if self.obs_ac_cost_fn is not None:
                    delta_cost = tf.reshape(
                        self.obs_ac_cost_fn(next_obs, cur_acs),
                        [-1, self.npart]
                    )
                else:
                    delta_cost = tf.reshape(
                        self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs),
                        [-1, self.npart]
                    )
                return t + 1, total_cost + delta_cost, \
                    self.obs_postproc2(next_obs), cur_acs
            pass
        else:
            raise NotImplementedError

        total_cost, cur_obs = init_costs, init_obs
        for t in range(self.plan_hor):
            _, total_cost, cur_obs, cur_acs = iteration(t, total_cost, cur_obs)

        costs = total_cost

        # replace nan costs with very high cost
        return tf.reduce_mean(tf.where(tf.is_nan(costs),
                              1e6 * tf.ones_like(costs), costs), axis=1)

    def _compile_cost(self, ac_seqs, get_pred_trajs=False,
                      cem_type=None, tf_data_dict=None):

        if cem_type in ['POPLINA-INIT', 'POPLINA-REPLAN']:
            return self._compile_POPLINA_cost(ac_seqs, cem_type, tf_data_dict)
        elif cem_type in ['POPLINP-UNI', 'POPLINP-SEP']:
            return self._compile_POPLINP_cost(ac_seqs, cem_type, tf_data_dict)
        # default value self.plan --> 30, self.dU --> 6 (cheetah)
        t, nopt = tf.constant(0), tf.shape(ac_seqs)[0]
        init_costs = tf.zeros([nopt, self.npart])
        ac_seqs = tf.reshape(ac_seqs, [-1, self.plan_hor, self.dU])
        ac_seqs = tf.reshape(tf.tile(
            tf.transpose(ac_seqs, [1, 0, 2])[:, :, None],
            [1, 1, self.npart, 1]
        ), [self.plan_hor, -1, self.dU])
        init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.npart, 1])

        def continue_prediction(t, *args):
            return tf.less(t, self.plan_hor)

        if get_pred_trajs:
            pred_trajs = init_obs[None]

            def iteration(t, total_cost, cur_obs, pred_trajs):
                cur_acs = ac_seqs[t]
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                if self.obs_ac_cost_fn is not None:
                    delta_cost = tf.reshape(
                        self.obs_ac_cost_fn(next_obs, cur_acs),
                        [-1, self.npart]
                    )
                else:
                    delta_cost = tf.reshape(
                        self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs),
                        [-1, self.npart]
                    )
                next_obs = self.obs_postproc2(next_obs)
                pred_trajs = tf.concat([pred_trajs, next_obs[None]], axis=0)
                return t + 1, total_cost + delta_cost, next_obs, pred_trajs

            _, costs, _, pred_trajs = tf.while_loop(
                cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs, pred_trajs],
                shape_invariants=[
                    t.get_shape(), init_costs.get_shape(), init_obs.get_shape(), tf.TensorShape([None, None, self.dO])
                ]
            )

            # Replace nan costs with very high cost
            costs = tf.reduce_mean(tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs), axis=1)
            pred_trajs = tf.reshape(pred_trajs, [self.plan_hor + 1, -1, self.npart, self.dO])
            return costs, pred_trajs
        else:
            def iteration(t, total_cost, cur_obs):
                cur_acs = ac_seqs[t]
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                if self.obs_ac_cost_fn is not None:
                    delta_cost = tf.reshape(
                        self.obs_ac_cost_fn(next_obs, cur_acs),
                        [-1, self.npart]
                    )
                else:
                    delta_cost = tf.reshape(
                        self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
                    )
                return t + 1, total_cost + delta_cost, self.obs_postproc2(next_obs)

            _, costs, _ = tf.while_loop(
                cond=continue_prediction, body=iteration, loop_vars=[t, init_costs, init_obs]
            )

            # Replace nan costs with very high cost
            return tf.reduce_mean(tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs), axis=1)

    def _predict_next_obs(self, obs, acs):
        proc_obs = self.obs_preproc(obs)

        if self.model.is_tf_model:
            # TS Optimization: Expand so that particles are only passed through one of the networks.
            if self.prop_mode == "TS1":
                proc_obs = tf.reshape(proc_obs, [-1, self.npart, proc_obs.get_shape()[-1]])
                sort_idxs = tf.nn.top_k(
                    tf.random_uniform([tf.shape(proc_obs)[0], self.npart]),
                    k=self.npart
                ).indices
                tmp = tf.tile(tf.range(tf.shape(proc_obs)[0])[:, None], [1, self.npart])[:, :, None]
                idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
                proc_obs = tf.gather_nd(proc_obs, idxs)
                proc_obs = tf.reshape(proc_obs, [-1, proc_obs.get_shape()[-1]])
            if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
                proc_obs, acs = self._expand_to_ts_format(proc_obs), self._expand_to_ts_format(acs)

            # Obtain model predictions
            inputs = tf.concat([proc_obs, acs], axis=-1)
            mean, var = self.model.create_prediction_tensors(inputs)
            if self.model.is_probabilistic and not self.ign_var:
                predictions = mean + tf.random_normal(shape=tf.shape(mean), mean=0, stddev=1) * tf.sqrt(var)
                if self.prop_mode == "MM":
                    model_out_dim = predictions.get_shape()[-1].value

                    predictions = tf.reshape(predictions, [-1, self.npart, model_out_dim])
                    prediction_mean = tf.reduce_mean(predictions, axis=1, keep_dims=True)
                    prediction_var = tf.reduce_mean(tf.square(predictions - prediction_mean), axis=1, keep_dims=True)
                    z = tf.random_normal(shape=tf.shape(predictions), mean=0, stddev=1)
                    samples = prediction_mean + z * tf.sqrt(prediction_var)
                    predictions = tf.reshape(samples, [-1, model_out_dim])
            else:
                predictions = mean

            # TS Optimization: Remove additional dimension
            if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
                predictions = self._flatten_to_matrix(predictions)
            if self.prop_mode == "TS1":
                predictions = tf.reshape(predictions, [-1, self.npart, predictions.get_shape()[-1]])
                sort_idxs = tf.nn.top_k(
                    -sort_idxs,
                    k=self.npart
                ).indices
                idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
                predictions = tf.gather_nd(predictions, idxs)
                predictions = tf.reshape(predictions, [-1, predictions.get_shape()[-1]])

            return self.obs_postproc(obs, predictions)
        else:
            raise NotImplementedError()

    def _expand_to_ts_format(self, mat):
        dim = mat.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(mat, [-1, self.model.num_nets, self.npart // self.model.num_nets, dim]),
                [1, 0, 2, 3]
            ),
            [self.model.num_nets, -1, dim]
        )

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(ts_fmt_arr, [self.model.num_nets, -1, self.npart // self.model.num_nets, dim]),
                [1, 0, 2, 3]
            ),
            [-1, dim]
        )
