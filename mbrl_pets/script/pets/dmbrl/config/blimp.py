from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow.compat.v1 as tf ###
tf.disable_eager_execution() ###

from dotmap import DotMap
# import gym
import sys

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC

class BlimpConfigModule:
    ENV_NAME = "blimp"
    TASK_HORIZON = 60 #60
    NTRAIN_ITERS = 500
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 5 # 5 7 10 13 15
    MODEL_IN, MODEL_OUT = 23, 15 

    def __init__(self):
        from dmbrl.env.blimp import BlimpEnv
        self.ENV = BlimpEnv()
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    """
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
        Note: Only needs to process NumPy arrays.3
    """
    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    """
    .obs_cost_fn (func): A function which computes the cost of every observation
        in a 2D matrix.
        Note: Must be able to process both NumPy and Tensorflow arrays.
    .ac_cost_fn (func): A function which computes the cost of every action
        in a 2D matrix.
    """
    @staticmethod
    def obs_cost_fn(obs):
        w_dist = 0.8
        w_ang = 0.05
        w_dir = 0.05

        '''
        state
        0:2 relative_angle
        3:5 angular velocity
        6:8 relative_position
        9:11 velocity
        12:14 acceleration
        '''

        # define distance cost
        dist_mse_cost = tf.sqrt(tf.reduce_sum(tf.square(obs[:, 6:9]), axis=1)) # mse distance

        # define angle cost (phi, the)
        ang_mse_cost = tf.sqrt(tf.reduce_sum(tf.square(obs[:, 0:2]), axis=1)) # mse angle

        # define direction cost (psi)
        dir_abs_cost = tf.abs(obs[:, 2]) # psi angle

        return w_dist*dist_mse_cost + w_ang*ang_mse_cost + w_dir*dir_abs_cost ###

    @staticmethod
    def ac_cost_fn(acs):
        w_act = 0.1

        # define action cost
        act_mse_cost = tf.reduce_sum(tf.square(acs), axis=1)

        return w_act*act_mse_cost

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(75, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.000025))
            model.add(FC(75, activation="swish", weight_decay=0.00005))
            model.add(FC(75, activation="swish", weight_decay=0.000075))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0001))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model

CONFIG_MODULE = BlimpConfigModule
