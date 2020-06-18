#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint
import sys
import rospy

from dotmap import DotMap

from dmbrl.misc.MBExp import MBRLExperiment
from dmbrl.controller.MPC import MPC
from dmbrl.config import create_config

def main(env, ctrl_type, ctrl_args, overrides, logdir):

    rospy.init_node('pets_node', anonymous=False)
    rospy.loginfo("[PETS Node] Initialising...")

    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    cfg.pprint()

    if ctrl_type == "MPC":
        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)

    exp = MBRLExperiment(cfg.exp_cfg)

    if not os.path.exists(exp.logdir):
        os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))
    exp.run_experiment()


if __name__ == "__main__":

    env = 'blimp'
    ctrl_type = "MPC"
    ctrl_args = []
    overrides = []
    logdir = '/home/yliu_local/blimpRL_ws/src/RL_log/pets_log'

    main(env, ctrl_type, ctrl_args, overrides, logdir)
