#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint
import sys
import rospy

#TODO don't use direct path
pkg_path = '/home/yliu_local/blimpRL_ws/src/blimpRL/mbrl_pets/script'
sys.path.append(pkg_path)
pkg_path = '/home/yliu_local/blimpRL_ws/src/blimpRL/mbrl_pets/script/pets'
sys.path.append(pkg_path)
pkg_path = '/home/yliu_local/blimpRL_ws/src/blimpRL/mbrl_pets/script/pets/config'
sys.path.append(pkg_path)
pkg_path = '/home/yliu_local/blimpRL_ws/src/blimpRL/mbrl_pets/script/pets/controller'
sys.path.append(pkg_path)
pkg_path = '/home/yliu_local/blimpRL_ws/src/blimpRL/mbrl_pets/script/pets/misc'
sys.path.append(pkg_path)
pkg_path = '/home/yliu_local/blimpRL_ws/src/blimpRL/mbrl_pets/script/pets/modeling'
sys.path.append(pkg_path)

from dotmap import DotMap

from pets import MBRLExperiment
from pets.controller import MPC
from pets.config import create_config

def main(env, ctrl_type, ctrl_args, overrides, logdir):

    rospy.init_node('pets_node', anonymous=False)
    rospy.loginfo("[PETS Node] Initialising...")

    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})

    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    cfg.pprint()

    if ctrl_type == "MPC":
        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)

    exp = MBRLExperiment(cfg.exp_cfg)

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
