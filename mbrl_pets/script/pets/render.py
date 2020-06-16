#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint
import sys
import rospy

pkg_path = '/home/yliu2/catkin_ws_py3/src/mbrl_hof/script/'
sys.path.append(pkg_path)
pkg_path = '/home/yliu2/catkin_ws_py3/src/mbrl_hof/script/pets/config'
sys.path.append(pkg_path)
pkg_path = '/home/yliu2/catkin_ws_py3/src/mbrl_hof/script/pets/controller'
sys.path.append(pkg_path)
pkg_path = '/home/yliu2/catkin_ws_py3/src/mbrl_hof/script/pets/misc'
sys.path.append(pkg_path)
pkg_path = '/home/yliu2/catkin_ws_py3/src/mbrl_hof/script/pets/modeling'
sys.path.append(pkg_path)
pkg_path = '/home/yliu2/catkin_ws_py3/src/mbrl_hof/script/pets/modeling/layers'
sys.path.append(pkg_path)
pkg_path = '/home/yliu2/catkin_ws_py3/src/mbrl_hof/script/pets/modeling/models'
sys.path.append(pkg_path)
pkg_path = '/home/yliu2/catkin_ws_py3/src/mbrl_hof/script/pets/modeling/utils/'
sys.path.append(pkg_path)

from dotmap import DotMap

from pets import MBRLExperiment
from pets.controller import MPC
from pets.config import create_config

def main(env, ctrl_type, ctrl_args, overrides, logdir):
    rospy.init_node('main_node', anonymous=False)
    rospy.loginfo("Main Node Initialising...")
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})

    ## change this get access to the model
    model_dir = "/home/yliu2/catkin_ws_py3/src/mbrl_hof/script/pets/log/2020-01-18--12:14:55"

    overrides.append(["ctrl_cfg.prop_cfg.model_init_cfg.model_dir", model_dir])
    overrides.append(["ctrl_cfg.prop_cfg.model_init_cfg.load_model", "True"])
    overrides.append(["ctrl_cfg.prop_cfg.model_pretrained", "True"])
    overrides.append(["exp_cfg.exp_cfg.ninit_rollouts", "0"])
    overrides.append(["exp_cfg.exp_cfg.ntrain_iters", "1"])
    overrides.append(["exp_cfg.log_cfg.nrecord", "1"])

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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-env', '--env_arg', action='append', nargs=2, default=[],
    #                     help='Environment name: blimp')
    # parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
    #                     help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
    # parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
    #                     help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
    # parser.add_argument('-logdir', type=str, default='log',
    #                     help='Directory to which results will be logged (default: ./log)')
    # args = parser.parse_args()

    main('blimp', "MPC", [], [], '/home/yliu2/catkin_ws_py3/src/mbrl_hof/script/pets/log')
