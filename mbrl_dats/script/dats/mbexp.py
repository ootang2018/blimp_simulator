#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint
import copy
import sys
import rospy

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config
from dmbrl.misc import logger

def main(env, ctrl_type, ctrl_args, overrides, logdir, args):
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    logger.info('\n' + pprint.pformat(cfg))    

    # add the part of popsize
    if ctrl_type == "MPC":
        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)

    cfg.exp_cfg.misc = copy.copy(cfg)
    exp = MBExperiment(cfg.exp_cfg)

    if not os.path.exists(exp.logdir):
        os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

    exp.run_experiment()


if __name__ == "__main__":   
    env = "blimp"
    ctrl_arg = [("opt-type","CEM"),("model-type","PE"),("prop-type","E")]
    override = [("exp_cfg.sim_cfg.delay_hor" ,"5"),("ctrl_cfg.prop_cfg.delay_step", "5")]
    logdir = "/home/yliu_local/blimp_data/DATS-MPC/STAY/3act"


    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, default=env,
                        help='Environment name: select from [blimp, cartpole, reacher, pusher, halfcheetah]')
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=ctrl_arg,
                        help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
    parser.add_argument('-o', '--override', action='append', nargs=2, default=override,
                        help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
    parser.add_argument('-logdir', type=str, default=logdir,
                        help='Directory to which results will be logged (default: ./log)')
    parser.add_argument('-e_popsize', type=int, default=500,
                        help='different popsize to use')

    args = parser.parse_args(sys.argv[4:]) ###

    main(args.env, "MPC", args.ctrl_arg, args.override, args.logdir, args)
