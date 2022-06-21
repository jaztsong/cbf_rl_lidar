#!/usr/bin/env python
import subprocess
import multiprocessing
import itertools

env_name = "tunnel1"
seed_list = [i for i in range(5)]
beta = [0, 0.1]
if_fixed_h = [False, True]


for beta_ in beta:
    exp_name = env_name + "_"
    if beta_ == 0:
        exp_name += "sac"
    else:
        exp_name += "lagrangian-sac"

    for seed in seed_list:
        cmd = [
            "python",
            "./sac.py",
            "--exp-name",
            exp_name,
            "--seed",
            str(seed),
            "--beta",
            str(beta_),
        ]
        print("Start to run {} with seed {} with command".format(exp_name, seed))
        print(" ".join(cmd))
        subprocess.call(cmd)

for if_fixed_h_ in if_fixed_h:
    exp_name = env_name + "_"
    if if_fixed_h_:
        exp_name += "fixed-h"
    else:
        exp_name += "adaptive-h"

    for seed in seed_list:
        print("Start to run {} with seed {}".format(exp_name, seed))
        cmd = [
            "python",
            "./sac.py",
            "--exp-name",
            exp_name,
            "--seed",
            str(seed),
            "--beta",
            "0.0",
            "--if-cbf",
        ]
        if if_fixed_h_:
            cmd += ["--if-fixed-h"]
        print("Start to run {} with seed {} with command".format(exp_name, seed))
        print(" ".join(cmd))
        subprocess.call(cmd)
