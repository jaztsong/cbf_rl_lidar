#!/usr/bin/env python
import subprocess
import time

base_name = "tunnel2"
seed_list = [1234, 1235, 1236, 1237, 1238, 1239]
beta = [0.0, 0.2]
if_fixed_h = [False, True]


def run_ros_job(cmd):
    # gazebo_p = subprocess.Popen(["roslaunch", "racecar_gazebo", "racecar_tunnel.launch"])
    # time.sleep(10)
    print("Start to run '{}' ".format(" ".join(cmd)))
    subprocess.call(cmd)
    # subprocess.call(["pkill", "-9", "gzserver"])
    # subprocess.call(["pkill", "-9", "gzclient"])
    # subprocess.call(["pkill", "-9", "roslaunch"])
    # time.sleep(3)


for seed in seed_list:

    for if_fixed_h_ in if_fixed_h:
        exp_name = base_name + "_"
        if if_fixed_h_:
            exp_name += "fixed-h"
        else:
            exp_name += "adaptive-h"

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
        run_ros_job(cmd)

    # for beta_ in beta:
    #     exp_name = base_name + "_"
    #     if beta_ == 0:
    #         exp_name += "sac"
    #     else:
    #         exp_name += "lagrangian-sac"

    #     cmd = [
    #         "python",
    #         "./sac.py",
    #         "--exp-name",
    #         exp_name,
    #         "--seed",
    #         str(seed),
    #         "--beta",
    #         str(beta_),
    #     ]
    #     run_ros_job(cmd)
