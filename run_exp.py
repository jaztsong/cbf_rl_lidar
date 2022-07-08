#!/usr/bin/env python
import subprocess
import time

base_name = "tunnel2"
seed_list = [1235]  # , 
# beta = [0, 0.2]
beta = []
# if_fixed_h = [False, True]
if_fixed_h = [False]


def run_ros_job(cmd):
    gazebo_p = subprocess.Popen(["roslaunch", "racecar_gazebo", "racecar_tunnel.launch"])
    time.sleep(10)
    print("========================================")
    print("Start to run '{}' ".format(" ".join(cmd)))
    subprocess.call(cmd)
    subprocess.call(["rosnode", "kill", "-a"])
    subprocess.call(["pkill", "-9", "roslaunch"])
    subprocess.call(["pkill", "-9", "gzserver"])
    subprocess.call(["pkill", "-9", "gzclient"])
    time.sleep(3)


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

    for beta_ in beta:
        exp_name = base_name + "_"
        if beta_ == 0:
            exp_name += "sac"
        else:
            exp_name += "lagrangian-sac"

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
        run_ros_job(cmd)
