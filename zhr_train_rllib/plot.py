import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from user_config import DEFAULT_IMG_DIR, DEFAULT_DATA_DIR
import argparse
import glob

def smooth(data, sm=1, value="Averagetest_reward"):
    if sm > 1:
        smooth_data = []
        for d in data:
            x = np.asarray(d[value])
            y = np.ones(sm)*1.0/sm
            d[value] = np.convolve(y, x, "same")

            smooth_data.append(d)

        return smooth_data
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_name', default=None)
    parser.add_argument('--seed', default='10', type=int)
    parser.add_argument('--smooth', default='2', type=int)
    parser.add_argument('--output_name', default=None, type=str)
    parser.add_argument('--x', default="timesteps_total", type=str)
    parser.add_argument('--y', default="custom_metrics/distance_travelled_mean", type=str)
    args = parser.parse_args()

    from user_config import DEFAULT_DATA_DIR, DEFAULT_IMG_DIR
    dir_name = "train_multi_scenario_fc3_its"
    file_dir = os.path.join(DEFAULT_DATA_DIR, dir_name)
    dataset = []
    for root, dirs, name in os.walk(file_dir):
        if "progress.csv" in name:
            data_file = os.path.join(root, "progress.csv")
            data = pd.read_table(data_file, sep=",")
            seed = root.split("/")[-1].split("=")[1][0:2]
            data["seed"] = seed

            # x = data["seed"]
            # x = data[args.x]

            # data[args.x] = int(data[args.x])
            # data[args.y] = float(data[args.y])
            dataset.append(data)

    data = pd.concat(dataset, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=data, x=args.x, y=args.y)
    sns.lineplot(data=data, x=args.x, y="custom_metrics/distance_travelled_max")

    output_name = "crash_del200_" + "distance_travelled"
    out_file = os.path.join(DEFAULT_IMG_DIR, output_name + ".png")
    plt.legend(loc='best').set_draggable(True)
    plt.tight_layout(pad=0.5)
    plt.savefig(out_file)

    # tasks = ["hopper", "halfcheetah", "walker2d"]
    # for task in tasks:
    #     plt.cla()
    #     data = []
    #     for kl in [0.03, 0.05, 0.07]:
    #         for seed in [20, 30, 40]:
    #             taskname = "ppo_kl_" + task + "_clipv_maxgrad_anneallr3e-4_normal_maxkl" + str(kl) \
    #                 + "_gae_norm-state-return_steps2048_batch64_notdone_lastv_4_entropy_update10"
    #             file_dir = os.path.join(DEFAULT_DATA_DIR, taskname)
    #             file_seed = os.path.join(file_dir, taskname+"_s" + str(seed), "progress.txt")
    #             pd_data = pd.read_table(file_seed)
    #             pd_data["KL"] = "max_kl" + str(kl)

    #             data.append(pd_data)

    #     smooth(data, sm=args.smooth)
    #     data = pd.concat(data, ignore_index=True)

    #     sns.set(style="darkgrid", font_scale=1.5)
    #     sns.lineplot(data=data, x=args.x, y=args.y, hue="KL")

    #     output_name = "ppo_" + task + "_smooth"
    #     out_file = os.path.join(DEFAULT_IMG_DIR, output_name + ".png")
    #     plt.legend(loc='best').set_draggable(True)
    #     plt.tight_layout(pad=0.5)
    #     plt.savefig(out_file)