from matplotlib import pyplot as plt
from pyrsistent import b
import seaborn as sns
import pandas as pd
import numpy as np
import os, sys

sns.set_theme(style="darkgrid")
ROOT_DIR = "/home/song3/Research/cbf_rl_lidar/data"
OUT_DIR = "/home/song3/Research/cbf_rl_lidar/plot/"
LENGTH = 10000
MAX_REWARD = 190

name_table = {
    "sac": "SAC",
    "lagrangian-sac": "Lagrangian SAC",
    "adaptive-h": "SAC + Online CBF (Ours)",
    "fixed-h": "SAC + Offline CBF",
}

fig_order1 = ["SAC", "Lagrangian SAC", "SAC + Offline CBF", "SAC + Online CBF (Ours)"]
order2 = ["Vanilla Fitted LQR", "Vanilla Deep Koopman",
          "Fitted LQR w/ GMM", "Ours"]
order_hist = [
    "DDPG",
    "SAC",
    "PETS",
    "Vanilla Fitted LQR",
    "Vanilla Deep Koopman",
    "Fitted LQR w/ GMM",
    "Ours",
]


def read_data(label):
    df_list = []
    for folder in os.listdir(ROOT_DIR):
        if not folder.startswith(label):
            continue
        key = folder.split("_")[1]
        for sub_folder in os.listdir("/".join([ROOT_DIR, folder])):
            fname = "/".join([ROOT_DIR, folder, sub_folder, "progress.txt"])
            seed = sub_folder.split("_")[-1]
            if os.path.exists(fname):
                try:
                    df = pd.read_csv(fname, sep="\t", header=0)
                except pd.errors.EmptyDataError:
                    continue
                df["Algorithm"] = name_table[key]
                df["Full_tag"] = folder + "/" + sub_folder
                df["Seed"] = seed
                df_list.append(df)
            else:
                print("ERROR: File {} NOT FOUND!".format(
                    sub_folder + "/progress.txt"))

    return df_list


def plot_timeseries(df_list, column, label, bin_size=200, max_length=4000):
    new_df_list = []
    column_map = {"Reward": "AverageEpRet", "Collision": "Coll"}
    for df in df_list:
        # adjust the case when reaches to max during training state
        # df.loc[(df['key'].str.contains('train')) & (df['y'] == 1000), 'x'] = df[(
        #     df['key'].str.contains('train')) & (df['y'] == 1000)]['x'] - 1000

        df["xtics"] = (df["TotalEnvInteracts"] // bin_size + 1) * bin_size
        df = df.groupby(["xtics"]).max().reset_index()
        # cut out later data after reaches to max
        # end_index = df[(df['y'] >= MAX_REWARD)].index
        # if len(end_index) > 0:
        #     df = df[df.index <= end_index.min()]

        df.index = df["xtics"]
        df.index.name = None

        # max_len = min(LENGTH, df['xtics'].max())
        new_df = pd.DataFrame(
            index=np.linspace(0, max_length, int(max_length / bin_size) + 1),
            columns=[column_map[column], "xtics", "Algorithm", "Seed"]
        )
        new_df.update(df)
        new_df["xtics"] = new_df.index
        new_df.fillna(method="ffill", inplace=True)
        new_df.fillna(method="bfill", inplace=True)
        new_df = new_df.rename(
            columns={column_map[column]: column, "xtics": "Step"})
        new_df.loc[0, :] = np.NAN
        new_df.loc[:, [column]] = new_df[column].rolling(3).mean()
        new_df_list.append(new_df)

    plot_df = pd.concat(new_df_list).reset_index()

    # plot_df["xtics"] = (plot_df["TotalEnvInteracts"] // bin_size + 1) * bin_size
    # plot_df = plot_df.groupby(["xtics"]).max().reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    # plt.rcParams["font.family"] = "Helvetica"
    # ax.set_ylabel(ax.get_ylabel(), fontsize=20)
    t_df = plot_df[plot_df["Algorithm"].isin(fig_order1)]
    g = sns.lineplot(
        x="Step",
        y=column,
        hue="Algorithm",
        style="Algorithm",
        markers=True,
        data=t_df,
        style_order=fig_order1,
        hue_order=fig_order1,
        palette="tab10",
        ci=80,
        ax=ax,
        lw=2,
        markersize=10,
        # markevery=int(max_length / bin_size / 30),
        markeredgewidth=None,
        markeredgecolor=None,
    )
    # g.set(xscale='log')
    # if "pendulum" in fprefix:
    # ax.set_ylim((0, 200))
    # elif "lunar" in fprefix:
    #     ax.set_ylim((-870, 240))

    # ax.set_yticks([200*i for i in range(6)])
    handles, labels = ax.get_legend_handles_labels()
    new_order = []
    for i in range(len(fig_order1)):
        f_indexes = [j for j, elem in enumerate(
            labels) if fig_order1[i] in elem]
        if len(f_indexes) > 0:
            new_order.append(f_indexes[0])

    ax.legend(
        handles=[handles[il] for il in new_order],
        labels=[labels[il] for il in new_order],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.14),
        markerscale=2,
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR + label + "_" + "-".join(column.split(" ")) + "_ALL.pdf")

    for algo in fig_order1:
        fig, ax = plt.subplots(figsize=(7, 5))
        t_df = plot_df[plot_df["Algorithm"] == algo]
        t_df = t_df.sort_values(by=["Seed", "Step"])
        g = sns.lineplot(
            x="Step",
            y=column,
            hue="Seed",
            style="Seed",
            markers=True,
            data=t_df,
            palette="tab10",
            ax=ax,
            lw=2,
            markersize=10,
            markeredgewidth=None,
            markeredgecolor=None,
        )
        ax.legend(
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, -0.14),
            markerscale=2,
        )
        plt.tight_layout()
        plt.savefig(OUT_DIR + label + "_" + "-".join(column.split(" ")) + "_" + algo + ".pdf")


def plot_hist(fprefix, df_list, max_step, max_reward=None):
    new_df_list = []
    for df in df_list:
        if max_reward:
            df = df[(df.key.str.endswith("rew")) & (df.y >= max_reward)]
            df = df.iloc[0:1]
        else:
            df = df[(df.key.str.endswith("test/rew")) & (df.x <= max_step)]
            df = df.iloc[(len(df) - 1):]

        new_df_list.append(df)

    plot_df = pd.concat(new_df_list).reset_index()
    if "bag_size" in fprefix:
        plot_df["exp"] = plot_df["full_tag"].apply(
            lambda x: x.split("_")[-1].replace("bag", "")
        )

    plot_df = plot_df.rename(
        columns={"y": "Reward", "xtics": "Step", "exp": "Algorithm"}
    )

    figsize = (7, 5.2)
    if "bag_size" in fprefix:
        figsize = (7 * 0.7, 5.2 * 0.7)

    fig, ax = plt.subplots(figsize=figsize)
    # if 'lunar' in fprefix:
    #     plot_df['Reward'] = plot_df['Reward'].apply(lambda x: max(x, -300))
    if "bag_size" in fprefix:
        if max_reward:
            g = sns.barplot(
                x="Algorithm",
                y="x",
                data=plot_df,
                ax=ax,
                order=("3", "5", "7", "9", "11", "13"),
                ci=None,
                capsize=0.2,
            )
        else:
            g = sns.barplot(
                x="Algorithm",
                y="Reward",
                data=plot_df,
                ax=ax,
                order=("3", "5", "7", "9", "11", "13"),
                ci=None,
                capsize=0.2,
            )

    else:
        g = sns.barplot(
            y="Algorithm",
            x="Reward",
            data=plot_df,
            ax=ax,
            order=order_hist,
            capsize=0.2,
        )

    if max_reward:
        fprefix += "_step"

    if "lunar" in fprefix:
        ax.set_xlim([-300, 250])
        xlabels = ["{:d}".format(int(x)) for x in g.get_xticks()]
        xlabels[0] = r"$\leqslant$-300"
        g.set_xticklabels(xlabels)
    elif "bag_size" in fprefix:
        ax.set_xlabel("# of GMM clusters")
        if max_reward:
            ax.set_ylabel("# of steps taken to solve the task")

    plt.tight_layout()
    plt.savefig(fprefix + "_hist_algo.pdf")


def plot_cbf_param(label, seed):
    import torch
    sys.path.insert(1, '/home/song3/Research/cbf_rl_lidar')
    # from cbf import CBF

    exp_name = label + "_adaptive-h"
    model = torch.load(ROOT_DIR + "/" +  exp_name + "/" + exp_name + "_s" + str(seed) + "/pyt_save/model.pt")
    fig, ax = plt.subplots(figsize=(4, 4))
    param = model.P.detach().cpu().numpy().squeeze()
    angle_range = 2.3561944902
    angles = [rad * 180.0 / np.pi for rad in np.linspace(-angle_range, angle_range, len(param))]
    hand_craft = np.ones_like(param) * -1.0 / len(param)
    binsize = 2 * angle_range / len(param) * 80
    df_0 = pd.DataFrame({"angle": angles, "P": param, "Algorithm": "Online Learning CBF"})
    df_0["xtics"] = df_0["angle"] // binsize * binsize
    df_1 = pd.DataFrame({"angle": angles, "P": hand_craft, "Algorithm": "Hand-Crafted CBF"})
    df_1["xtics"] = df_1["angle"] // binsize * binsize
    plot_df = pd.concat([df_0, df_1]).reset_index()

    g = sns.lineplot(
        x="xtics",
        y="P",
        hue="Algorithm",
        style="Algorithm",
        # markers=True,
        data=plot_df,
        palette="tab10",
        ax=ax,
        lw=3,
        # markersize=5,
        # markevery=int(1081 / 20),
        markeredgewidth=None,
        markeredgecolor=None,
    )
    ax.set_xlabel(r"Scan Angle ($^o$)")
    ax.set_ylabel(r"Weight of $P$ in CBF")
    ax.set_ylim([-0.023, 0.002])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR + label + "_s" + str(seed) + "_cbf-param.pdf")

def plot_cbf_effect(label, seed):
    exp_name = label + "_adaptive-h"
    filename = ROOT_DIR + "/" +  exp_name + "/" + exp_name + "_s" + str(seed) + "/replaybuffer.npy"
    try:
        data = np.load(filename, allow_pickle=True)
    except FileNotFoundError:
        print("No such file: {}".format(filename))
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax2 = ax.twinx()
    a_rl = data[()]['act_rl'][:4000]
    a = data[()]['act'][:4000]
    binsize = 20
    df_0 = pd.DataFrame({"Step": range(len(a)), "Angle": (a - a_rl)[:, 0] * 180.0 / np.pi})
    df_0["xtics"] = df_0["Step"] // binsize * binsize
    df_1 = pd.DataFrame({"Step": range(len(a)), "Velocity": (a - a_rl)[:, 1]})
    df_1["xtics"] = df_1["Step"] // binsize * binsize

    g = sns.lineplot(
        x="xtics",
        y="Angle",
        data=df_0,
        ax=ax,
        lw=2,
        label="Angle",
        # markevery=int(1081 / 20),
    )
    ax.legend(loc='upper left')
    g = sns.lineplot(
        x="xtics",
        y="Velocity",
        markers=True,
        data=df_1,
        palette="tab10",
        ax=ax2,
        lw=2,
        label="Velocity",
        color='r',
        # markevery=int(1081 / 20),

    )
    ax.set_xlabel(r"Step")
    ax.set_ylabel(r"Steering Angle Adjusted by CBF ($^o$)")
    ax2.set_ylabel(r"Velocity Adjusted by CBF ($m/s$)")
    ax.set_ylim([-100, 100])
    ax2.set_ylim([-4, 4])
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR + label + "_s" + str(seed) + "_cbf-effect.pdf")


if __name__ == "__main__":
    # plot_cbf_effect("tunnel1", 2)
    # plot_cbf_effect("tunnel2", 1235)

    plot_cbf_param("tunnel1", 1237)
    plot_cbf_param("tunnel2", 2)

    # for label in ["tunnel1", "tunnel2"]:
    #     df_list = read_data(label)
    #     plot_timeseries(
    #         df_list, column="Reward", label=label
    #     )
    #     plot_timeseries(
    #         df_list, column="Collision", label=label
    #     )

    # plot_hist("./exp_data/pendulum", df_list, 1000)
    # df_list = read_data("./exp_data/lunarLander")
    # plot_timeseries('./exp_data/lunarLander', df_list, 100, 10000)
    # plot_hist("./exp_data/lunarLander", df_list, 3000)

    # df_list = read_data('./exp_data/bag_size')
    # plot_hist('./exp_data/bag_size', df_list, 1000)
    # plot_hist('./exp_data/bag_size', df_list, 1000, 996)
