from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set_theme(style="darkgrid")
ROOT_DIR = "/home/song3/Research/cbf_rl_lidar/data"
OUT_DIR = "/home/song3/Research/cbf_rl_lidar/plot/"
LENGTH = 10000
MAX_REWARD = 190

name_table = {
    "sac": "SAC",
    "lagrangian-sac": "Lagrangian SAC",
    "adaptive-h": "CBF SAC (Adaptive)",
    "fixed-h": "CBF SAC (Fixed)",
}

fig_order1 = ["SAC", "Lagrangian SAC", "CBF SAC (Fixed)", "CBF SAC (Adaptive)"]
order2 = ["Vanilla Fitted LQR", "Vanilla Deep Koopman", "Fitted LQR w/ GMM", "Ours"]
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
            if os.path.exists(fname):
                try:
                    df = pd.read_csv(fname, sep="\t", header=0)
                except pd.errors.EmptyDataError:
                    continue
                df["Algorithm"] = name_table[key]
                df["Full_tag"] = folder + "/" + sub_folder
                df_list.append(df)
            else:
                print("ERROR: File {} NOT FOUND!".format(sub_folder + "/progress.txt"))

    # return df_list
    return pd.concat(df_list).reset_index()


def plot_timeseries(plot_df, column, label, bin_size=200, max_length=4000):
    # new_df_list = []
    # for df in df_list:
    #     # adjust the case when reaches to max during training state
    #     # df.loc[(df['key'].str.contains('train')) & (df['y'] == 1000), 'x'] = df[(
    #     #     df['key'].str.contains('train')) & (df['y'] == 1000)]['x'] - 1000

    #     df["xtics"] = (df["TotalEnvInteracts"] // bin_size + 1) * bin_size
    #     df = df.groupby(["xtics"]).max().reset_index()
    #     # cut out later data after reaches to max
    #     # end_index = df[(df['y'] >= MAX_REWARD)].index
    #     # if len(end_index) > 0:
    #     #     df = df[df.index <= end_index.min()]

    #     df.index = df["xtics"]
    #     df.index.name = None

    #     # max_len = min(LENGTH, df['xtics'].max())
    #     new_df = pd.DataFrame(
    #         index=np.linspace(0, length, int(length / bin_size) + 1), columns=df.columns
    #     )
    #     new_df.loc[new_df.index == 0, ["x", "xtics", "y"]] = 0
    #     new_df["xtics"] = new_df.index
    #     new_df.update(df)
    #     new_df.fillna(method="ffill", inplace=True)
    #     new_df.fillna(method="bfill", inplace=True)
    #     new_df_list.append(new_df)

    column_map = {"Reward": "AverageEpRet", "Safety Violation": "Coll"}
    plot_df["xtics"] = (plot_df["TotalEnvInteracts"] // bin_size + 1) * bin_size
    # plot_df = plot_df.groupby(["xtics"]).max().reset_index()
    plot_df = plot_df.loc[:, [column_map[column], "xtics", "Algorithm"]]
    plot_df = plot_df.rename(columns={column_map[column]: column, "xtics": "Step"})

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
        f_indexes = [j for j, elem in enumerate(labels) if fig_order1[i] in elem]
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
    plt.savefig(OUT_DIR + label + "_" + column + "_algos.pdf")


def plot_hist(fprefix, df_list, max_step, max_reward=None):
    new_df_list = []
    for df in df_list:
        if max_reward:
            df = df[(df.key.str.endswith("rew")) & (df.y >= max_reward)]
            df = df.iloc[0:1]
        else:
            df = df[(df.key.str.endswith("test/rew")) & (df.x <= max_step)]
            df = df.iloc[(len(df) - 1) :]

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


if __name__ == "__main__":
    df_list = read_data("tunnel1")
    plot_timeseries(
        df_list, column="Reward", label="tunnel1"
    )
    plot_timeseries(
        df_list, column="Safety Violation", label="tunnel1"
    )
    # plot_hist("./exp_data/pendulum", df_list, 1000)

    # df_list = read_data("./exp_data/lunarLander")
    # plot_timeseries('./exp_data/lunarLander', df_list, 100, 10000)
    # plot_hist("./exp_data/lunarLander", df_list, 3000)

    # df_list = read_data('./exp_data/bag_size')
    # plot_hist('./exp_data/bag_size', df_list, 1000)
    # plot_hist('./exp_data/bag_size', df_list, 1000, 996)
