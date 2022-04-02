from helper import smooth
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
from pathlib import Path

def read_h5_reward_file(loc):
    with h5py.File(loc, "r") as file:
        rewards = np.array(file["rewards"])
        header = dict(file.attrs)

    return rewards, header

def read_all_rewards(dir, fig="rewards", plot_all_paths=True):
    files = Path(dir).rglob(f"*Rewards*.h5")
    files = np.array([path for path in files]).flatten()
    rewards = np.ones((len(files), 5000))
    headers = []
    min_len = 5000

    for i, file in enumerate(files):
        l_rewards, header = read_h5_reward_file(file)
        headers.append(header)
        rewards[i, :len(l_rewards)] = l_rewards
        min_len = np.minimum(min_len, len(l_rewards))
        if plot_all_paths:
            fig = plt.figure(num=fig)
            ax = np.array(fig.axes).flatten()[0]
            ax.plot(smooth(l_rewards, window=21, poly=1),
                    c="gray", alpha=0.5, ls="dashed", lw=0.75)

    rewards = rewards[:, :min_len]

    label = f"example_label (n={len(headers)})"

    return rewards, label

def plot_rewards_batch(rewards, label, window=51, sigma=1, fig="rewards"):
    fig = plt.figure(num=fig)
    ax = np.array(fig.axes).flatten()[0]

    mean_rewards = np.mean(rewards, axis=0)
    std_rewards = np.std(rewards, axis=0)

    smooth_mean = smooth(mean_rewards, window=window, poly=1)
    smooth_std = smooth(std_rewards, window=window, poly=1)

    ax.plot(smooth_mean,
            label=label)
    ax.fill_between(np.arange(len(smooth_mean)),
                    np.clip(smooth_mean + smooth_std * sigma, a_min=0., a_max=None),
                    np.clip(smooth_mean - smooth_std * sigma, a_min=0., a_max=None),
                    alpha=0.1,
                    label=f"{sigma} "r"$\sigma$ CI")

def plot_rewards_comparison(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    rewards, label = read_all_rewards(Path("batch=2022-04-02-13-26-14_a=1e-02_g=8e-01_hlay=(32, 32)"))
    plot_rewards_batch(rewards, label, window=51)

    ax.set_ylim(0., None)
    ax.set_xlim(0., None)

    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'DQN Rewards during Training')
    ax.legend()

    # fig.suptitle('example sup title', fontsize=16)

    plt.show()
    return


def main():
    plot_rewards_comparison()


if __name__ == '__main__':
    main()