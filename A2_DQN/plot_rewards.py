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

def read_all_rewards(dir):
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

    rewards = rewards[:, :min_len]

    label="example_label"

    return rewards, label

def plot_rewards_batch(rewards, label, window=51, fig="rewards"):
    fig = plt.figure(num=fig)
    ax = np.array(fig.axes).flatten()[0]

    mean_rewards = np.mean(rewards, axis=0)
    std_rewards = np.std(rewards, axis=0)

    smooth_mean = smooth(mean_rewards, window=window, poly=1)
    smooth_std = smooth(std_rewards, window=window, poly=1)

    ax.plot(smooth_mean,
            label=label)
    ax.fill_between(np.arange(len(smooth_mean)),
                    smooth_mean + smooth_std,
                    smooth_mean - smooth_std,
                    alpha=0.1)

def plot_rewards_comparison(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    rewards, label = read_all_rewards(Path("batch=2022-03-31-11-20-29-alpha1e-02-gamma1e+00-12, 6"))
    plot_rewards_batch(rewards, label, window=11)

    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'Rewards')
    ax.legend()

    # fig.suptitle('example sup title', fontsize=16)

    plt.show()
    return


def main():
    plot_rewards_comparison()


if __name__ == '__main__':
    main()