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

def read_all_rewards(dir, fig="rewards", plot_all_paths=False):
    files = Path(dir).rglob(f"*Rewards*.h5")
    files = np.array([path for path in files]).flatten()
    rewards = np.ones((len(files), 5000))
    headers = []
    min_len = 5000

    fig = plt.figure(num=fig)
    ax = np.array(fig.axes).flatten()[0]

    for i, file in enumerate(files):
        l_rewards, header = read_h5_reward_file(file)
        headers.append(header)
        rewards[i, :len(l_rewards)] = l_rewards
        min_len = np.minimum(min_len, len(l_rewards))
        if plot_all_paths:
            smooth_rewards = smooth(l_rewards, window=21, poly=1)
            # smooth_rewards = smooth(l_rewards, window=1, poly=0)  # for where we want to look at single episodes

            ax.plot(smooth_rewards,
                    c="gray", alpha=0.1, ls="dashed", lw=0.75)
            ax.scatter(len(smooth_rewards) - 1.,
                       smooth_rewards[-1],
                       c="k",
                       s=2.5)

    rewards = rewards[:, :min_len]

    return rewards, headers

def plot_rewards_batch(rewards, header, label, window=21, sigma=1, fig="rewards"):
    fig = plt.figure(num=fig)
    ax = np.array(fig.axes).flatten()[0]

    mean_rewards = np.mean(rewards, axis=0)
    std_rewards = np.std(rewards, axis=0)

    smooth_mean = smooth(mean_rewards, window=window, poly=1)
    smooth_std = smooth(std_rewards, window=window * 3, poly=1)

    max_reward = float(header[0]["max_reward"])


    ax.plot(smooth_mean,
            label=label)
    ax.fill_between(np.arange(len(smooth_mean)),
                    np.clip(smooth_mean + smooth_std * sigma, a_min=0., a_max=200.),
                    np.clip(smooth_mean - smooth_std * sigma, a_min=0., a_max=200.),
                    alpha=0.1)#,label=f"{sigma} "r"$\sigma$ CI")


def plot_rewards_a2c_test(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    rewards, headers = read_all_rewards(Path("runs"))
    label = f"Name (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)


    ax.set_ylim(0., None)
    ax.set_xlim(0., 249)

    ax.axhline(200, ls='--', c='gray', label="200 Reward Limit")
    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'DQN Rewards during Training: Buffer/Batch Size Comparison')
    ax.legend()

    # fig.suptitle('example sup title', fontsize=16)

    plt.show()
    # plt.savefig("bufferbatch.png")
    plt.close()
    return

def main():
    plot_rewards_a2c_test()

if __name__ == '__main__':
    main()
