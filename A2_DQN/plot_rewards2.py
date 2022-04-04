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

    for i, file in enumerate(files):
        l_rewards, header = read_h5_reward_file(file)
        headers.append(header)
        rewards[i, :len(l_rewards)] = l_rewards
        min_len = np.minimum(min_len, len(l_rewards))
        if plot_all_paths:
            fig = plt.figure(num=fig)
            ax = np.array(fig.axes).flatten()[0]
            smooth_rewards = l_rewards # smooth(l_rewards, window=21, poly=1)
            ax.plot(smooth_rewards,
                    c="gray", alpha=0.5, ls="dashed", lw=0.75)
            ax.scatter(len(smooth_rewards) - 1.,
                       smooth_rewards[-1],
                       c="k",
                       s=2.5)

    rewards = rewards[:, :min_len]

    #label = f"example_label (n={len(headers)})"

    return rewards, len(headers)

def plot_rewards_batch(rewards, label, window=21, sigma=1, fig="rewards"):
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
                    alpha=0.1)#,label=f"{sigma} "r"$\sigma$ CI")

def plot_rewards_comparison_alpha(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    rewards, n = read_all_rewards(Path("batch=2022-04-03-16-07-43_a=1e-02_g=9e-01_hlay=(512, 256, 64)"))
    label = f"$\\alpha = 0.01$ (n={n})"
    plot_rewards_batch(rewards, label)
    
    rewards, n = read_all_rewards(Path("batch=2022-04-03-23-45-18_a=1e-01_g=9e-01_hlay=(512, 256, 64)"))
    label = f"$\\alpha = 0.1$ (n={n})"
    plot_rewards_batch(rewards, label)
    
    rewards, n = read_all_rewards(Path("batch=2022-04-04-09-06-35_a=2e-01_g=9e-01_hlay=(512, 256, 64)"))
    label = f"$\\alpha = 0.25$ (n={n})"
    plot_rewards_batch(rewards, label)
    
    rewards, n = read_all_rewards(Path("batch=2022-04-04-12-04-15_a=5e-01_g=9e-01_hlay=(512, 256, 64)"), plot_all_paths=True)
    label = f"$\\alpha = 0.5$ (n={n})"
    plot_rewards_batch(rewards, label)
    
    ax.set_ylim(0., None)
    ax.set_xlim(0., 250)
    
    ax.axhline(200, ls='--', c='gray', label="200 Reward Limit")
    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'DQN Rewards during Training: Learning Rates Comparison')
    ax.legend()

    # fig.suptitle('example sup title', fontsize=16)

    plt.show()
    return


def plot_rewards_comparison_gamma(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    rewards, n = read_all_rewards(Path("batch=2022-04-03-16-07-43_a=1e-02_g=9e-01_hlay=(512, 256, 64)"))
    label = f"$\\gamma = 0.1$ (n={n})"
    plot_rewards_batch(rewards, label)
    
    rewards, n = read_all_rewards(Path("batch=2022-04-03-23-45-18_a=1e-01_g=9e-01_hlay=(512, 256, 64)"))
    label = f"$\\gamma = 0.3$ (n={n})"
    plot_rewards_batch(rewards, label)
    
    rewards, n = read_all_rewards(Path("batch=2022-04-04-09-06-35_a=2e-01_g=9e-01_hlay=(512, 256, 64)"))
    label = f"$\\gamma = 0.6$ (n={n})"
    plot_rewards_batch(rewards, label)
    
    rewards, n = read_all_rewards(Path("batch=2022-04-04-12-04-15_a=5e-01_g=9e-01_hlay=(512, 256, 64)"), plot_all_paths=True)
    label = f"$\\gamma = 0.9$ (n={n})"
    plot_rewards_batch(rewards, label)
    
    ax.set_ylim(0., None)
    ax.set_xlim(0., 250)
    
    ax.axhline(200, ls='--', c='gray', label="200 Reward Limit")
    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'DQN Rewards during Training: Discount Factors Comparison')
    ax.legend()

    # fig.suptitle('example sup title', fontsize=16)

    plt.show()
    return

def main():
    plot_rewards_comparison_alpha()


if __name__ == '__main__':
    main()
