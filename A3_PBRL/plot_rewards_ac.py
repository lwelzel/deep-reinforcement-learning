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

def read_all_rewards(dir, fig="rewards", plot_all_paths=False, rewards=None):
    if rewards is None:
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
                smooth_rewards = smooth(l_rewards[np.nonzero(l_rewards)], window=51, poly=1)
                # smooth_rewards = smooth(l_rewards, window=1, poly=0)  # for where we want to look at single episodes

                ax.plot(smooth_rewards,
                        c="gray", alpha=0.1, ls="dashed", lw=0.75)
                ax.scatter(len(smooth_rewards) - 1.,
                           smooth_rewards[-1],
                           c="k",
                           s=2.5)

        rewards = rewards[:, :min_len]
        return rewards, headers
    else:
        fig = plt.figure(num=fig)
        ax = np.array(fig.axes).flatten()[0]
        for reward in rewards:
            smooth_rewards = smooth(reward[np.nonzero(reward)], window=201, poly=1)
            # smooth_rewards = smooth(l_rewards, window=1, poly=0)  # for where we want to look at single episodes

            ax.plot(smooth_rewards,
                    c="gray", alpha=0.1, ls="dashed", lw=0.75)
            ax.scatter(len(smooth_rewards) - 1.,
                       smooth_rewards[-1],
                       c="k",
                       s=2.5)


def plot_rewards_batch(rewards, header, label, window=21, sigma=1, fig="rewards", clip=200., **kwargs):
    fig = plt.figure(num=fig)
    ax = np.array(fig.axes).flatten()[0]

    mean_rewards = np.mean(rewards, axis=0)
    std_rewards = np.std(rewards, axis=0)

    smooth_mean = smooth(mean_rewards, window=window, poly=1)
    smooth_std = smooth(std_rewards, window=window * 3, poly=1)

    max_reward = float(header[0]["max_reward"])


    ax.plot(smooth_mean,
            label=label,
            **kwargs)
    ax.fill_between(np.arange(len(smooth_mean)),
                    np.clip(smooth_mean + smooth_std * sigma, a_min=0., a_max=clip),
                    np.clip(smooth_mean - smooth_std * sigma, a_min=0., a_max=clip),
                    alpha=0.1)#,label=f"{sigma} "r"$\sigma$ CI")


def plot_rewards_a2c_batch(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    rewards, headers = read_all_rewards(Path("A2C_runs/batch_size/para10"))
    label = f"Batch size 10 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/batch_size/para100"))
    label = f"Batch size 100 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/batch_size/para1000"))
    label = f"Batch size 1000 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/default"))
    label = f"Default (60) (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)


    ax.set_ylim(0., None)
    ax.set_xlim(0., 1000)

    ax.axhline(200, ls='--', c='gray', label="200 Reward Limit")
    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'A2C Effect of Batch Size')
    ax.legend()


    plt.savefig("./plots/ac_batch_size.png")
    plt.show()
    plt.close()
    return


def plot_rewards_a2c_ablation(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    rewards, headers = read_all_rewards(Path("A2C_runs/BLS/paraFalse"))
    label = f"AC - BLS (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/BS/paraFalse"))
    label = f"AC - BS (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/nothing/paraFalse"))
    label = f"AC - BS - BLS (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/default"))
    label = f"A2C (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/default_anneal"))
    label = f"A2C + policy anneal (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.1_decay_para1.0"))
    label = f"A2C + policy & batch size anneal (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)


    ax.set_ylim(0., None)
    ax.set_xlim(0., 1000)

    ax.axhline(200, ls='--', c='gray', label="200 Reward Limit")
    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'A2C Ablation Study')
    ax.legend()

    plt.savefig("./plots/ac_ablation.png")
    plt.show()
    plt.close()
    return


def plot_rewards_a2c_anneal(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    rewards, headers = read_all_rewards(Path("A2C_runs/default"))
    label = f"Default (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/default_anneal"))
    label = f"Default Anneal (T0=1, alpha=0.99) (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/decay_tanhtemp/d0.8para100.0"))
    label = f"T0=100, alpha=0.8 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/decay_tanhtemp/d0.8para1000.0"))
    label = f"T0=1000, alpha=0.8 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/decay_tanhtemp/d0.8para10000.0"))
    label = f"T0=10000, alpha=0.8 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/decay_tanhtemp/d0.9para100.0"))
    label = f"T0=100, alpha=0.9 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/decay_tanhtemp/d0.9para1000.0"))
    label = f"T0=1000, alpha=0.9 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/decay_tanhtemp/d0.9para10000.0"))
    label = f"T0=10000, alpha=0.9 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/decay/para0.8"))
    label = f"T0=1, alpha=0.8 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/decay/para0.95"))
    label = f"T0=1, alpha=0.95 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/decay/para0.999"))
    label = f"T0=1, alpha=0.999 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/tanhtemp/para1.0"))
    label = f"T0=1, alpha=0.99 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/tanhtemp/para10.0"))
    label = f"T0=10, alpha=0.99 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/tanhtemp/para100.0"))
    label = f"T0=100, alpha=0.99 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    ax.set_ylim(0., None)
    ax.set_xlim(0., 1000)

    ax.axhline(200, ls='--', c='gray', label="200 Reward Limit")
    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'A2C Effect of Anneal')
    ax.legend()

    plt.savefig("./plots/ac_anneal.png")
    plt.show()
    plt.close()
    return


def plot_rewards_a2c_discount(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    rewards, headers = read_all_rewards(Path("A2C_runs/discount/para0.5"))
    label = f"gamma = 0.5 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/discount/para0.8"))
    label = f"gamma = 0.8 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)


    rewards, headers = read_all_rewards(Path("A2C_runs/discount/para0.9"))
    label = f"gamma = 0.9 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/default"))
    label = f"Default (0.95) (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)


    ax.set_ylim(0., None)
    ax.set_xlim(0., 1000)

    ax.axhline(200, ls='--', c='gray', label="200 Reward Limit")
    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'A2C Effect of Discount Rate')
    ax.legend()


    plt.savefig("./plots/ac_discount.png")
    plt.show()
    plt.close()
    return


def plot_rewards_a2c_nn(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    rewards, headers = read_all_rewards(Path("A2C_runs/default"))
    label = f"Default (32, 16) (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/layers/para(8, 4)"))
    label = f"(8, 4) (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/layers/para(16, 8)"))
    label = f"(16, 8) (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/layers/para(64, 32)"))
    label = f"(64, 32) (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/layers/para(254, 64, 32)"))
    label = f"(254, 64, 32) (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/layers/para(254, 64, 32, 16)"))
    label = f"(254, 64, 32, 16) (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)


    ax.set_ylim(0., None)
    ax.set_xlim(0., 1000)

    ax.axhline(200, ls='--', c='gray', label="200 Reward Limit")
    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'A2C Effect of Hidden Layers')
    ax.legend()

    plt.savefig("./plots/ac_layers.png")
    plt.show()
    plt.close()
    return


def plot_rewards_a2c_learning(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    rewards, headers = read_all_rewards(Path("A2C_runs/default"))
    label = f"Default (5e-3) (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/learning_rate/para0.1"))
    label = f"1e-1 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/learning_rate/para0.01"))
    label = f"1e-2 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/learning_rate/para0.001"))
    label = f"1e-3 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/learning_rate/para0.0001"))
    label = f"1e-4 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/learning_rate/para1e-05"))
    label = f"1e-5 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)



    ax.set_ylim(0., None)
    ax.set_xlim(0., 1000)

    ax.axhline(200, ls='--', c='gray', label="200 Reward Limit")
    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'A2C Effect of Learning Rate')
    ax.legend()

    plt.savefig("./plots/ac_learning_rate.png")
    plt.show()
    plt.close()
    return

def plot_rewards_a2c_activation_actor(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    rewards, headers = read_all_rewards(Path("A2C_runs/default"))
    label = f"Default (tanh) (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/act_fun/actor_paraelu"))
    label = f"elu (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/act_fun/actor_paraexponential"))
    label = f"exp (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/act_fun/actor_parahard_sigmoid"))
    label = f"sigmoid (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/act_fun/actor_paralinear"))
    label = f"lin (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/act_fun/actor_pararelu"))
    label = f"relu (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/act_fun/actor_parasoftmax"))
    label = f"softmax (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)



    ax.set_ylim(0., None)
    ax.set_xlim(0., 1000)

    ax.axhline(200, ls='--', c='gray', label="200 Reward Limit")
    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'A2C Effect of Actor Activation Function')
    ax.legend()

    plt.savefig("./plots/ac_actor_act_fun.png")
    plt.show()
    plt.close()
    return

def plot_rewards_a2c_activation_critic(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    rewards, headers = read_all_rewards(Path("A2C_runs/default"))
    label = f"Default (tanh) (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/act_fun/critic_paraelu"))
    label = f"elu (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/act_fun/critic_paraexponential"))
    label = f"exp (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/act_fun/critic_parahard_sigmoid"))
    label = f"sigmoid (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/act_fun/critic_paralinear"))
    label = f"lin (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/act_fun/critic_pararelu"))
    label = f"relu (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)

    rewards, headers = read_all_rewards(Path("A2C_runs/act_fun/critic_parasoftmax"))
    label = f"softmax (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label)



    ax.set_ylim(0., None)
    ax.set_xlim(0., 1000)

    ax.axhline(200, ls='--', c='gray', label="200 Reward Limit")
    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'A2C Effect of Critic Activation Function')
    ax.legend()

    plt.savefig("./plots/ac_critic_act_fun.png")
    plt.show()
    plt.close()
    return

def plot_rewards_a2c_batch_anneal(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.05_decay_para0.85"))
    label = f"nu=0.05, omega=0.85 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="dotted")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.05_decay_para0.875"))
    label = f"nu=0.05, omega=0.875 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="dotted")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.05_decay_para0.9"))
    label = f"nu=0.05, omega=0.9 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="dotted")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.05_decay_para0.925"))
    label = f"nu=0.05, omega=0.925 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="dotted")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.05_decay_para0.95"))
    label = f"nu=0.05, omega=0.95 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="dotted")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.05_decay_para1.0"))
    label = f"nu=0.05, omega=1 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="dotted")

    # ==========================

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.1_decay_para0.85"))
    label = f"nu=0.1, omega=0.85 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="solid")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.1_decay_para0.875"))
    label = f"nu=0.1, omega=0.875 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="solid")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.1_decay_para0.9"))
    label = f"nu=0.1, omega=0.9 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="solid")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.1_decay_para0.925"))
    label = f"nu=0.1, omega=0.925 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="solid")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.1_decay_para0.95"))
    label = f"nu=0.1, omega=0.95 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="solid")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.1_decay_para1.0"))
    label = f"nu=0.1, omega=1 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="solid")

    # ==========================

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.15_decay_para0.85"))
    label = f"nu=0.15, omega=0.85 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="dashed")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.15_decay_para0.875"))
    label = f"nu=0.15, omega=0.875 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="dashed")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.15_decay_para0.9"))
    label = f"nu=0.15, omega=0.9 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="dashed")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.15_decay_para0.925"))
    label = f"nu=0.15, omega=0.925 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="dashed")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.15_decay_para0.95"))
    label = f"nu=0.15, omega=0.95 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="dashed")

    rewards, headers = read_all_rewards(Path("A2C_runs/batch/base_0.15_decay_para1.0"))
    label = f"nu=0.15, omega=1 (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linestyle="dashed")

    # ==========================

    rewards, headers = read_all_rewards(Path("A2C_runs/default"))
    label = f"Default (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linewidth=2, c="black")

    rewards, headers = read_all_rewards(Path("A2C_runs/default_anneal"))
    label = f"Default Anneal (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, linewidth=2, c="black", linestyle="dashed")



    ax.set_ylim(0., None)
    ax.set_xlim(0., 1000)

    ax.axhline(200, ls='--', c='gray', label="200 Reward Limit")
    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(r'A2C Effect of Batch Annealing ($L_{B}(0)=20$, policy anneal active)')
    ax.legend()


    plt.savefig("./plots/ac_batch_anneal.png")
    plt.show()
    plt.close()
    return

def plot_rewards_a2c_long(*args):
    fig, ax = plt.subplots(num="rewards",
                           nrows=1, ncols=1,
                           constrained_layout=True,
                           figsize=(9, 6))
    # read_all_rewards(None, fig="rewards", plot_all_paths=True, rewards=rewards)

    rewards, headers = read_all_rewards(Path("A2C_runs/long"))
    label = f"A2C + policy & batch size anneal (n={len(headers)})"
    plot_rewards_batch(rewards, headers, label, clip=2500., window=101)
    read_all_rewards(None, fig="rewards", plot_all_paths=True, rewards=rewards)


    ax.set_ylim(0., None)
    ax.set_xlim(0., None)

    ax.axhline(2500, ls='--', c='gray', label="2500 Reward Limit")
    ax.set_xlabel('Episode [-]')
    ax.set_ylabel('Mean reward [-]')
    ax.set_title(f'A2C Ablation Study')
    ax.legend()

    plt.savefig("./plots/ac_ablation.png")
    plt.show()
    plt.close()
    return

def main():
    plot_rewards_a2c_batch()
    plot_rewards_a2c_ablation()
    plot_rewards_a2c_anneal()
    plot_rewards_a2c_discount()
    plot_rewards_a2c_nn()
    plot_rewards_a2c_learning()
    plot_rewards_a2c_activation_actor()
    plot_rewards_a2c_activation_critic()
    plot_rewards_a2c_batch_anneal()
    plot_rewards_a2c_long()


if __name__ == '__main__':
    main()
