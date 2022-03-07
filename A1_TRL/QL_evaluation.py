import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import MarkovTransitionField
from Environment import StochasticWindyGridworld
from Helper import argmax
from Q_learning import QLearningAgent, q_learning

def q_val_vis(env=None, gamma=1., threshold=0.001):
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'softmax'  # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0

    if env is None:
        env = StochasticWindyGridworld(initialize_model=True)

    fig, all_axes = plt.subplots(nrows=3, ncols=5,
                                 constrained_layout=True, subplot_kw={'aspect': 1},
                                 sharex=True, sharey=True,
                                 figsize=(22, 4.8 * 2.8))

    epsilons = [0.01, 0.05, 0.2]
    temps = [0.01, 0.1, 1.0]
    iter_stops = [5, 25, 50, 250, 2500]

    action_effects = env.action_effects
    winds = np.array(env.winds)

    for j, (epsilon, temp, axs) in enumerate(zip(epsilons, temps, all_axes)):
        axes = np.array(axs)
        for i, (iter_stop, ax) in enumerate(zip(iter_stops, axes.flatten())):
            rewards, Q_sa = q_learning(iter_stop, learning_rate, gamma, policy, epsilon, temp,
                                       plot=False, ret_Q_sa=True,
                                       max_steps=250)

            # safety
            # np.nan_to_num(Q_sa, copy=False, nan=0., posinf=0, neginf=0)

            vstars = np.zeros((10, 7))
            for s, vstar in enumerate(np.max(Q_sa, axis=1)):
                vstars[tuple(env._state_to_location(s))] = vstar

            im = ax.imshow(vstars.T, vmin=-5, vmax=35,
                           cmap='viridis', origin="lower")
            # this is bad, but also not that bad
            ax.hlines(y=np.arange(0, 7) + 0.5,
                      xmin=np.full(7, 0) - 0.5,
                      xmax=np.full(7, 10) - 0.5,
                      linewidth=0.5,
                      color="k")
            ax.vlines(x=np.arange(0, 10) + 0.5,
                      ymin=np.full(10, 0) - 0.5,
                      ymax=np.full(10, 7) - 0.5,
                      linewidth=0.5,
                      color="k")

            # this is even worse but its late
            for s, a in enumerate(np.argmax(Q_sa, axis=1)):
                loc = env._state_to_location(s)
                if tuple(loc) == (7, 3):
                    ax.annotate("G", loc + 1,
                                loc,
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=18, weight="bold", c="w", textcoords='data',
                                xycoords="data",
                                arrowprops=dict(arrowstyle='<|-',
                                                fc='w',
                                                color="w",
                                                alpha=0))
                elif tuple(loc) == (0, 3):
                    ax.annotate(" S ", loc + np.array(action_effects[a]) * 0.75,
                                loc,
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=18, weight="bold", c="w", textcoords='data',
                                xycoords="data",
                                arrowprops=dict(arrowstyle='-|>',
                                                fc='w',
                                                color="w",
                                                alpha=1))
                else:
                    ax.annotate("     ", loc + np.array(action_effects[a]) * 0.75,
                                loc,
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=18, weight="bold", c="w", textcoords='data',
                                xycoords="data",
                                arrowprops=dict(arrowstyle='-|>',
                                                fc='w',
                                                color="w",
                                                alpha=1))

                    ax.annotate("", loc + np.array([0, (winds[loc[0]]) * 0.15]),
                                loc - np.array([0, (winds[loc[0]]) * 0.15]),
                                horizontalalignment='center', verticalalignment='center',
                                fontsize=10, c="w", textcoords='data',
                                xycoords="data",
                                arrowprops=dict(arrowstyle='-|>',
                                                fc='gray',
                                                color="gray",
                                                alpha=0.8))

            if policy == 'egreedy':
                ax.set_title(r'$V^*(s)$ and $\pi^*(s)$'f"  ("r"$\varepsilon$"f": {epsilon:.2f}, it: {iter_stop:02.0f})",
                             fontsize=13)
            elif policy == 'softmax':
                ax.set_title(r'$V^*(s)$ and $\pi^*(s)$'f"  ("r"$T$"f": {temp:.2f}, it: {iter_stop:02.0f})",
                             fontsize=13)

        cb = plt.colorbar(im, ax=axes, location="bottom",
                          fraction=0.95, aspect=80)
        cb.set_label(label=r"$V(s) = \max_a \left[ Q(s,~a) \right]$", fontsize=13)
        axes[0].set_ylabel('y [cells]')

    for ax in all_axes[-1]:
        ax.set_xlabel('x [cells]')
    # fig.suptitle(r'TRL Dynamic Programming: $\pi(s)$ at different $\Delta$', fontsize=18, weight="bold")
    # plt.savefig("TRL_QL_pi_ev.png", dpi=350, format="png", )
    plt.show()



if __name__ == '__main__':
    q_val_vis()
