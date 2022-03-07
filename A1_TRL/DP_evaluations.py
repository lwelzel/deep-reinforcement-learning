import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from Environment import StochasticWindyGridworld
from Helper import argmax
from DynamicProgramming import QValueIterationAgent, Q_value_iteration


def q_val_vis(env=None, QIagent=None, gamma=1., threshold=0.001):
    if env is None:
        env = StochasticWindyGridworld(initialize_model=True)
    if QIagent is None:
        QIagent = Q_value_iteration(env, gamma, threshold)

    fig, axes = plt.subplots(nrows=1, ncols=5,
                             constrained_layout=True, subplot_kw={'aspect': 1},
                             sharey=True, # sharex=True, sharey=True,
                             figsize=(22, 4.8))

    thresholds = [np.inf, 30.0, 23.0, 22., 0.001]
    iter_stops = [100, 100, 100, 100, 100]

    action_effects = env.action_effects
    winds = np.array(env.winds)

    axes=np.array(axes)
    for i, (thr, iter_stop, ax) in enumerate(zip(thresholds, iter_stops, axes.flatten())):
        QIagent, iters = Q_value_iteration(env, gamma, thr, return_iters=True, iter_stop=iter_stop)

        vstars = np.zeros((10, 7))
        for s, vstar in enumerate(np.max(QIagent.Q_sa, axis=1)):
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
        for s, a in enumerate(np.argmax(QIagent.Q_sa, axis=1)):
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

        ax.set_xlabel('x [cells]')
        ax.set_title(r'$V^*(s)$ and $\pi^*(s)$ for $\eta=$'f'{thr}'f"  (iteration {iters:02.0f})",
                     fontsize=13)

    cb = plt.colorbar(im, ax=axes, location="bottom",
                      fraction=0.95, aspect=80)
    cb.set_label(label=r"$V(s) = \max_a \left[ Q(s,~a) \right]$", fontsize=13)
    axes[0].set_ylabel('y [cells]')
    # fig.suptitle(r'TRL Dynamic Programming: $\pi(s)$ at different $\Delta$', fontsize=18, weight="bold")
    plt.savefig("TRL_DP_pi_ev.png", dpi=350, format="png", )
    # plt.show()


def initial_value(env=None, QIagent=None, gamma=1., threshold=0.001):
    if env is None:
        env = StochasticWindyGridworld(initialize_model=True)

    n = 10
    v_3 = np.zeros(n)

    for i in range(n):
        QIagent = Q_value_iteration(env, gamma, threshold)
        v_3[i] = np.max(QIagent.Q_sa[3])


    print(f"Mean: {np.mean(v_3)}")
    print(f"5th and 95th percentiles: {np.percentile(v_3, [1, 99])}")

    n, bins, patches = plt.hist(v_3, 20, density=True, facecolor='k', alpha=1)

    plt.xlabel(r'$V^\star(s=3)$ [-]')
    plt.ylabel('Probability [-]')
    plt.show()

def trial(env=None, QIagent=None, gamma=1., threshold=0.001):
    if env is None:
        env = StochasticWindyGridworld(initialize_model=True)
    if QIagent is None:
        QIagent = Q_value_iteration(env, gamma, threshold)

    max_i = 50
    reward = np.zeros(max_i)

    # View optimal policy
    s = env.reset()
    for i in np.arange(max_i):
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        reward[i] = r
        s = s_next
        if done:
            break

if __name__ == '__main__':
    q_val_vis()
