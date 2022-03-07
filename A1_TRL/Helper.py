#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.special import expit
from cycler import cycler
from itertools import cycle

class AnnealScheduler(object):
    """
    Class for keeping track of anneals
    """
    def __init__(self, n_buffer=50, start=1., final=0.,
                 Q_tresh=1e-3, timesteps=50000,
                 max_reward=1.1, r_thresh=0.1,
                 percentage=0.5):

        super(AnnealScheduler).__init__()

        self.start = start
        self.final = final
        self.buffer = np.full(n_buffer, fill_value=1., dtype=np.float64)
        self.current_buffer_idx = 0
        self.timesteps = timesteps
        self.active = "inactive"

        self.Q_tresh = Q_tresh
        self.old_Q_sa = np.array([1])

        self.old_r = 1
        self.r_tresh = r_thresh

        self.percentage = percentage
        self.k = 1 / timesteps

        self._tracker = np.full(timesteps, fill_value=np.nan)

        self.function = {"q_error_anneal": self.q_error_anneal,
                         "r_diff_anneal": self.r_diff_anneal,
                         "linear_anneal": self.linear_anneal,
                         "logistic_anneal": self.logistic_anneal
                         }

    def q_error_anneal(self, new_Q_sa, **kwargs):
        ''' Annealing scheduler based on change in Q_sa averaged over buffer
        '''
        self.active = r"$\Delta Q_{s,a}$"
        error = np.max(np.abs(self.old_Q_sa - new_Q_sa)) / np.max((np.abs(self.old_Q_sa).max(), np.abs(new_Q_sa).max()))
        anneal_status = np.clip(error, a_min=0.,  a_max=1.) * np.clip(error - self.Q_tresh + 1., a_min=0.,  a_max=1.).astype(int)


        np.put(self.buffer, self.current_buffer_idx, anneal_status,
               mode="wrap")
        self._tracker[self.current_buffer_idx] = anneal_status
        self.current_buffer_idx += 1

        self.old_Q_sa = np.copy(new_Q_sa)

        return self.final + (self.start - self.final) * np.mean(self.buffer)

    def r_diff_anneal(self, new_r,  **kwargs):
        ''' Annealing scheduler based on r difference to some (estimated) max reward
        '''
        self.active = r"$\Delta r$"
        error = np.abs(self.old_r - new_r) / np.max((self.old_r, new_r))
        anneal_status = np.clip(error, a_min=0., a_max=1.) \
                        * np.clip(error - self.r_tresh + 1., a_min=0.,  a_max=1.).astype(int)
        np.put(self.buffer, ind=self.current_buffer_idx, v=anneal_status,
               mode="wrap")

        self._tracker[self.current_buffer_idx] = anneal_status
        self.current_buffer_idx += 1

        self.old_r = np.copy(new_r)

        # print(error, np.mean(self.buffer))

        return (self.final) + (self.start - self.final) * np.mean(self.buffer)

    def linear_anneal(self, t, **kwargs):
        ''' Linear annealing scheduler
        t: current timestep
        T: total timesteps
        start: initial value
        final: value after percentage*T steps
        percentage: percentage of T after which annealing finishes
        '''
        self.active = r"$linear$"

        return self.final + (self.start - self.final) * (self.timesteps - t) / self.timesteps

    def logistic_anneal(self, t, k=1, **kwargs):
        ''' Logistic annealing scheduler
        Since we are essentially asking a binary question: "Is it profitable to exploit?" - Y/N
        I would expect the answer to trend to a logistic function over the timesteps
        '''
        self.active = r"$logistic$"
        x = - k * self.k * (t - self.percentage * self.timesteps)

        return self.final + (self.start - self.final) * expit(x)

    def exponential_decay(self):
        """
        Not implemented because no time but I think this might also work quite well
        """
        return

    def __reset__(self, cval=1.):
        self.buffer = np.full(len(self.buffer), fill_value=cval, dtype=np.float64)
        self.current_buffer_idx = 0
        self.active = "inactive"

    @property
    def tracker(self):
        return self._tracker[np.isreal(self._tracker)]

    def show(self):
        self.start = 0.9
        self.final = 0.1
        self.buffer = np.full(10, fill_value=1., dtype=np.float64)
        self.current_buffer_idx = 0
        self.timesteps = 1000

        self.percentage = 0.5
        self.k = 1 / self.timesteps

        rng = np.random.default_rng()


        fig, ax = plt.subplots(nrows=1, ncols=1,
                                 constrained_layout=True,
                               sharex=True, sharey=True,
                                 figsize=(6.8, 4.5))
        x = np.linspace(0, 1, self.timesteps)

        for func in [self.linear_anneal, self.logistic_anneal]:
            ax.plot(x,  np.abs(np.array([func(xl * self.timesteps, k=10) for xl in x])), label=self.active)
            self.__reset__()

        self.__reset__(cval=0.)
        self.Q_tresh = 0.15
        self.Q_sa = 1
        ax.plot(x,
                np.abs(np.array([self.q_error_anneal(old_Q_sa=1, new_Q_sa=xl)
                                 for xl in np.flip(x)])), label=self.active + r" ($\eta=$"f"{self.Q_tresh:.2f})")
        self.__reset__(cval=0.)

        x_reward = np.linspace(1., 0., self.timesteps)
        for t, ls in zip([0.01, 0.1], ["solid", "dashed"]):
            self.r_tresh = t
            ax.plot(np.flip(x_reward),
                    np.abs(np.array([self.r_diff_anneal(old_r=1, new_r=xl)
                                     for xl in x_reward])), label=self.active + r" ($\eta=$"f"{t:.2f})",
                    ls=ls, c="r")
            self.__reset__(cval=0.)


        ax.axhline(0.5, c="gray", ls="dotted", linewidth=0.75)
        ax.axvline(0.5, c="gray", ls="dotted", linewidth=0.75)
        ax.axvline(1., c="black", ls="solid", linewidth=0.75)

        # ax.axhline(self.max_reward, c="orange", ls="dashed", linewidth=1, label=r"$(r)_{max, ~expected}$")

        ax.axhline(self.start, c="black", ls="dashed", linewidth=1, label="Start/Final")
        ax.axhline(self.final, c="black", ls="dashed", linewidth=1)

        ax.set_xlim(0, 1.)
        ax.set_ylim(0, 1.)
        ax.legend()
        ax.set_ylabel('Annealing [-]\n(exploring: 1, exploiting: 0)')
        ax.set_xlabel('Runtime, 'r'$\Delta_{r}$ or $\Delta_{Q_{s,a}}$ [-]')
        # plt.show()
        fig.savefig("annealing", dpi=300)

class LearningCurvePlot:

    def __init__(self,title=None):
        self.fig, self.ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Reward')

        # self.cc = (cycler(color=['crimson', "darkblue", "magenta", "cyan", "darkorange", "yellow", "olive", "limegreen"]) *
        #            cycler(linestyle=['solid', 'dashed', 'dotted']))

        plt.rc('lines', linewidth=0.75)
        # plt.rc('axes', prop_cycle=self.cc)
        # self.ax.set_prop_cycle(self.cc)

        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self, y, label=None, color=None, ls=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(y, label=label, color=color, ls=ls)
        else:
            self.ax.plot(y, color=color, ls=ls)
    
    def set_ylim(self,lower,upper):
        self.ax.set_ylim([lower,upper])

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name, dpi=300)

def plot_q_value_func(Q_sa, env, title):
    fig, ax = plt.subplots(nrows=1, ncols=1,
                                 constrained_layout=True, subplot_kw={'aspect': 1},
                                 sharex=True, sharey=True,
                                 figsize=(10, 7))

    action_effects = env.action_effects
    winds = np.array(env.winds)


    # safety
    # np.nan_to_num(Q_sa, copy=False, nan=0., posinf=0, neginf=0)

    vstars = np.zeros((10, 7))
    for s, vstar in enumerate(np.max(Q_sa, axis=1)):
        vstars[tuple(env._state_to_location(s))] = vstar

    im = ax.imshow(vstars.T, vmin=None, vmax=None,
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

    if title is None:
        title = r'$V^*(s)$ and $\pi^*(s)$'
    ax.set_title(title,
                 fontsize=13)


    cb = plt.colorbar(im, location="right",
                      fraction=0.046, pad=0.04)
    cb.set_label(label=r"$V(s) = \max_a \left[ Q(s,~a) \right]$", fontsize=13)
    ax.set_ylabel('y [cells]')

    ax.set_xlabel('x [cells]')
    plt.savefig("Q_value_max.png", dpi=300, format="png", )
    # plt.show()

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    ### Note: I had to pass the mirror parameter since my setup does not ply nice with MKL:
    # Intel MKL ERROR: Parameter 6 was incorrect on entry to DGELSD.
    # this means that the boundary conditions in the savgol_filter cannot be properly evaluated by extrapolating
    # the last fitted polynomial. Instead I use mirror to use the last n previous values in reverse order.
    # This should be ok since we do not expect large changes towards the end.
    return savgol_filter(y, window, poly, mode="mirror")

def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax 
    return np.exp(z)/np.sum(np.exp(z)) # compute softmax

def argmax(x):
    ''' Own variant of np.argmax with random tie breaking '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:  # blank exception? Oof..
        return np.argmax(x)

def linear_anneal(t,T,start,final,percentage):
    ''' Linear annealing scheduler
    t: current timestep
    T: total timesteps
    start: initial value
    final: value after percentage*T steps
    percentage: percentage of T after which annealing finishes
    ''' 
    final_from_T = int(percentage*T)
    if t > final_from_T:
        return final
    else:
        return final + (start - final) * (final_from_T - t)/final_from_T

def q_error_logical_anneal(t,T,start,final,percentage):
    ''' Linear annealing scheduler
    t: current timestep
    T: total timesteps
    start: initial value
    final: value after percentage*T steps
    percentage: percentage of T after which annealing finishes
    '''
    thresh = 0.0001
    final_from_T = int(percentage*T)
    if t > final_from_T:
        return final
    else:
        return final + (start - final) * (final_from_T - t)/final_from_T

def r_diff_logical_anneal(t,T,start,final,percentage):
    ''' Linear annealing scheduler
    t: current timestep
    T: total timesteps
    start: initial value
    final: value after percentage*T steps
    percentage: percentage of T after which annealing finishes
    '''
    final_from_T = int(percentage*T)
    if t > final_from_T:
        return final
    else:
        return final + (start - final) * (final_from_T - t)/final_from_T

if __name__ == '__main__':
    a = AnnealScheduler()
    a.show()