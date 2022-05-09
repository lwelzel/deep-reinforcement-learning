# Overview
This repo contains a very cool DQN agent and the functions to train them. 
Please do not feed or pet the agent.

requirements-option1.txt contains list of dependencies.
dqn_base_class.py contains the agent and environment. many_dqn_wrapper.py adds multiprocessing for deep Q learning agent.
Run plot_rewards.py for plots shown in report (requires saved rewards from BATCHES folder)
If you wish to run any experiments yourself, Experiment.py runs the agent/environment (using many_dqn_wrapper) 
and saves rewards and DQN weights. Change the parameters in main() depending on which settings you wish to run.
Additional supplementary functions located in helper.py.


# Read Me:
## Use:
TLDR:
1. Run Experiment.py, dqn_base_class.py or many_dqn_wrapper.py to train agents.
2. Run plot_rewards.py to plot specific results or the saved results in the BATCHES directory

### Preparation
1. Set up the environment with all required dependencies using conda and requirements-option1.yml (preferred) or pip and requirements-option1.txt
2. Copy the saved results from the studies (not submitted via brightspace) into a BATCHES dir 
### Running the DQN
1. Run any of the programs below with the desired inputs.
   1. Experiment.py for a list of studies
   2. dqn_base_class.py for a test training
   3. many_dqn_wrapper.py for a parallel training of many agents (watch out, this might cause flashing lights ont the console due to scrambled print statements and progress bars)
2. A progress bar will show
3. The result (network and rewards) will be saved intermittently 
### Analyzing the DQNResult
1. Run plot_rewards.py to plot specific results or the saved results in the BATCHES directory

---
---

## Authors & Copyleft
See project overview for authors.
