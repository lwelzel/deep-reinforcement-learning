# Overview
This repo contains a very cool REINFORCE and A2C/SAC agent and the functions to train them. 
Please do not feed or pet the agent.

requirements*.txt and environment*.yml contains list of dependencies.
actor_critic.py contains the A2C agent and environment. wrapper.py adds multiprocessing for the agent.
Run plot_rewards_*.py for plots shown in report (requires saved rewards)
If you wish to run any experiments yourself, a2c_experiment.py runs the A2C agent/environment (using wrapper) 
and saves rewards. Change the parameters in main() depending on which settings you wish to run.
Additional supplementary functions located in helper.py.


# Read Me:
## Use A2C:
TLDR:
1. Run a2c_experiment.py, actor_critic.py or wrapper.py to train agents.
2. Run plot_rewards_ac.py to plot specific results or the saved results in the BATCHES directory

### Preparation
1. Set up the environment with all required dependencies using conda and requirements_py37.yml (preferred) or pip and requirements_py37.txt
2. Copy the saved results from the studies (not submitted via brightspace) into a BATCHES dir 
### Running the A2C
1. Run any of the programs below with the desired inputs.
   1. a2c_experiment.py for a list of studies
   2. actor_critic.py for a test training
   3. wrapper.py for a parallel training of many agents (watch out, this might cause flashing lights ont the console due to scrambled print statements and progress bars)
2. A progress bar will show
3. The result (network and rewards) will be saved intermittently 
### Analyzing the DQNResult
1. Run plot_rewards_ac.py to plot specific results or the saved results in the A2C_runs directory

## Use REINFORCE / CEM-Evolutionary
TLDR:
1. Run training_functions.py or experiment.py
2. Run plot_rewards.py to plot specific results or the saved results in the BATCHES directory
### Preparation
1. Set up the environment with all required dependencies using pip and requirements_py37.txt
2. Copy the saved results from the studies (not submitted via brightspace) into a BATCHES dir 
### Running the A2C
1. Run any of the programs below with the desired inputs.
   1. experiment.py for a list of studies (contains REINFORCE and CEM)
   2. training_functions for test training for either REINFORCE or CEM 
2. A progress bar will show
3. The result (rewards) will be saved intermittently 
### Analyzing the DQNResult
1. Run plot_rewards.py to plot specific results or the saved results in the runs directory

---
---

## Authors & Copyleft
See project overview for authors.
