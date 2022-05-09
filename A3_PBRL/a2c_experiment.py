from wrapper import run_parallel_a2c

# might not be memory safe? No idea tbh

# conda activate drl37
# cd C:\Users\lukas\Documents\Git\deep-reinforcement-learning\A3_PBRL
# python a2c_experiment.py

def do_ac_experiment():

    # # default
    # paras = [1, 2]
    # for para in paras:
    #     sdir = f"runs/default/"
    #     run_parallel_a2c(sdir=sdir)
    #
    # paras = [1, 2]
    # for para in paras:
    #     sdir = f"runs/default_anneal/"
    #     run_parallel_a2c(use_AN=True, sdir=sdir)
    #
    # # batch size
    # paras = [10, 100, 1000]
    # for para in paras:
    #     sdir = f"runs/batch_size/para{para}"
    #     run_parallel_a2c(batch_size=para, sdir=sdir)
    #
    # # BS
    # paras = [False]
    # for para in paras:
    #     sdir = f"runs/BS/para{para}"
    #     run_parallel_a2c(use_BS=para, sdir=sdir)
    #
    # # BLS
    # paras = [False]
    # for para in paras:
    #     sdir = f"runs/BLS/para{para}"
    #     run_parallel_a2c(use_BLS=para, sdir=sdir)
    #
    # # nothing
    # paras = [False]
    # for para in paras:
    #     sdir = f"runs/nothing/para{para}"
    #     run_parallel_a2c(use_BLS=para, use_BS=para, sdir=sdir)
    #
    # # anneal tanh-temp
    # paras = [1., 10., 100.]
    # for para in paras:
    #     sdir = f"runs/tanhtemp/para{para}"
    #     run_parallel_a2c(tanh_temp=para, use_AN=True, sdir=sdir)
    #
    # # anneal decay
    # paras = [0.8, 0.95, 0.999]
    # for para in paras:
    #     sdir = f"runs/decay/para{para}"
    #     run_parallel_a2c(decay=para, use_AN=True, sdir=sdir)
    #
    # # anneal decay_tanhtemp
    # paras = [100., 1000., 10000.,]
    # for para in paras:
    #     sdir = f"runs/decay_tanhtemp/d0.8para{para}"
    #     run_parallel_a2c(decay=0.8, tanh_temp=para, use_AN=True, sdir=sdir)
    #
    # # anneal decay_tanhtemp
    # paras = [100., 1000., 10000.,]
    # for para in paras:
    #     sdir = f"runs/decay_tanhtemp/d0.9para{para}"
    #     run_parallel_a2c(decay=0.9, tanh_temp=para, use_AN=True, sdir=sdir)
    #
    # # discount
    # paras = [0.9, 0.8, 0.5]
    # for para in paras:
    #     sdir = f"runs/discount/para{para}"
    #     run_parallel_a2c(discount=para, sdir=sdir)
    #
    # # learning_rate
    # paras = [1e-5, 1e-4, 1e-2, 1e-3, 1e-1]
    # for para in paras:
    #     sdir = f"runs/learning_rate/para{para}"
    #     run_parallel_a2c(learning_rate=para, sdir=sdir)
    #
    # # layers
    # paras = [(16, 8), (8, 4), (64, 32), (254, 64, 32), (254, 64, 32, 16)]
    # for para in paras:
    #     sdir = f"runs/layers/para{para}"
    #     run_parallel_a2c(hidden_layers_actor=para, hidden_layers_critic=para, sdir=sdir)

    # # activation functions actor
    # paras = ["elu", "exponential", "hard_sigmoid", "linear", "relu", "softmax"]
    # for para in paras:
    #     sdir = f"runs/act_fun/actor_para{para}"
    #     run_parallel_a2c(hidden_act_actor=para, sdir=sdir)
    #
    # # activation functions critic
    # paras = ["elu", "exponential", "hard_sigmoid", "linear", "relu", "softmax"]
    # for para in paras:
    #     sdir = f"runs/act_fun/critic_para{para}"
    #     run_parallel_a2c(hidden_act_critic=para, sdir=sdir)
    #
    # # batch anneal
    # paras = [0.85, 0.875, 0.9, 0.925, 0.95, 1.]
    # for para in paras:
    #     sdir = f"runs/batch/base_0.1_decay_para{para}"
    #     run_parallel_a2c(use_AN_batch=True, batch_decay=para, batch_size=20, use_AN=True, sdir=sdir)
    #
    # paras = [0.85, 0.875, 0.9, 0.925, 0.95, 1.]
    # for para in paras:
    #     sdir = f"runs/batch/base_0.05_decay_para{para}"
    #     run_parallel_a2c(use_AN_batch=True, batch_decay=para, batch_base=0.05, batch_size=20, use_AN=True, sdir=sdir)
    #
    # paras = [0.85, 0.875, 0.9, 0.925, 0.95, 1.]
    # for para in paras:
    #     sdir = f"runs/batch/base_0.15_decay_para{para}"
    #     run_parallel_a2c(use_AN_batch=True, batch_decay=para, batch_base=0.15, batch_size=20, use_AN=True, sdir=sdir)
    # full long run
    paras = [0]
    for para in paras:
        sdir = f"runs/long"
        run_parallel_a2c(use_AN_batch=True, batch_decay=1., batch_base=0.1,
                         batch_size=20, use_AN=True, max_reward=2500, num_epochs=5000, sdir=sdir)



if __name__ == '__main__':
    do_ac_experiment()