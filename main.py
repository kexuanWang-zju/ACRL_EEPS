from ACRL_EEPS import ACRL_EEPS
from SCAOPO import SCAOPO
import argparse
from scipy.io import savemat
import numpy as np

def main(args):
    cuda_index = 0

    ### old policy
    #args.start_seed = 0
    #for seed_index in range(args.start_seed, args.start_seed+1):
    #    args.seed = seed_index
    #    args.shape_flag = 1
    #    args.TL_flag = 0
    #    args.old_policy_flag = 1
    #    name = "QGPS_shape" + str(args.shape_flag) + "_TL" + str(args.TL_flag) + "_old" + str(args.old_policy_flag)
    #    print(name)
    #    if seed_index == args.start_seed:
    #        dataset_name1 = name + "_reward_begin_save.mat"
    #        dataset_name2 = name + "_cost_begin_save.mat"
    #        temp = np.zeros((args.num_old_policy, args.num_iteration))
    #        savemat(dataset_name1, {"array": temp})
    #        savemat(dataset_name2, {"array": temp})
    #    ACRL_EEPS(args, name, cuda_index)

    # test
    args.start_seed = 11
    num_seed = 10
    for seed_index in range(args.start_seed, args.start_seed+num_seed):
        args.seed = seed_index

        args.shape_flag = 1
        args.TL_flag = 1
        args.old_policy_flag = 0
        name = str(args.start_seed)+"_QGPS_shape" + str(args.shape_flag) + "_TL" + str(args.TL_flag) + "_old" + str(args.old_policy_flag)
        if seed_index == args.start_seed:
            dataset_name1 = name + "_reward_begin_save.mat"
            dataset_name2 = name + "_cost_begin_save.mat"
            temp = np.zeros((num_seed, args.num_iteration))
            savemat(dataset_name1, {"array": temp})
            savemat(dataset_name2, {"array": temp})
        print(name)
        ACRL_EEPS(args, name, cuda_index)


        args.shape_flag = 1
        args.TL_flag = 0
        args.old_policy_flag = 0
        name = str(args.start_seed)+"_QGPS_shape" + str(args.shape_flag) + "_TL" + str(args.TL_flag) + "_old" + str(args.old_policy_flag)
        if seed_index == args.start_seed:
            dataset_name1 = name + "_reward_begin_save.mat"
            dataset_name2 = name + "_cost_begin_save.mat"
            temp = np.zeros((num_seed, args.num_iteration))
            savemat(dataset_name1, {"array": temp})
            savemat(dataset_name2, {"array": temp})
        print(name)
        ACRL_EEPS(args, name, cuda_index)


        args.shape_flag = 0
        args.TL_flag = 0
        args.old_policy = 0
        name = str(args.start_seed)+"_QGPS_shape" + str(args.shape_flag) + "_TL" + str(args.TL_flag) + "_old" + str(args.old_policy)
        if seed_index == args.start_seed:
            dataset_name1 = name + "_reward_begin_save.mat"
            dataset_name2 = name + "_cost_begin_save.mat"
            temp = np.zeros((num_seed, args.num_iteration))
            savemat(dataset_name1, {"array": temp})
            savemat(dataset_name2, {"array": temp})
        print(name)
        ACRL_EEPS(args, name, cuda_index)


        name = str(args.start_seed)+"_SCAOPO"
        print(name)
        if seed_index == args.start_seed:
            dataset_name1 = name + "_reward_begin_save.mat"
            dataset_name2 = name + "_cost_begin_save.mat"
            temp = np.zeros((num_seed, args.num_iteration))
            savemat(dataset_name1, {"array": temp})
            savemat(dataset_name2, {"array": temp})
        temp = args.num_new_data
        args.num_new_data = 1000
        args.grad_T = args.num_new_data
        args.T = int(args.num_new_data / 2)
        SCAOPO(args, name, cuda_index)
        args.num_new_data = temp
        args.grad_T = args.num_new_data
        args.T = int(args.num_new_data / 2)



alpha_pow = 0.6
beta_pow = 0.7
gamma_pow = 0.3
gamma_pow_reward = gamma_pow
gamma_pow_cost = gamma_pow
tau_reward = 1
tau_cost = 1

user_num = 4
num_new_data = 200
grad_T = num_new_data
T = int(num_new_data/2)                                                   
Q_update_time = 10
window = 2000
num_iteration = 500

shape_flag = 1
TL_flag = 1
old_policy_flag = 1
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--start_seed', type=int, default=0)
parser.add_argument('--TL_flag', type=int, default=TL_flag)
parser.add_argument('--old_policy_flag', type=int, default=old_policy_flag)
parser.add_argument('--shape_flag', type=int, default=shape_flag)
parser.add_argument('--alpha_pow', type=float, default=alpha_pow)
parser.add_argument('--beta_pow', type=float, default=beta_pow)
parser.add_argument('--gamma_pow_reward', type=float, default=gamma_pow_reward)
parser.add_argument('--gamma_pow_cost', type=float, default=gamma_pow_cost)
parser.add_argument('--tau_reward', type=float, default=tau_reward)
parser.add_argument('--tau_cost', type=float, default=tau_cost)
parser.add_argument('--user_num', type=int, default=user_num)
parser.add_argument('--num_new_data', type=int, default=num_new_data)
parser.add_argument('--num_iteration', type=int, default=num_iteration)
parser.add_argument('--grad_T', type=int, default=grad_T)
parser.add_argument('--T', type=int, default=T)
parser.add_argument('--Q_update_time', type=int, default=Q_update_time)
parser.add_argument('--window', type=int, default=window)
parser.add_argument('--num_old_policy', type=int, default=5)
args = parser.parse_args()

main(args)