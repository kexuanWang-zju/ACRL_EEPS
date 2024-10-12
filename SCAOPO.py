from environment import Environment_XR
from critic_opt import Critic
from utils import update_policy
from model import GaussianPolicy
from buffer import DataStorage
import numpy as np
import torch
from scipy.io import loadmat
from scipy.io import savemat


def SCAOPO(args, name, cuda_index):
    reward_average_from_begin = []
    cost_max_from_begin = []

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:'+str(cuda_index) if torch.cuda.is_available() else 'cpu')

    T = args.T
    grad_T = args.grad_T
    num_new_data = args.num_new_data
    alpha_pow = args.alpha_pow
    beta_pow = args.beta_pow
    tau_reward = args.tau_reward
    tau_cost = args.tau_cost
    window = args.window
    Nt, UE_num = 8, args.user_num  # The number of antennas and users.
    buffer_max = 1000
    delay_max = np.ones((args.user_num,), dtype=np.int32)*4
    state_dim = 2 * UE_num * Nt + 2 * np.sum(delay_max)
    action_dim = UE_num + 1

    env = Environment_XR(seed=args.seed, Nt=Nt, UE_num=UE_num, buffer_max=buffer_max, delay_max=delay_max)
    constraint_dim = UE_num
    constr_lim = np.ones((args.user_num,),dtype=np.float32)*0.1
    actor = GaussianPolicy(state_dim, action_dim, device, num_new_data)
    actor.to(device)
    buffer = DataStorage(T, num_new_data, state_dim, action_dim, constraint_dim, window, 1)
    critic = Critic(env, args.user_num, grad_T, state_dim, action_dim, constraint_dim,1, device, args.shape_flag)
    critic.to(device)
    # Initialization

    theta_dim = 0
    for para in actor.net.parameters():
        theta_dim += para.numel()
    real_theta_dim = theta_dim + action_dim
    paras_torch = torch.zeros((real_theta_dim,), dtype=torch.float, device=device)
    ind = 0
    for para in actor.net.parameters():
        tmp = para.numel()
        paras_torch[ind: ind + tmp] = para.data.view(-1)
        ind = ind + tmp
    paras_torch[ind:] = actor.log_std  # comment this when using the Beta policy
    func_value = np.zeros(constraint_dim + 1)
    grad = np.zeros((constraint_dim + 1, real_theta_dim))

    # Training
    observation = env.reset()
    update_index = 0
    reward_from_begining = 0
    cost_from_begining = np.zeros((args.user_num,), dtype=np.float32)
    t_from_begining = 0

    for t in range(args.num_new_data*args.num_iteration+args.T*2):
        # generate new data (sample one step of the env)
        state = observation
        action, action_max = actor.sample_action(state)
        observation, reward, done, info = env.step(action)  # reward is the objective cost in the paper.
        next_state = observation

        if t >= 2 * args.T:
            reward_from_begining = (t_from_begining / (t_from_begining + 1)) * reward_from_begining + (1 / (t_from_begining + 1)) * reward
            for kkk in range(1, 1 + args.user_num):
                cost_from_begining[kkk - 1] = (t_from_begining / (t_from_begining + 1)) * cost_from_begining[kkk-1] + (1/(t_from_begining+1)) * info.get('cost_' + str(kkk), info.get('cost', 0))
            t_from_begining += 1
        costs = np.zeros(constraint_dim + 1)
        costs[0] = reward
        for k in range(1, constraint_dim + 1):
            costs[k] = (info.get('cost_' + str(k), info.get('cost', 0)) - constr_lim[k - 1])
        buffer.store_experiences(state, action, action_max, costs, next_state, reward)

        # update the policy
        if t >= args.T*2 and ((t-args.T*2) % (num_new_data) == 0):
            alpha = 1 / ((int(update_index / 10) + 1) ** alpha_pow)
            beta = 1 / ((int(update_index / 10) + 1) ** beta_pow)

            state_buffer, action_buffer, action_max_buffer, costs_buffer, next_state_buffer = buffer.take_experiences()
            func_value_tilda = np.mean(costs_buffer, axis=0)
            func_value = (1 - alpha) * func_value + alpha * func_value_tilda

            Q_hat = np.zeros((grad_T, 1 + constraint_dim))
            for _ in range(1, grad_T + 1):
                costs_tmp = costs_buffer[_: _ + T]
                Q_hat[_ - 1] = np.sum(costs_tmp, axis=0) - grad_T * func_value
            Q_hat[:, 0] = (Q_hat[:, 0] - np.mean(Q_hat[:, 0])) / (np.std(Q_hat[:, 0]) + 1e-6)
            for _ in range(1, 1 + constraint_dim):
                Q_hat[:, _] =((Q_hat[:, _] - np.mean(Q_hat[:, _]))) / (np.std(Q_hat[:, 0]) + 1e-6)
            Q_hat_torch = torch.tensor(Q_hat, dtype=torch.float, device=device)

            # estimate the gradient
            state_batch_torch = torch.tensor(state_buffer, dtype=torch.float, device=device)
            action_batch_torch = torch.tensor(action_buffer, dtype=torch.float, device=device)
            grad_tilda_torch = torch.zeros((1 + constraint_dim, real_theta_dim), dtype=torch.float, device=device)
            for _ in range(1 + constraint_dim):
                # calculate the gradient
                actor.zero_grad()
                log_prob = actor.evaluate_action(state_batch_torch, action_batch_torch)
                actor_loss = (Q_hat_torch[:, _] * log_prob).mean()
                actor_loss.backward()
                grad_tmp = torch.zeros(real_theta_dim, dtype=torch.float, device=device)
                ind = 0
                for para in actor.net.parameters():
                    tmp = para.numel()
                    grad_tmp[ind: ind + tmp] = para.grad.view(-1)
                    ind = ind + tmp
                grad_tmp[ind:] = actor.log_std.grad  # comment this when using the Beta policy
                grad_tilda_torch[_] = grad_tmp
            grad = (1 - alpha) * grad + alpha * grad_tilda_torch.detach().cpu().numpy()

            # update the policy parameter
            paras_bar = update_policy(func_value, grad, paras_torch.detach().cpu().numpy(), tau_reward=tau_reward, tau_cost=tau_cost)
            paras_bar_torch = torch.tensor(paras_bar, dtype=torch.float, device=device)
            paras_torch = (1 - beta) * paras_torch + beta * paras_bar_torch
            ind = 0
            for para in actor.net.parameters():
                tmp = para.numel()
                para.data = paras_torch[ind: ind + tmp].view(para.shape)
                ind = ind + tmp
            actor.log_std = paras_torch[ind:]  # comment this when using the Beta policy

            update_index += 1
            print("---iteration: " + str(update_index)+"---seed: " + str(args.seed)+"---")
            print('reward_average from begin:', reward_from_begining)
            print('cost_max from begin:', np.max(cost_from_begining))
            reward_average_from_begin.append(reward_from_begining)
            cost_max_from_begin.append(np.max(cost_from_begining))


    save_reward_average_from_begin = loadmat(name + "_reward_begin_save.mat")["array"]
    save_cost_max_from_begin = loadmat(name + "_cost_begin_save.mat")["array"]
    save_reward_average_from_begin[args.seed-args.start_seed, :] = reward_average_from_begin
    save_cost_max_from_begin[args.seed-args.start_seed, :] = cost_max_from_begin
    savemat(name + "_reward_begin_save.mat", {"array": save_reward_average_from_begin})
    savemat(name + "_cost_begin_save.mat", {"array": save_cost_max_from_begin})


