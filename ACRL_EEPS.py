from environment import Environment_XR
from critic_opt import Critic
from utils import update_policy
from model import GaussianPolicy
from buffer import DataStorage
import numpy as np
import torch
from scipy.io import loadmat
from scipy.io import savemat
save_folder = 'net_save_QGPS'


def ACRL_EEPS(args, name, cuda_index):

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
	gamma_pow_reward = args.gamma_pow_reward
	gamma_pow_cost = args.gamma_pow_cost
	tau_reward = args.tau_reward
	tau_cost = args.tau_cost
	Q_update_time = args.Q_update_time
	window = args.window
	Nt, UE_num = 8, args.user_num  # The number of antennas and users.
	buffer_max = 1000
	delay_max = np.ones((args.user_num,), dtype=np.int32)*4
	state_dim = 2 * UE_num * Nt + 2 * np.sum(delay_max)  # 2*4*8+2*4=72
	action_dim = UE_num + 1

	env = Environment_XR(seed=args.seed, Nt=Nt, UE_num=UE_num, buffer_max=buffer_max, delay_max=delay_max)
	constraint_dim = UE_num
	constr_lim = np.ones((args.user_num,), dtype=np.float32)*0.1

	if args.TL_flag == 0:
		actor = GaussianPolicy(state_dim, action_dim, device, num_new_data)
		actor.to(device)
		theta_dim = 0
		for para in actor.net.parameters():
			theta_dim += para.numel()
		real_theta_dim = theta_dim + action_dim  # the dimension of the policy parameter.
		paras_torch = torch.zeros((real_theta_dim,), dtype=torch.float, device=device)

		ind = 0
		for para in actor.net.parameters():
			tmp = para.numel()
			paras_torch[ind: ind + tmp] = para.data.view(-1)
			ind = ind + tmp
		paras_torch[ind:] = actor.log_std
	else:
		actor_list = []
		reuse_pro = torch.ones((args.num_old_policy+1,), dtype=torch.float, device=device)
		reuse_pro = reuse_pro / torch.sum(reuse_pro)
		for index in range(args.num_old_policy+1):
			actor_list.append(GaussianPolicy(state_dim, action_dim, device, num_new_data).to(device))
		for index in range(args.num_old_policy):
			subparas_torch = torch.tensor(loadmat("policy" + str(index))["array"][0, :], dtype=torch.float, device=device)
			ind = 0
			for para in actor_list[index].net.parameters():
				tmp = para.numel()
				para.data = subparas_torch[ind: ind + tmp].view(para.shape)
				ind = ind + tmp
			actor_list[index].log_std = subparas_torch[ind:]

		theta_dim = 0
		for para in actor_list[args.num_old_policy].net.parameters():
			theta_dim += para.numel()
		real_theta_dim = theta_dim + action_dim + args.num_old_policy+1 # the dimension of the policy parameter.
		paras_torch = torch.zeros((real_theta_dim,), dtype=torch.float, device=device)
		ind = 0
		for para in actor_list[args.num_old_policy].net.parameters():
			tmp = para.numel()
			paras_torch[ind: ind + tmp] = para.data.view(-1)
			ind = ind + tmp
		paras_torch[ind:(real_theta_dim-args.num_old_policy-1)] = actor_list[args.num_old_policy].log_std
		paras_torch[(real_theta_dim - args.num_old_policy-1):] = reuse_pro

	buffer = DataStorage(T, num_new_data, state_dim, action_dim, constraint_dim, window, Q_update_time)
	critic = Critic(env, args.user_num, grad_T, state_dim, action_dim, constraint_dim, Q_update_time, device, args.shape_flag)
	critic.to(device)
	########################################################
	func_value = np.zeros(constraint_dim + 1)
	grad = np.zeros((constraint_dim + 1, real_theta_dim))

	observation = env.reset()
	update_index = 0
	Q_update_index = 0
	reward_from_begining = 0
	cost_from_begining = np.zeros((args.user_num,), dtype=np.float32)
	t_from_begining = 0

	for t in range(args.num_new_data*args.num_iteration+args.T*2):
		state = observation
		if args.TL_flag == 1:
			reuse_pro = reuse_pro / torch.sum(reuse_pro)
			actor = np.random.choice(actor_list, p=reuse_pro.detach().cpu().numpy())
		action, action_max = actor.sample_action(state)
		observation, reward, done, info = env.step(action)
		next_state = observation
		if t >= 2*args.T:
			reward_from_begining = (t_from_begining / (t_from_begining + 1)) * reward_from_begining + (1 / (t_from_begining + 1)) * reward
			for kkk in range(1, 1 + args.user_num):
				cost_from_begining[kkk-1] = (t_from_begining / (t_from_begining + 1)) * cost_from_begining[kkk-1] + (1 / (t_from_begining + 1)) * info.get('cost_' + str(kkk), info.get('cost', 0))
			t_from_begining += 1
		costs = np.zeros(constraint_dim + 1)
		costs[0] = reward
		for k in range(1, constraint_dim + 1):
			costs[k] = (info.get('cost_' + str(k), info.get('cost', 0)) - constr_lim[k - 1])
		buffer.store_experiences(state, action, action_max, costs, next_state, reward)

		# update the policy
		if t >= args.T*2 and ((t-args.T*2) % (num_new_data/Q_update_time) == 0):
			Q_update_index += 1
			alpha = 1 / ((int(update_index/10) + 1) ** alpha_pow)
			beta = 1 / ((int(update_index/10) + 1) ** beta_pow)
			if Q_update_index == Q_update_time:
				gamma_reward = 1 / ((int(update_index/10) + 1) ** gamma_pow_reward)
				gamma_cost = 1 / ((int(update_index/10) + 1) ** gamma_pow_cost)
			else:
				gamma_reward = 0
				gamma_cost = 0

			state_buffer, action_buffer, action_max_buffer, costs_buffer, next_state_buffer = buffer.take_experiences()

			state_batch = state_buffer[(2 * T - grad_T):2 * T]
			action_batch = action_buffer[(2 * T - grad_T):2 * T]
			action_max_batch = action_max_buffer[(2 * T - grad_T):2 * T]
			costs_batch = costs_buffer[(2 * T - grad_T):2 * T]
			next_state_batch = next_state_buffer[(2 * T - grad_T):2 * T]
			next_action_batch = np.zeros((grad_T, action_dim))
			for jjj in range(grad_T):
				next_action_batch[jjj, :],  temp_action = actor.sample_action(next_state_buffer[(2 * T - grad_T) + jjj, :])

			state_batch_torch = torch.tensor(state_batch, dtype=torch.float, device=device)
			action_batch_torch = torch.tensor(action_batch, dtype=torch.float, device=device)

			# estimate the Q-value
			critic.critic_update(func_value, state_batch, action_batch, action_max_batch, costs_batch, next_state_batch, next_action_batch, gamma_reward, gamma_cost)

			if (Q_update_index == Q_update_time):
				# estimate the gradient
				Q_update_index = 0
				Q_hat_torch = critic.critic_value(state_batch_torch, action_batch_torch)
				Q_hat = Q_hat_torch.detach().cpu().numpy()
				Q_hat[:, 0] = (Q_hat[:, 0] - np.mean(Q_hat[:, 0])) / (np.std(Q_hat[:, 0]) + 1e-6)
				for _ in range(1, 1 + constraint_dim):
					Q_hat[:, _] = (Q_hat[:, _] - np.mean(Q_hat[:, _])) / (np.std(Q_hat[:, _]) + 1e-6)
				Q_hat_torch = torch.tensor(Q_hat, dtype=torch.float, device=device)
				grad_tilda_torch = torch.zeros((1 + constraint_dim, real_theta_dim), dtype=torch.float, device=device)

				func_value_tilda = np.mean(costs_buffer, axis=0)
				func_value = (1 - alpha) * func_value + alpha * func_value_tilda

				if args.TL_flag == 0:
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
					actor.log_std = paras_torch[ind:]
				else:
					for _ in range(1 + constraint_dim):
						# calculate the gradient
						grad_tmp = torch.zeros(real_theta_dim, dtype=torch.float, device=device)

						prob_list = []
						for index in range(args.num_old_policy+1):
							prob_list.append(reuse_pro[index]*actor_list[index].evaluate_action_TL(state_batch_torch, action_batch_torch))
							if index == 0:
								prob_average_over_policy = reuse_pro[index] * actor_list[index].evaluate_action_TL(state_batch_torch, action_batch_torch)
							else:
								prob_average_over_policy += reuse_pro[index]*actor_list[index].evaluate_action_TL(state_batch_torch, action_batch_torch)
						actor_list[args.num_old_policy].zero_grad()
						actor_loss = (Q_hat_torch[:, _] * prob_list[args.num_old_policy]/prob_average_over_policy.detach()).mean()
						actor_loss.backward()

						ind = 0
						for para in actor_list[args.num_old_policy].net.parameters():
							tmp = para.numel()
							grad_tmp[ind: ind + tmp] = para.grad.view(-1)
							ind = ind + tmp
						grad_tmp[ind:(real_theta_dim-args.num_old_policy-1)] = actor_list[args.num_old_policy].log_std.grad
						for index in range(args.num_old_policy+1):
							grad_tmp[(real_theta_dim-args.num_old_policy-1)+index] = (Q_hat_torch[:, _] * prob_list[index]/prob_average_over_policy).mean()/reuse_pro[index]
						grad_tilda_torch[_] = grad_tmp
					grad = (1 - alpha) * grad + alpha * grad_tilda_torch.detach().cpu().numpy()

					# update the policy parameter
					paras_bar = update_policy(func_value, grad, paras_torch.detach().cpu().numpy(), tau_reward=tau_reward,tau_cost=tau_cost)
					paras_bar_torch = torch.tensor(paras_bar, dtype=torch.float, device=device)
					paras_torch = (1 - beta) * paras_torch + beta * paras_bar_torch
					ind = 0
					for para in actor_list[args.num_old_policy].net.parameters():
						tmp = para.numel()
						para.data = paras_torch[ind: ind + tmp].view(para.shape)
						ind = ind + tmp
					#actor_list[args.num_old_policy].log_std = paras_torch[ind:(real_theta_dim-args.num_old_policy-1)]
					reuse_pro = paras_torch[(real_theta_dim-args.num_old_policy-1):]
					reuse_pro[reuse_pro <= 0.05] = 0.05
					reuse_pro = reuse_pro / torch.sum(reuse_pro)

				update_index += 1
				print("---iteration: " + str(update_index)+"---seed: " + str(args.seed)+"---")
				print('reward_average from begin:', reward_from_begining)
				print('cost_max from begin:', np.max(cost_from_begining))
				if args.TL_flag == 1:
					print("probabilities: ", reuse_pro)
				reward_average_from_begin.append(reward_from_begining)
				cost_max_from_begin.append(np.max(cost_from_begining))

	if args.old_policy_flag == 1 and args.TL_flag == 0:
		savemat("policy"+str(args.seed), {"array": paras_torch.detach().cpu().numpy()})

	save_reward_average_from_begin = loadmat(name + "_reward_begin_save.mat")["array"]
	save_cost_max_from_begin = loadmat(name + "_cost_begin_save.mat")["array"]
	save_reward_average_from_begin[args.seed-args.start_seed, :] = reward_average_from_begin
	save_cost_max_from_begin[args.seed-args.start_seed, :] = cost_max_from_begin
	savemat(name + "_reward_begin_save.mat", {"array": save_reward_average_from_begin})
	savemat(name + "_cost_begin_save.mat", {"array": save_cost_max_from_begin})

