import numpy as np
import copy
import math


class Environment_XR(object):
    """The environment class of the MIMO power allocation.
    For conciseness, we adopt the 'delay' Q/mu in the simulation."""

    def __init__(self, seed, Nt, UE_num, buffer_max, delay_max):
        super(Environment_XR, self).__init__()
        self.buffer_max = buffer_max
        self.delay_max = delay_max
        self.poisson_lambda = np.random.uniform(10, 15, UE_num)
        self.arrive_probability = np.random.uniform(0.2, 0.4, UE_num)

        self.seed = seed
        self.seed_step = seed
        self.Nt = Nt
        self.UE_num = UE_num
        self.user_per_group = 1
        self.group_num = int(UE_num / self.user_per_group)
        self.B_state = np.zeros((2, np.sum(self.delay_max)), np.float64)
        self.state_dim = 2 * UE_num * Nt + 2 * np.sum(self.delay_max)
        self.action_dim = UE_num + 1
        self.Np = 4

        np.random.seed(seed)
        PathGain_dB = np.random.uniform(-10, 10, self.group_num)
        self.PathGain = 10 ** (PathGain_dB / 10)
        alpha_power_group = np.zeros((self.group_num, self.Np))
        for group in range(self.group_num):
            tmp = np.random.exponential(scale=1, size=self.Np)
            alpha_power_group[group] = (tmp * self.PathGain[group]) / np.sum(tmp)
        self.alpha_power = np.tile(alpha_power_group, (self.user_per_group, 1))

        array_reponse_group = np.zeros((self.group_num * self.Nt, self.Np)) + 1j * np.zeros((self.group_num * self.Nt, self.Np))
        for group in range(self.group_num):
            A_tmp = np.zeros((self.Nt, self.Np)) + 1j * np.zeros((self.Nt, self.Np))
            for i in range(self.Np):
                AoD = self.laprnd(mu=0, angular_spread=5)
                A_tmp[:, i] = np.exp(1j * np.pi * np.sin(AoD) * np.arange(0, self.Nt))
            array_reponse_group[group * self.Nt: (group+1) * self.Nt] = A_tmp
        self.array_response = np.tile(array_reponse_group, (self.user_per_group, 1))

        self.H_g = np.zeros((self.group_num, Nt)) + 1j * np.zeros((self.group_num, Nt))
        self.H = np.zeros((UE_num, Nt)) + 1j * np.zeros((UE_num, Nt))
        self.package_generate = np.zeros(self.UE_num)
        self.state = np.zeros(self.state_dim)
        self.noise_power = 1e-6

    def reset(self):
        # Reset the environment and return the initial state.
        np.random.seed(self.seed)
        for g in range(self.group_num):
            alpha_power_g = self.alpha_power[g]
            A_g = self.array_response[g * self.Nt: (g + 1) * self.Nt]
            alpha_g = np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np) + 1j * np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np)
            h_g_temp = np.dot(A_g, alpha_g)
            self.H_g[g] = np.conjugate(h_g_temp)
        self.H = np.repeat(self.H_g, self.user_per_group, axis=0)
        self.B_state = np.zeros((2, np.sum(self.delay_max)))
        h_real = np.real(self.H)
        h_real = h_real.reshape(-1)
        h_imag = np.imag(self.H)
        h_imag = h_imag.reshape(-1)
        self.state = np.hstack((h_real, h_imag, self.B_state[0, :], self.B_state[1, :]))
        return self.state


    def step(self, action):
        np.random.seed(self.seed_step)
        self.seed_step += 1

        costs = np.zeros(self.UE_num)
        action = action.reshape(-1)
        action[action <= 0] = 1e-6
        power = action[0: self.UE_num]
        reg_factor = action[self.UE_num]
        reward = np.sum(power)

        try:
            V = np.dot(self.H.conjugate().T, np.linalg.inv(np.dot(self.H, self.H.conjugate().T) + reg_factor * np.eye(self.UE_num)))
        except:
            V = np.dot(self.H.conjugate().T, np.linalg.pinv(np.dot(self.H,self.H.conjugate().T) + reg_factor * np.eye(self.UE_num)))
        norm_vector = np.zeros(self.UE_num)
        for k in range(self.UE_num):
            norm_vector[k] = 1 / (np.linalg.norm(V[:, k]) + 1e-7)
        V_tilda = np.dot(V, np.diag(norm_vector))

        hv_tilda = np.dot(self.H, V_tilda)
        r_d = np.zeros(self.UE_num)
        for k in range(self.UE_num):
            module_squ = np.abs(hv_tilda[k]) ** 2
            numerator = power[k] * module_squ[k]
            module_squ[k] = 0
            dominator = np.sum(power * module_squ) + self.noise_power
            r_d[k] = np.log2(1 + numerator / dominator)
        a=1

        for user in range(self.UE_num):
            tail_index = 0
            for jj in range(user):
                tail_index += self.delay_max[jj]
            head_index = -1
            for jj in range(user + 1):
                head_index += self.delay_max[jj]
            if np.sum(self.B_state[0, tail_index:(head_index + 1)]) == 0:
                pass
            else:
                if r_d[user] >= np.sum(self.B_state[0, tail_index:(head_index + 1)]):
                    self.B_state[0, tail_index:(head_index + 1)] = np.zeros((1, self.delay_max[user]), np.float64)
                    str1 = str(user)
                else:
                    for package_index in range(self.delay_max[user]):
                        index = tail_index + package_index
                        front = np.sum(self.B_state[0, (index + 1):(head_index + 1)])
                        if (r_d[user] - front) >= 0 and (r_d[user] - front - self.B_state[0, index]) < 0:
                            if index == head_index:  # timeout
                                self.B_state[0, index] = 0
                                self.B_state[1, index] = 0
                                costs[user] += 1
                            else:
                                self.B_state[0, index] = self.B_state[0, index] - (r_d[user] - front)
                                self.B_state[1, index] = self.B_state[1, index] + (r_d[user] - front)
                                self.B_state[0, (index + 1):(head_index + 1)] = np.zeros((1, head_index - index), np.float64)
                                self.B_state[0, (index + 1):(head_index + 1)] = np.zeros((1, head_index - index), np.float64)


        B_state_temp = copy.deepcopy(self.B_state)
        self.B_state = np.zeros((2, np.sum(self.delay_max)), np.float64)
        for user in range(self.UE_num):
            head_index = -1
            for jj in range(user + 1):
                head_index += self.delay_max[jj]
            self.B_state[:, (head_index - self.delay_max[user] + 2):(head_index + 1)] = B_state_temp[:, (head_index-self.delay_max[user] + 1):head_index]

        for user in range(self.UE_num):
            self.package_generate[user] = np.random.poisson(lam=self.poisson_lambda[user] * 2, size=1) / 2
            sample = np.random.uniform(0, 1, 1)
            if sample > self.arrive_probability[user]:
                self.package_generate[user] = 0

        # overflow
        for user in range(self.UE_num):
            tail_index = 0
            for jj in range(user):
                tail_index += self.delay_max[jj]
            if (self.package_generate[user] + np.sum(
                    self.B_state[0, tail_index:(tail_index + self.delay_max[user] - 1)])) <= self.buffer_max:
                self.B_state[0, tail_index] = self.package_generate[user]
            else:  # overflow
                self.B_state[0, tail_index] = 0
                costs[user] += 1

        for g in range(self.group_num):
            alpha_power_g = self.alpha_power[g]
            A_g = self.array_response[g * self.Nt: (g + 1) * self.Nt]
            alpha_g = np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np) + \
                      1j * np.sqrt(alpha_power_g / 2) * np.random.randn(self.Np)
            self.H_g[g] = np.dot(A_g, alpha_g)
            h_g_temp = np.dot(A_g, alpha_g)
            self.H_g[g] = np.conjugate(h_g_temp)
        self.H = np.repeat(self.H_g, self.user_per_group, axis=0)
        h_real = np.real(self.H)
        h_real = h_real.reshape(-1)
        h_imag = np.imag(self.H)
        h_imag = h_imag.reshape(-1)
        self.state = np.hstack((h_real, h_imag, self.B_state[0], self.B_state[1]))
        d = False

        info = {'cost_' + str(i): costs[i - 1] for i in range(1, self.UE_num + 1)}
        info['cost'] = np.sum(costs)

        tail = 0
        temp_costs = np.zeros(self.UE_num)
        for jjj in range(self.UE_num):
            temp_costs[jjj] = np.sum(self.B_state[0, tail:(tail + self.delay_max[jjj])])
            tail += self.delay_max[jjj]
        info_soft = {'cost_' + str(i): temp_costs[i - 1] for i in range(1, self.UE_num + 1)}
        info_soft['cost'] = np.sum(temp_costs)

        return self.state, reward, d, info

    def laprnd(self, mu, angular_spread):
        b = angular_spread / np.sqrt(2)
        a = np.random.rand(1) - 0.5
        x = mu - b * np.sign(a) * np.log(1 - 2 * np.abs(a))
        return x




