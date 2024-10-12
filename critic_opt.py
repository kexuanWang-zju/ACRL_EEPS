import numpy as np
import torch.nn.functional as F
from model import Critic_net
from model import Value_net
from utils import hard_update
from utils import soft_update
import torch
import torch.nn as nn

class Critic(nn.Module):

    def __init__(self, env, user_num, num_new_data, state_dim, action_dim, constraint_dim, q, device, shape_flag):
        super(Critic, self).__init__()
        self.env = env
        self.user_num = user_num
        self.shape_flag = shape_flag
        self.constraint_dim = constraint_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.to(self.device)
        self.iter = 0
        self.num_new_data = num_new_data
        self.q = q

        # ----------------------
        self.net0 = Critic_net(self.state_dim, self.action_dim, self.device)
        self.target_net0 = Critic_net(self.state_dim, self.action_dim, self.device)
        self.critic0_optimizer = torch.optim.Adam(self.net0.parameters(), 0.01)
        # ----------------------
        self.net1 = Critic_net(self.state_dim, self.action_dim, self.device)
        self.target_net1 = Critic_net(self.state_dim, self.action_dim, self.device)
        self.critic1_optimizer = torch.optim.Adam(self.net1.parameters(), 0.01)
        # ----------------------
        self.net2 = Critic_net(self.state_dim, self.action_dim, self.device)
        self.target_net2 = Critic_net(self.state_dim, self.action_dim, self.device)
        self.critic2_optimizer = torch.optim.Adam(self.net2.parameters(), 0.01)
        # ----------------------
        self.net3 = Critic_net(self.state_dim, self.action_dim, self.device)
        self.target_net3 = Critic_net(self.state_dim, self.action_dim, self.device)
        self.critic3_optimizer = torch.optim.Adam(self.net3.parameters(), 0.01)
        # ----------------------
        self.net4 = Critic_net(self.state_dim, self.action_dim, self.device)
        self.target_net4 = Critic_net(self.state_dim, self.action_dim, self.device)
        self.critic4_optimizer = torch.optim.Adam(self.net4.parameters(), 0.01)

        hard_update(self.target_net0, self.net0)
        hard_update(self.target_net1, self.net1)
        hard_update(self.target_net2, self.net2)
        hard_update(self.target_net3, self.net3)
        hard_update(self.target_net4, self.net4)

        if self.shape_flag == 1:
            self.value_net1 = Value_net(self.state_dim, self.device)
            self.value_target_net1 = Value_net(self.state_dim, self.device)
            self.value1_optimizer = torch.optim.Adam(self.value_net1.parameters(), 0.01)
            self.value_net2 = Value_net(self.state_dim, self.device)
            self.value_target_net2 = Value_net(self.state_dim, self.device)
            self.value2_optimizer = torch.optim.Adam(self.value_net2.parameters(), 0.01)
            self.value_net3 = Value_net(self.state_dim, self.device)
            self.value_target_net3 = Value_net(self.state_dim, self.device)
            self.value3_optimizer = torch.optim.Adam(self.value_net3.parameters(), 0.01)
            self.value_net4 = Value_net(self.state_dim, self.device)
            self.value_target_net4 = Value_net(self.state_dim, self.device)
            self.value4_optimizer = torch.optim.Adam(self.value_net4.parameters(), 0.01)

            hard_update(self.value_target_net1, self.value_net1)
            hard_update(self.value_target_net2, self.value_net2)
            hard_update(self.value_target_net3, self.value_net3)
            hard_update(self.value_target_net4, self.value_net4)

    def critic_update(self, func_value, state_batch, action_batch, action_max_batch, costs_batch, next_state_batch, next_action_batch, gamma_reward, gamma_cost):

        func_value_torch = torch.tensor(func_value, dtype=torch.float, device=self.device)
        state_batch_torch = torch.tensor(state_batch, dtype=torch.float, device=self.device)
        action_batch_torch = torch.tensor(action_batch, dtype=torch.float, device=self.device)
        action_max_batch_torch = torch.tensor(action_max_batch, dtype=torch.float, device=self.device)
        next_state_batch_torch = torch.tensor(next_state_batch, dtype=torch.float, device=self.device)
        next_action_batch_torch = torch.tensor(next_action_batch, dtype=torch.float, device=self.device)
        costs_batch_torch = torch.tensor(costs_batch, dtype=torch.float, device=self.device)

        # ---------------------- optimize critic ----------------------
        next_val0 = torch.squeeze(self.target_net0.forward(next_state_batch_torch, next_action_batch_torch).detach())
        y_expected0 = costs_batch_torch[:, 0] - func_value_torch[0] + next_val0
        y_predicted0 = torch.squeeze(self.net0.forward(state_batch_torch, action_batch_torch))
        loss_critic0 = F.smooth_l1_loss(y_predicted0, y_expected0)
        self.critic0_optimizer.zero_grad()
        loss_critic0.backward()
        self.critic0_optimizer.step()
        soft_update(self.target_net0, self.net0, gamma_reward)

        # ----------------------
        next_val1 = torch.squeeze(self.target_net1.forward(next_state_batch_torch, next_action_batch_torch).detach())
        if self.shape_flag ==1:
            y_expected1 = costs_batch_torch[:, 1] - func_value_torch[1] + next_val1+ torch.squeeze(self.value_target_net1.forward(next_state_batch_torch)) - torch.squeeze(self.value_net1(state_batch_torch))
            y_predicted1 = torch.squeeze(self.net1.forward(state_batch_torch, action_batch_torch))
            loss_critic1 = F.smooth_l1_loss(y_predicted1, y_expected1)
            self.critic1_optimizer.zero_grad()
            self.value1_optimizer.zero_grad()
            loss_critic1.backward()
            self.critic1_optimizer.step()
            self.value1_optimizer.step()
            soft_update(self.target_net1, self.net1, gamma_reward)
            soft_update(self.value_target_net1, self.value_net1, gamma_reward)
        else:
            y_expected1 = costs_batch_torch[:, 1] - func_value_torch[1] + next_val1
            y_predicted1 = torch.squeeze(self.net1.forward(state_batch_torch, action_batch_torch))
            loss_critic1 = F.smooth_l1_loss(y_predicted1, y_expected1)
            self.critic1_optimizer.zero_grad()
            loss_critic1.backward()
            self.critic1_optimizer.step()
            soft_update(self.target_net1, self.net1, gamma_cost)
        # ----------------------
        next_val2 = torch.squeeze(self.target_net2.forward(next_state_batch_torch, next_action_batch_torch).detach())
        if self.shape_flag == 1:
            y_expected2 = costs_batch_torch[:, 2] - func_value_torch[2] + next_val2 + torch.squeeze(self.value_target_net2.forward(next_state_batch_torch)) - torch.squeeze(self.value_net2(state_batch_torch))
            y_predicted2 = torch.squeeze(self.net2.forward(state_batch_torch, action_batch_torch))
            loss_critic2 = F.smooth_l1_loss(y_predicted2, y_expected2)
            self.critic2_optimizer.zero_grad()
            self.value2_optimizer.zero_grad()
            loss_critic2.backward()
            self.critic2_optimizer.step()
            self.value2_optimizer.step()
            soft_update(self.target_net2, self.net2, gamma_reward)
            soft_update(self.value_target_net2, self.value_net2, gamma_reward)
        else:
            y_expected2 = costs_batch_torch[:, 2] - func_value_torch[2] + next_val2
            y_predicted2 = torch.squeeze(self.net2.forward(state_batch_torch, action_batch_torch))
            loss_critic2 = F.smooth_l1_loss(y_predicted2, y_expected2)
            self.critic2_optimizer.zero_grad()
            loss_critic2.backward()
            self.critic2_optimizer.step()
            soft_update(self.target_net2, self.net2, gamma_cost)
        # ----------------------
        next_val3 = torch.squeeze(self.target_net3.forward(next_state_batch_torch, next_action_batch_torch).detach())
        if self.shape_flag == 1:
            y_expected3 = costs_batch_torch[:, 3] - func_value_torch[3] + next_val3 + torch.squeeze(self.value_target_net3.forward(next_state_batch_torch)) - torch.squeeze(self.value_net3(state_batch_torch))
            y_predicted3 = torch.squeeze(self.net3.forward(state_batch_torch, action_batch_torch))
            loss_critic3 = F.smooth_l1_loss(y_predicted3, y_expected3)
            self.critic3_optimizer.zero_grad()
            self.value3_optimizer.zero_grad()
            loss_critic3.backward()
            self.critic3_optimizer.step()
            self.value3_optimizer.step()
            soft_update(self.target_net3, self.net3, gamma_reward)
            soft_update(self.value_target_net3, self.value_net3, gamma_reward)
        else:
            y_expected3 = costs_batch_torch[:, 3] - func_value_torch[3] + next_val3
            y_predicted3 = torch.squeeze(self.net3.forward(state_batch_torch, action_batch_torch))
            loss_critic3 = F.smooth_l1_loss(y_predicted3, y_expected3)
            self.critic3_optimizer.zero_grad()
            loss_critic3.backward()
            self.critic3_optimizer.step()
            soft_update(self.target_net3, self.net3, gamma_cost)
        # ----------------------
        next_val4 = torch.squeeze(self.target_net4.forward(next_state_batch_torch, next_action_batch_torch).detach())
        if self.shape_flag == 1:
            y_expected4 = costs_batch_torch[:, 4] - func_value_torch[4] + next_val4 + torch.squeeze(self.value_target_net4.forward(next_state_batch_torch)) - torch.squeeze(self.value_net4(state_batch_torch))
            y_predicted4 = torch.squeeze(self.net4.forward(state_batch_torch, action_batch_torch))
            loss_critic4 = F.smooth_l1_loss(y_predicted4, y_expected4)
            self.critic4_optimizer.zero_grad()
            self.value4_optimizer.zero_grad()
            loss_critic4.backward()
            self.critic4_optimizer.step()
            self.value4_optimizer.step()
            soft_update(self.target_net4, self.net4, gamma_reward)
            soft_update(self.value_target_net4, self.value_net4, gamma_reward)
        else:
            y_expected4 = costs_batch_torch[:, 4] - func_value_torch[4] + next_val4
            y_predicted4 = torch.squeeze(self.net4.forward(state_batch_torch, action_batch_torch))
            loss_critic4 = F.smooth_l1_loss(y_predicted4, y_expected4)
            self.critic4_optimizer.zero_grad()
            loss_critic4.backward()
            self.critic4_optimizer.step()
            soft_update(self.target_net4, self.net4, gamma_cost)

        if self.shape_flag == 1:
            y_value_expected1 = torch.squeeze(self.value_target_net1.forward(state_batch_torch).detach()) + torch.squeeze(self.target_net1.forward(state_batch_torch, action_max_batch_torch).detach())
            y_value_predicted1 = torch.squeeze(self.value_net1.forward(state_batch_torch))
            loss_value1 = F.smooth_l1_loss(y_value_predicted1, y_value_expected1)
            self.value1_optimizer.zero_grad()
            loss_value1.backward()
            self.value1_optimizer.step()
            soft_update(self.value_target_net1, self.value_net1, gamma_cost)

            y_value_expected2 = torch.squeeze(self.value_target_net2.forward(state_batch_torch).detach()) + torch.squeeze(self.target_net2.forward(state_batch_torch, action_max_batch_torch).detach())
            y_value_predicted2 = torch.squeeze(self.value_net2.forward(state_batch_torch))
            loss_value2 = F.smooth_l1_loss(y_value_predicted2, y_value_expected2)
            self.value2_optimizer.zero_grad()
            loss_value2.backward()
            self.value2_optimizer.step()
            soft_update(self.value_target_net2, self.value_net2, gamma_cost)

            y_value_expected3 = torch.squeeze(self.value_target_net3.forward(state_batch_torch).detach()) + torch.squeeze(self.target_net3.forward(state_batch_torch, action_max_batch_torch).detach())
            y_value_predicted3 = torch.squeeze(self.value_net3.forward(state_batch_torch))
            loss_value3 = F.smooth_l1_loss(y_value_predicted3, y_value_expected3)
            self.value3_optimizer.zero_grad()
            loss_value3.backward()
            self.value3_optimizer.step()
            soft_update(self.value_target_net3, self.value_net3, gamma_cost)

            y_value_expected4 = torch.squeeze(self.value_target_net4.forward(state_batch_torch).detach()) + torch.squeeze(self.target_net4.forward(state_batch_torch, action_max_batch_torch).detach())
            y_value_predicted4 = torch.squeeze(self.value_net4.forward(state_batch_torch))
            loss_value4 = F.smooth_l1_loss(y_value_predicted4, y_value_expected4)
            self.value4_optimizer.zero_grad()
            loss_value4.backward()
            self.value4_optimizer.step()
            soft_update(self.value_target_net4, self.value_net4, gamma_cost)

    def critic_value(self, state_batch_torch, action_batch_torch):
        Q_hat = np.matrix(np.zeros((self.num_new_data, 1 + self.constraint_dim)))
        Q_hat[:, 0] = self.target_net0.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy()
        Q_hat[:, 1] = self.target_net1.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy()
        Q_hat[:, 2] = self.target_net2.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy()
        Q_hat[:, 3] = self.target_net3.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy()
        Q_hat[:, 4] = self.target_net4.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy()
        Q_hat_torch = torch.tensor(Q_hat, dtype=torch.float, device=self.device)
        return Q_hat_torch


