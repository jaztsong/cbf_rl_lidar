# %% [markdown]
# In this lab, we will implement SAC.

# %%
import numpy as np
import gym
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import random
from collections import deque
import copy

from tqdm.std import tqdm


env = gym.make('Pendulum-v0')
n_state = int(np.prod(env.observation_space.shape))
n_action = int(np.prod(env.action_space.shape))
print("# of state", n_state)
print("# of action", n_action)

# %%

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_episode(env, policy, render=False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    states = []
    rewards = []
    actions = []
    while True:
        if render:
            env.render()

        states.append(obs)
        action = policy(obs)
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    return states, actions, rewards


# %%
class SAC:
    def __init__(self, n_state, n_action):
        self.q1_net = nn.Sequential(
            nn.Linear(n_state + n_action, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        self.act_net_base = nn.Sequential(
            nn.Linear(n_state, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
        )
        self.act_net_mu = nn.Linear(300, n_action)
        self.act_net_log_std = nn.Linear(300, n_action)
        self.q1_net.to(device)
        self.act_net_base.to(device)
        self.act_net_mu.to(device)
        self.act_net_log_std.to(device)
        self.gamma = 0.99
        self.act_lim = 2
        self.replaybuff = ReplayBuffer(50000)

        self.target_q1_net = copy.deepcopy(self.q1_net)
        self.target_q1_net.to(device)

        self.q2_net = copy.deepcopy(self.q1_net)
        self.q2_net.to(device)
        self.target_q2_net = copy.deepcopy(self.q2_net)
        self.target_q2_net.to(device)

        self.alpha = 0.2
        self.network_sync_counter = 0
        self.network_sync_freq = 5

        self.optimizer_q1 = torch.optim.Adam(self.q1_net.parameters(), lr=5e-3)
        self.optimizer_q2 = torch.optim.Adam(self.q2_net.parameters(), lr=5e-3)
        self.optimizer_act = torch.optim.Adam(
            list(self.act_net_base.parameters()) +
            list(self.act_net_mu.parameters()) +
            list(self.act_net_log_std.parameters()), lr=5e-3)

    def update(self):
        obs, act, reward, next_obs, done = self.replaybuff.sample(64)

        # if sum(reward) > 0:
        #     print("haha")
        # Update q_net
        with torch.no_grad():
            # Calculate logprob
            output = self.act_net_base(next_obs)
            mu = self.act_lim * torch.tanh(self.act_net_mu(output))
            log_std = self.act_net_log_std(output).clamp(-20, 2)
            std = torch.exp(log_std)
            dist = Normal(mu, std)
            act_ = dist.sample()
            logprob_ = dist.log_prob(act_).squeeze()
            # Calculate y
            q_input = torch.cat(
                [next_obs, act_], axis=1)
            y1 = reward + self.gamma * (1 - done) * \
                self.target_q1_net(q_input).squeeze()
            y2 = reward + self.gamma * (1 - done) * \
                self.target_q2_net(q_input).squeeze()
            y = torch.min(y1, y2) - self.alpha*logprob_

        q_input = torch.cat([obs, act], axis=1)
        self.optimizer_q1.zero_grad()
        loss_q1 = F.mse_loss(y, self.q1_net(q_input).squeeze())
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(self.q1_net.parameters(), 10)
        self.optimizer_q1.step()

        self.optimizer_q2.zero_grad()
        loss_q2 = F.mse_loss(y, self.q2_net(q_input).squeeze())
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(self.q2_net.parameters(), 10)
        self.optimizer_q2.step()

        # Update act_net
        output = self.act_net_base(obs)
        mu = self.act_lim*torch.tanh(self.act_net_mu(output))
        log_std = self.act_net_log_std(output).clamp(-20, 2)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        act_ = dist.rsample()
        logprob_ = dist.log_prob(act_).squeeze()
        q_input = torch.cat([obs, act_], axis=1)
        y1 = self.q1_net(q_input).squeeze()
        y2 = self.q2_net(q_input).squeeze()
        y = torch.min(y1, y2)
        loss_act = (self.alpha*logprob_ - y).mean()
        self.optimizer_act.zero_grad()
        loss_act.backward()
        torch.nn.utils.clip_grad_norm_(self.act_net_base.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(self.act_net_mu.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(self.act_net_log_std.parameters(), 10)
        self.optimizer_act.step()

        tau = 5e-3
        for target_param, param in zip(self.target_q1_net.parameters(), self.q1_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_q2_net.parameters(), self.q2_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau)

        return loss_q1.item(), loss_q2.item(), loss_act.item()

    def __call__(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device)
            output = self.act_net_base(obs)
            mu = self.act_lim*torch.tanh(self.act_net_mu(output))
            log_std = self.act_net_log_std(output).clamp(-20, 2)
            std = torch.exp(log_std)
            dist = Normal(mu, std)
            action = dist.sample()
            action = action.detach().cpu().numpy()
        return np.clip(action, -self.act_lim, self.act_lim)
# %%


class ReplayBuffer:
    def __init__(self, size):
        self.buff = deque(maxlen=size)

    def add(self, obs, act, reward, next_obs, done):
        self.buff.append([obs, act, reward, next_obs, done])

    def sample(self, sample_size):
        if(len(self.buff) < sample_size):
            sample_size = len(self.buff)

        sample = random.sample(self.buff, sample_size)
        obs = torch.FloatTensor([exp[0] for exp in sample]).to(device)
        act = torch.FloatTensor([exp[1] for exp in sample]).to(device)
        reward = torch.FloatTensor([exp[2] for exp in sample]).to(device)
        next_obs = torch.FloatTensor([exp[3] for exp in sample]).to(device)
        done = torch.FloatTensor([exp[4] for exp in sample]).to(device)
        return obs, act, reward, next_obs, done

    def __len__(self):
        return len(self.buff)


# %%
loss_q_list, loss_act_list, reward_list = [], [], []
agent = SAC(n_state, n_action)
update_freq = 5
n_step = 0
loss_q1, loss_act = 0, 0

for i in tqdm(range(500)):
    obs, rew = env.reset(), 0
    while True:
        act = agent(obs)
        next_obs, reward, done, _ = env.step(act)
        rew += reward
        n_step += 1

        agent.replaybuff.add(obs, act, reward, next_obs, done)
        obs = next_obs

        if len(agent.replaybuff) > 1e3 and n_step % update_freq == 0:
            loss_q1, loss_q2, loss_act = agent.update()
        if done:
            break

    if i > 0 and i % 50 == 0:
        run_episode(env, agent, False)[2]
        print("itr:({:>5d}) loss_q:{:>3.4f} loss_act:{:>3.4f} reward:{:>3.1f}".format(
            i, np.mean(loss_q_list[-50:]),
            np.mean(loss_act_list[-50:]),
            np.mean(reward_list[-50:])))

    loss_q_list.append(loss_q1), loss_act_list.append(
        loss_act), reward_list.append(rew)

# %%
scores = [sum(run_episode(env, agent, False)[2]) for _ in range(100)]
print("Final score:", np.mean(scores))

import pandas as pd
df = pd.DataFrame({'loss_q': loss_q_list,
                   'loss_act': loss_act_list,
                   'reward': reward_list})
df.to_csv("./ClassMaterials/Lab_04_DDPG/data/sac-pen-tuned.csv",
          index=False, header=True)
