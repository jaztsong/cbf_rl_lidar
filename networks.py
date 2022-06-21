import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def conv1d_nets(
    input_size, output_sizes, kernel_sizes, strides, out_linear, activation=nn.ReLU
):
    assert len(output_sizes) == len(kernel_sizes), "input sizes do not match"
    layers = []
    conv_outsize = input_size
    in_ = 1
    for i in range(len(output_sizes)):
        layers += [
            nn.Conv1d(
                in_, output_sizes[i], kernel_size=kernel_sizes[i], stride=strides[i]
            ),
            activation(),
        ]
        conv_outsize = (conv_outsize - (kernel_sizes[i] - 1) - 1) // strides[i] + 1
        in_ = output_sizes[i]

    layers += [
        nn.Flatten(start_dim=1),
        nn.Linear(conv_outsize * output_sizes[-1], out_linear),
        activation(),
    ]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianActor(nn.Module):
    def __init__(
        self, obs_dim, act_dim, hidden_sizes, activation, act_limit
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.net_conv = conv1d_nets(
            obs_dim, [32, 32], [20, 10], [5, 3], 128
        )
        self.net_mlp = mlp(
            [128] + list(hidden_sizes), activation
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        # net_out = self.net(obs)None
        obs = obs.reshape((-1, 1, self.obs_dim))
        net_conv_out = self.net_conv(obs)
        net_out = self.net_mlp(net_conv_out).squeeze()
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        # print("action before shifting: {}", pi_action)
        pi_action = (
            pi_action * (self.act_limit[1] - self.act_limit[0]) / 2.0
            + (self.act_limit[1] + self.act_limit[0]) / 2.0
        )
        # print("action after shifting: {}", pi_action)

        return pi_action, logp_pi


class QFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # layers = []
        # layers += [nn.Conv1d(1, 32, kernel_size=20, stride=5), activation()]
        # conv_out1size = (obs_dim - (20 - 1) - 1) // 5 + 1
        # layers += [nn.Conv1d(32, 32, kernel_size=10, stride=3), activation()]
        # conv_out2size = (conv_out1size - (10 - 1) - 1) // 3 + 1
        # layers += [nn.Flatten(start_dim=1),
        #            nn.Linear(conv_out2size * 32, 128), activation()]
        # self.q_conv = nn.Sequential(*layers)
        self.q_conv = conv1d_nets(
            obs_dim, [32, 32], [20, 10], [5, 3], 128
        )
        self.q_mlp = mlp(
            [128 + act_dim] + list(hidden_sizes) + [1], activation
        )

    def forward(self, obs, act):
        q_conv = self.q_conv(obs[:, None, :])
        q = self.q_mlp(torch.cat([q_conv, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = nn.Parameter(
            torch.FloatTensor([action_space.low, action_space.high]),
            requires_grad=False,
        )

        # build policy and value functions
        self.pi = SquashedGaussianActor(
            obs_dim, act_dim, hidden_sizes, activation, act_limit
        )
        self.q1 = QFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = QFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.qc = QFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()


class NN_Dynamics(nn.Module):
    def __init__(
        self, obs_dim, act_dim, f_hidden_sizes, g_hidden_sizes, activation=nn.ReLU
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # self.f_conv = convlayer(obs_dim, [64], [40], [10], 128)
        # self.f_mlp = mlp([128] + list(f_hidden_sizes) + [obs_dim], activation)
        # self.g_conv = convlayer(obs_dim, [64], [40], [10], 128)
        # self.g_mlp = mlp([128] + list(g_hidden_sizes) + [obs_dim * act_dim], activation)
        # self.f = mlp([obs_dim] + list(f_hidden_sizes) + [obs_dim], activation)
        self.g = mlp([obs_dim] + list(g_hidden_sizes) + [obs_dim * act_dim], activation)

    def forward(self, obs, act):
        # obs = obs[:, None, :]
        # f = self.f_mlp(self.f_conv(obs))
        # g = self.g_mlp(self.g_conv(obs))
        f = obs
        g = self.g(obs)
        g = g.reshape((-1, self.obs_dim, self.act_dim))
        return f + (g @ act[:, :, None]).squeeze()

    def get_dynamics(self, obs, device):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        # obs = obs[None, None, :]
        # f = self.f_mlp(self.f_conv(obs))
        # g = self.g_mlp(self.g_conv(obs))
        # g = g.reshape((-1, self.obs_dim, self.act_dim))
        f = obs  # + self.f(obs)
        g = self.g(obs).reshape((-1, self.obs_dim, self.act_dim))
        return f.detach().cpu().numpy(), g.squeeze().detach().cpu().numpy()
