from turtle import color
import numpy as np
from qpsolvers import solve_qp
import torch
import torch.nn as nn


class CBF(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        M = np.ones((1, obs_dim)) * -1.0 / obs_dim
        self.P = nn.Parameter(torch.FloatTensor(M), requires_grad=True)
        self.Q = nn.Parameter(torch.FloatTensor([1.0 / 0.8]), requires_grad=True)

    def h(self, obs):
        # For training purpose
        return torch.matmul(obs[:, None, :], self.P.t()).squeeze() + self.Q

    def control_barrier(self, obs, u_rl, f, g):
        # objective function, same for all CBF-QP
        P = np.diag([0.1, 1.0])
        q = -u_rl.astype(np.double) @ P
        lb = np.array([-0.9, 0])
        ub = np.array([0.9, 2])

        P_h = self.P.detach().cpu().numpy().astype(np.double)
        Q_h = self.Q.detach().cpu().numpy().astype(np.double)

        gamma = 0.5
        G = -1 * np.dot(P_h, g)

        h = np.dot(P_h, f) + Q_h + (gamma - 1) * (np.dot(P_h, obs) + Q_h)
        h = np.array(h)

        u_filtered = None
        try:
            u_filtered = solve_qp(P, q, G, h, lb=lb, ub=ub, solver="cvxopt")
            if not np.isclose(np.mean(u_filtered - u_rl), 0, atol=1e-3):
                print(
                    "CBF tweaking: h(x) = {} new_h(x) = {}".format(
                        (np.dot(P_h, obs) + Q_h), (np.dot(P_h, f + np.dot(g, u_filtered)) + Q_h)
                    )
                )
                print("(cbf): action diff: {}".format(u_filtered - u_rl))
                # print('.', end='')
        except:
            print("CBF failed: h(x) = {}".format((np.dot(P_h, obs) + Q_h)))
        return u_filtered if u_filtered is not None else u_rl
