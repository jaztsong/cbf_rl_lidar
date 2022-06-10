from turtle import color
import numpy as np
from qpsolvers import solve_qp


def _h(state, type='handcraft'):
    if type == 'handcraft':
        safe_d = 1.2
        mean_m = np.ones_like(state) * 1.0 / state.shape[0]
        return 1.0 / safe_d - np.dot(mean_m, state)


def control_barrier(obs, u_rl, f, g):

    # objective function, same for all CBF-QP
    P = np.diag([0.2, 1.])
    q = -u_rl.astype(np.double)
    lb = np.array([-0.9, 0])
    ub = np.array([0.9, 1])

    mean_m = np.ones((1, obs.shape[0])) * 1.0 / obs.shape[0]
    safe_d = 0.8
    gamma = 0.2
    G = np.dot(mean_m, g)

    h = 1.0 / safe_d - np.dot(mean_m, f) + (gamma - 1) * (1.0 / safe_d - np.dot(mean_m, obs))
    h = np.array(h)

    u_filtered = None
    try:
        u_filtered = solve_qp(P, q, G, h,
                              lb=lb,
                              ub=ub,
                              solver="cvxopt")
        print("CBF tweaking: h(x) = {} new_h(x) = {}".format(
            (1.0 / safe_d - np.dot(mean_m, obs)),
            (1.0 / safe_d - np.dot(mean_m, f + np.dot(g, u_filtered)))))
        if not np.isclose(np.mean(u_filtered - u_rl), 0, atol=1e-3):
            print("(cbf): action diff: {}".format(u_filtered - u_rl))
    except:
        print("CBF failed: h(x) = {}".format((1.0 / safe_d - np.dot(mean_m, obs))))
    return u_filtered if u_filtered is not None else u_rl
