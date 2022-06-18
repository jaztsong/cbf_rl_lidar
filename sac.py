from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import networks as core
from utils.logx import EpochLogger
from tqdm.std import tqdm
from envs.rccar_gazebo_env import RccarGazeboEnv
from matplotlib import pyplot as plt
from cbf import CBF


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.cbf_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, cost):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def label_cbf(self, unsafe_step=3):
        assert self.done_buf[self.ptr - 1] == 1, "Not a good time to label"
        ptr = self.ptr - 2
        for j in range(unsafe_step * 2):
            if not self.done_buf[ptr - j]:
                if j < unsafe_step:
                    self.cbf_buf[ptr - j] = -1
                else:
                    self.cbf_buf[ptr - j] = 1

    def sample_batch(self, batch_size=32, device="cpu"):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            cost=self.cost_buf[idxs],
            cbf=self.cbf_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}

    def sample_latest(self, size=1e3, device="cpu"):
        idxs = range(int(max(self.ptr - size, 0)), self.ptr)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            cost=self.cost_buf[idxs],
            cbf=self.cbf_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}


def sac(
    env,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    dynamics_kwargs=dict(),
    seed=0,
    steps_per_epoch=10000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.9,
    lr=1e-3,
    alpha=0.2,
    beta=0.1,
    if_cbf=False,
    if_fixed_h=False,
    batch_size=64,
    start_steps=10,
    update_after=20,
    update_every=50,
    num_test_episodes=0,
    max_ep_len=1000,
    logger_kwargs=dict(),
    save_freq=1,
):
    """
    Soft Actor-Critic (SAC) from Spinning Up
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger = EpochLogger(**logger_kwargs)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # env, test_env = env_fn(), env_fn()
    test_env = env

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Create neural nets for fitting dynamics
    dynamics = core.NN_Dynamics(obs_dim, act_dim, **dynamics_kwargs)

    # Create cbf
    cbf = CBF(obs_dim)

    ac.to(device=device)
    ac_targ.to(device=device)
    dynamics.to(device=device)
    cbf.to(device=device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = list(ac.q1.parameters()) + list(ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log("\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" % var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d, c = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
            data["cost"],
        )

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            qc_pi_targ = ac.qc(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(), Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC QC-losses
    def compute_loss_qc(data):
        o, a, r, o2, d, c = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
            data["cost"],
        )

        qc = ac.qc(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, _ = ac.pi(o2)

            # Target QC-values
            qc_pi_targ = ac.qc(o2, a2)
            cost_backup = c + gamma * (1 - d) * (qc_pi_targ)

        # MSE loss against Bellman backup
        loss_qc = ((qc - cost_backup) ** 2).mean()

        # Useful info for logging
        qc_info = dict(QCVals=qc.detach().cpu().numpy())

        return loss_qc, qc_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data["obs"]
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        qc_pi = ac.qc(o, pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi + beta * qc_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)
    qc_optimizer = Adam(ac.qc.parameters(), lr=lr)
    # Set up optimizers for cbf related paramaters
    dynamics_optimizer = Adam(dynamics.parameters(), lr=lr)
    cbf_optimizer = Adam(cbf.parameters(), lr=1e-4)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        qc_optimizer.zero_grad()
        loss_qc, qc_info = compute_loss_qc(data)
        loss_qc.backward()
        qc_optimizer.step()
        # Record things
        logger.store(LossQ=loss_q.item(), **q_info, LossQC=loss_qc.item(), **qc_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params + list(ac.qc.parameters()):
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params + list(ac.qc.parameters()):
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32, device=device), deterministic)

    def update_dynamics(data):
        o, a, r, o2, d = data["obs"], data["act"], data["rew"], data["obs2"], data["done"]

        index_all = [j for j in range(len(o))]
        for i in range(0, len(index_all), batch_size):
            index = index_all[i : i + batch_size]
            # MSE loss
            o2_ = dynamics(o[index], a[index])
            loss_dynamics = ((o2_ - o2[index]) ** 2).mean()
            # g = dynamics.get_gx(o)
            # _, s, _ = torch.linalg.svd(g)
            # loss_dynamics += -1e-5 * (torch.abs(s[:, 0] * s[:, 1])).mean()

            dynamics_optimizer.zero_grad()
            loss_dynamics.backward()
            dynamics_optimizer.step()

            # Record things
            logger.store(LossDynamics=loss_dynamics.item())

    def update_cbf(data):
        o, h = data["obs"], data["cbf"]

        index_all = [j for j in range(len(o)) if h[j] != 0]
        for i in range(0, len(index_all), batch_size):
            index = index_all[i : i + batch_size]
            # MSE loss
            h_ = cbf.h(o[index])
            loss_cbf = ((h_ - h[index]) ** 2).mean()

            cbf_optimizer.zero_grad()
            loss_cbf.backward()
            cbf_optimizer.step()

            # Record things
            logger.store(LossCBF=loss_cbf.item())

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    n_coll = 0

    updated, finished = False, False
    # Main loop: collect experience in env and update/log each epoch
    for t in tqdm(range(total_steps)):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a = get_action(o)
            if updated and if_cbf:
                f, g = dynamics.get_dynamics(o, device=device)
                a = cbf.control_barrier(o, a, f, g)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        c = info["cost"]
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d, c)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0
            finished = True
            n_coll += d
            # update dynamics
            if if_cbf:
                if not if_fixed_h:
                    replay_buffer.label_cbf(unsafe_step=3)
                data = replay_buffer.sample_latest(size=2e3, device=device)
                for _ in range(5):
                    update_dynamics(data=data)
                    if not if_fixed_h:
                        update_cbf(data=data)

                # test_data = replay_buffer.sample_batch(1, device=device)
                # plt.plot(test_data['obs2'][0].cpu().numpy(), 'r-', label='ground_truth')
                # plt.plot(dynamics(test_data['obs'], test_data['act']).detach().cpu().numpy()[0], 'g--', label='predicted')
                # plt.legend()
                # plt.show()

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size, device=device)
                update(data=batch)

            updated = True

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0 and updated and finished:
            epoch = (t + 1) // steps_per_epoch
            logger.store(Coll=n_coll)
            n_coll = 0

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({"env": env}, None)

            # Test the performance of the deterministic version of the agent.
            # test_agent()

            # Log info about epoch
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("EpRet", with_min_and_max=True)
            logger.log_tabular("Coll", average_only=True)
            logger.log_tabular("TotalEnvInteracts", t)
            logger.log_tabular("LossQ", average_only=True)
            logger.log_tabular("LossQC", average_only=True)
            logger.log_tabular("Q1Vals", average_only=True)
            logger.log_tabular("Q2Vals", average_only=True)
            logger.log_tabular("QCVals", average_only=True)
            logger.log_tabular("LogPi", average_only=True)
            logger.log_tabular("LossPi", average_only=True)
            if if_cbf:
                logger.log_tabular("LossDynamics", average_only=True)
                logger.log_tabular("LossCBF", average_only=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()
            updated, finished = False, False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--hid", type=int, default=128)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--if-cbf", action='store_true')
    parser.add_argument("--if-fixed-h", action='store_true')
    parser.add_argument("--exp-name", type=str, default="sac")
    args = parser.parse_args()

    from utils.logx import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    # env = gym.make("Pendulum-v0")
    env = RccarGazeboEnv({"collision_reward": 10})
    sac(
        env,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        dynamics_kwargs=dict(f_hidden_sizes=[512, 512], g_hidden_sizes=[512, 512]),
        gamma=args.gamma,
        seed=args.seed,
        alpha=args.alpha,
        beta=args.beta,
        if_cbf=args.if_cbf,
        # if_cbf=True,
        if_fixed_h=args.if_fixed_h,
        # if_fixed_h=False,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
