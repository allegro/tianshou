import os
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Batch, Collector, ReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import CQLPolicy
from tianshou.trainer import offline_trainer
from tianshou.utils import BasicLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

EPOCH = 2
STEP_PER_EPOCH = 5000
TEST_NUM = 10
BATCH_SIZE = 256
TASK = "halfcheetah-expert-v1"
SEED = 1234
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
RENDER = 1 / 35


def test_cql():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_sizes = [256, 256]
    test_num = TEST_NUM

    env = gym.make(TASK)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]  # float

    test_envs = SubprocVectorEnv([lambda: gym.make(TASK) for _ in range(test_num)])
    # seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    test_envs.seed(SEED)

    # Models
    net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(
        net_a,
        action_shape,
        max_action=max_action,
        device=device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)

    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=CRITIC_LR)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=CRITIC_LR)

    policy = CQLPolicy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim
    )

    test_collector = Collector(policy, test_envs)

    # Logger
    log_path = "test"
    writer = SummaryWriter(log_path)
    writer.add_text("args")
    logger = BasicLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    dataset = d4rl.qlearning_dataset(env)
    dataset_size = dataset["rewards"].size

    print("dataset_size", dataset_size)
    replay_buffer = ReplayBuffer(dataset_size)

    for i in range(dataset_size):
        replay_buffer.add(
            Batch(
                obs=dataset["observations"][i],
                act=dataset["actions"][i],
                rew=dataset["rewards"][i],
                done=dataset["terminals"][i],
                obs_next=dataset["next_observations"][i],
            )
        )
    print("dataset loaded")

    result = offline_trainer(
        policy,
        replay_buffer,
        test_collector,
        EPOCH,
        STEP_PER_EPOCH,
        test_num,
        BATCH_SIZE,
        save_fn=save_fn,
        logger=logger,
    )
    pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(SEED)
    test_collector.reset()
    result = test_collector.collect(n_episode=test_num, render=RENDER)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    test_cql()
