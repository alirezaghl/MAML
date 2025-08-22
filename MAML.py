import gymnasium as gym
import random
import matplotlib
matplotlib.use('Agg')  
from matplotlib import pyplot as plt
import numpy as np
from collections import namedtuple, deque
import imageio
from copy import deepcopy
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

os.environ['MUJOCO_GL'] = 'egl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = gym.make("HalfCheetah-v5", render_mode="rgb_array")



def show_video_info(path):
    if os.path.exists(path):
        file_size = os.path.getsize(path) / (1024 * 1024)  
        print(f"Video saved: {path}")


class HalfCheetahBackward(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
        self.forward_reward_weight = 1.0
        self.ctrl_cost_weight = 0.05

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)[0]

    def step(self, action):
        obs, _, done, tr, info = self.env.step(action)
        reward = -1 * self.forward_reward_weight * info["reward_forward"] + self.ctrl_cost_weight * info["reward_ctrl"]
        return obs, reward, done, tr, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers = nn.ModuleList(
            [nn.Linear(obs_dim, hidden_sizes[0])] +
            [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes) - 1)]
        )
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        mean, std = self(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action, dist


def rollout(env, policy, max_steps=200, gamma=0.99):
    obs = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32)
    obs_buf, logp_buf = [], []
    rewards = []
    
    for _ in range(max_steps):
        action, dist = policy.get_action(obs)
        obs_buf.append(obs)
        logp_buf.append(dist.log_prob(action).sum(dim=-1))
        action = action.detach().cpu().numpy()
        obs, reward, done, tr, info = env.step(action)
        rewards.append(reward)
        obs = torch.tensor(obs, dtype=torch.float32)
        if done:
            break
    
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    ret_buf = torch.tensor(returns, dtype=torch.float32)
    return obs_buf, logp_buf, ret_buf


def evaluate(model, env, num_episodes=10, max_episode_len=200, path="./Final_Evaluation.mp4", no_video=False):
    frames = []
    total_rewards = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = torch.tensor(obs, dtype=torch.float32)
        episode_reward = 0
        
        for _ in range(max_episode_len):
            if not no_video and episode == 0:  
                frames.append(env.render())
            
            with torch.no_grad():  
                action, dist = model.get_action(obs)
            
            action_np = action.detach().cpu().numpy()
            obs, reward, done, tr, info = env.step(action_np)
            
            obs = torch.tensor(obs, dtype=torch.float32)
            
            episode_reward += reward
            if done:
                break
                
        total_rewards += episode_reward
    
    mean_reward = total_rewards / num_episodes
    print(f"Mean Reward: {mean_reward:.3f}")
    
    if not no_video and frames:
        try:
            imageio.mimsave(path, frames, fps=20)
            show_video_info(path)
        except Exception as e:
            print(f"Failed to save video: {e}")
    
    env.close()
    return mean_reward


class MAML:
    def __init__(
        self,
        task_env_cls,
        inner_lr,
        outer_lr,
        inner_steps,
        meta_batch_size,
        max_episode_len=200,
        gamma=0.99
    ):
        self.task_env_cls = task_env_cls
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size
        self.max_episode_len = max_episode_len
        self.gamma = gamma
        self.loss_history   = []
        self.reward_history = []

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.meta_policy = Policy(obs_dim, act_dim)
        self.meta_opt = optim.Adam(self.meta_policy.parameters(), lr=outer_lr)

    def inner_update(self, env):
        obs_s, logp_s, ret_s = rollout(env, self.meta_policy, self.max_episode_len, self.gamma)
        logp_s = torch.stack(logp_s)
        pg_loss = -(logp_s * ret_s).mean()


        l2_reg = 1e-5
        l2_loss = sum(torch.norm(param, p=2) ** 2 for param in self.meta_policy.parameters())
        loss_s = pg_loss + l2_reg * l2_loss

        grads = torch.autograd.grad(loss_s, self.meta_policy.parameters(), create_graph=True)
        return grads

    def adapt_policy(self, grads):
      adapted = deepcopy(self.meta_policy)
      for (name, param), grad in zip(adapted.named_parameters(), grads):
          param.data = param.data - self.inner_lr * grad  
      return adapted

    def meta_step(self):
        total_meta_loss = 0.0
        total_reward = 0.0

        for _ in range(self.meta_batch_size):
            env = self.task_env_cls()

            grads = self.inner_update(env)


            adapted = self.adapt_policy(grads)

            obs_q, logp_q, ret_q = rollout(env, adapted, self.max_episode_len, self.gamma)
            logp_q = torch.stack(logp_q)
            loss_q = -(logp_q * ret_q).mean()
            total_meta_loss += loss_q

            total_reward += ret_q.mean().item()

            env.close()

        meta_loss = total_meta_loss / self.meta_batch_size
        self.meta_opt.zero_grad()
        meta_loss.backward()
        self.meta_opt.step()

        self.loss_history.append(meta_loss.item())
        self.reward_history.append(total_reward / self.meta_batch_size)

        return meta_loss.item()

    def train(self, meta_iters=501):
        for it in tqdm(range(1, meta_iters)):
            loss = self.meta_step()
            if it % 10 == 0:
                print(f"\t[Iter {it}]\tloss={loss:.3f},\treward={self.reward_history[-1]:.3f}")
        self.plot_metrics()
        return self.meta_policy

    def plot_metrics(self):
        iters = range(1, len(self.loss_history) + 1)

        plt.figure()
        plt.plot(iters, self.loss_history, label="Loss")
        plt.plot(iters, self.reward_history, label="Avg Query Reward")
        plt.xlabel("Iteration")
        plt.legend()
        plt.title("Training Progress")
        plt.show()


def train(inner_lr, outer_lr, inner_steps, meta_batch_size, max_episode_len, gamma=0.99):
    maml = MAML(
        task_env_cls=HalfCheetahBackward,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        inner_steps=inner_steps,
        meta_batch_size=meta_batch_size,
        max_episode_len=max_episode_len,
        gamma=gamma
    )
    meta_policy = maml.train(meta_iters=500)
    return meta_policy


def main():
    
    policy = train(
        inner_lr=1e-4,
        outer_lr=3e-4,
        inner_steps=1,
        meta_batch_size=4,
        max_episode_len=300,
    )
    
    
    env = HalfCheetahBackward()
    mean_reward = evaluate(policy, env, num_episodes=10)
    
    print(f"\nevaluation reward: {mean_reward:.3f}")
    
    return policy


if __name__ == "__main__":
    policy = main()