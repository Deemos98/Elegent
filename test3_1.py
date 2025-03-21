import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import gc
import os
from gym import spaces
from tqdm import tqdm
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# 设置显存优化配置
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### 1. 交易环境设计（TradingEnv）
class TradingEnv(gym.Env):
    def __init__(self, data, leverage, init_balance=10000, chunk_size=50000):
        super(TradingEnv, self).__init__()
        self.chunk_size = chunk_size
        self.data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        self.current_chunk = 0
        self.data = self.data_chunks[0].copy()
        self.scaler = StandardScaler()
        self.feature = data.columns
        self.data.loc[:, self.feature] = self.scaler.fit_transform(self.data[self.feature])
        self.data = self.data[(np.abs(self.data[self.feature]) < 10).all(axis=1)]
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        self.init_balance = init_balance
        self.balance = init_balance
        self.position = 0
        self.entry_price = 0.0
        self.leverage = leverage
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32
        )
        self.trade_count = 0
        self.max_drawdown = 0.0
        self.peak_balance = init_balance

    def _get_state(self):
        current_price = self.data.iloc[self.current_step]['close']
        features = self.data.iloc[self.current_step][self.feature].values
        return_rate = (
                                  current_price / self.entry_price - 1) * self.leverage * self.position if self.position != 0 else 0.0
        state = np.concatenate([features, [self.position, return_rate, self.balance / self.init_balance]])
        return np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)

    def step(self, action):
        done = False
        info = {}
        if self.current_step >= len(self.data):
            raise IndexError(f"current_step {self.current_step} exceeds data length {len(self.data)}")
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
        self.max_drawdown = min(self.max_drawdown, current_drawdown)
        self.peak_balance = max(self.peak_balance, self.balance)

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            fee = self.balance * 0.001  # 开仓手续费 [[2]]
            self.balance -= fee
            self.trade_count += 1
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = current_price
            fee = self.balance * 0.001  # 开仓手续费 [[3]]
            self.balance -= fee
            self.trade_count += 1
        elif action == 3 and self.position != 0:
            fee = self.balance * 0.001  # 平仓手续费 [[4]]
            self.balance -= fee
            pnl = (current_price / self.entry_price - 1) * self.leverage * self.position
            self.balance *= (1 + pnl)
            reward = pnl * 10
            self.position = 0

        if self.position != 0:
            unrealized_pnl = (current_price / self.entry_price - 1) * self.leverage * self.position
            reward += unrealized_pnl * 0.1  # 未实现盈亏奖励保持不变

        if self.current_step >= self.max_steps or current_drawdown < -0.7:
            done = True
        else:
            self.current_step += 1

        self.current_step = min(self.current_step, len(self.data) - 1)
        next_state = self._get_state() if not done else np.zeros(self.observation_space.shape)
        info = {
            "balance": self.balance,
            "position": self.position,
            "step": self.current_step
        }
        return next_state, reward, done, info

    def reset(self):
        while True:
            if self.current_step >= self.max_steps:
                del self.data
                torch.cuda.empty_cache()
                gc.collect()

                self.current_chunk = (self.current_chunk + 1) % len(self.data_chunks)
                self.data = self.data_chunks[self.current_chunk].copy()
                self.data.loc[:, self.feature] = self.scaler.transform(self.data[self.feature])
                self.data = self.data[(np.abs(self.data[self.feature]) < 10).all(axis=1)]
                self.max_steps = len(self.data) - 1
                break
            else:
                break

        self.current_step = 0
        self.balance = self.init_balance
        self.position = 0
        self.entry_price = 0.0
        self.trade_count = 0
        return self._get_state()


### 2. PPO智能体实现
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.actor = nn.Linear(32, action_dim)
        self.critic = nn.Linear(32, 1)
        for layer in self.feature:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        nn.init.xavier_normal_(self.actor.weight)
        nn.init.xavier_normal_(self.critic.weight)

    def forward(self, x):
        x = self.feature(x)
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        dist = Categorical(action_probs)
        entropy = dist.entropy().mean()
        return action_probs, state_value, entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory, batch_size=2048):
        states = torch.stack(memory.states).to(device)
        actions = torch.stack(memory.actions).to(device)
        old_logprobs = torch.stack(memory.logprobs).to(device)
        rewards = torch.tensor(memory.rewards, dtype=torch.float32).to(device)
        masks = torch.tensor(memory.masks, dtype=torch.float32).to(device)

        discounted_rewards = []
        running_reward = 0
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            running_reward = reward + self.gamma * running_reward * mask
            discounted_rewards.insert(0, running_reward)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        dataloader = DataLoader(TensorDataset(states, actions, old_logprobs, discounted_rewards),
                                batch_size=batch_size, shuffle=True)

        for _ in range(4):
            for batch in dataloader:
                state_batch, action_batch, old_logprob_batch, reward_batch = [t.to(device) for t in batch]
                action_probs, state_values, entropy = self.policy(state_batch)
                dist = Categorical(action_probs)
                logprobs = dist.log_prob(action_batch)
                ratios = torch.exp(logprobs - old_logprob_batch.detach())
                advantages = reward_batch - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.MseLoss(state_values, reward_batch)
                # 添加KL散度约束 [[4]]
                with torch.no_grad():
                    old_dist = Categorical(logits=self.policy_old.actor(self.policy_old.feature(state_batch)))
                kl_div = torch.distributions.kl_divergence(old_dist, dist).mean()
                loss = actor_loss + 0.5 * critic_loss + 0.1 * kl_div - 0.01 * entropy  # 添加熵正则化 [[7]]

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        del states, actions, old_logprobs, rewards, masks
        torch.cuda.empty_cache()


### 3. 训练流程整合
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.masks = []
        self.batch_size = None

    def cuda(self, batch_size=None):
        if batch_size is None:
            batch_size = len(self.states)

        self.cpu()
        torch.cuda.empty_cache()
        gc.collect()

        self.states = [t.to('cuda') for t in self.states[:batch_size]]
        self.actions = [t.to('cuda') for t in self.actions[:batch_size]]
        self.logprobs = [t.to('cuda') if isinstance(t, torch.Tensor) else torch.tensor(t).to('cuda')
                         for t in self.logprobs[:batch_size]]
        self.rewards = [torch.tensor(r).to('cuda') for r in self.rewards[:batch_size]]
        self.masks = [torch.tensor(m).to('cuda') for m in self.masks[:batch_size]]
        self.batch_size = batch_size

    def cpu(self):
        self.states = [t.to('cpu') for t in self.states]
        self.actions = [t.to('cpu') for t in self.actions]
        self.logprobs = [t.to('cpu') for t in self.logprobs]
        self.rewards = [torch.tensor(r).to('cpu') for r in self.rewards]
        self.masks = [torch.tensor(m).to('cpu') for m in self.masks]
        torch.cuda.empty_cache()

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.masks[:]
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.masks = []
        gc.collect()
        torch.cuda.empty_cache()


def train_ppo(env, hyperparams, total_episodes=1000):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo = PPO(state_dim, action_dim, lr=hyperparams['lr'], gamma=hyperparams['gamma'])
    memory = Memory()
    max_retry = 5
    min_batch_size = 64
    initial_batch_size = 2048
    init_balance = env.init_balance

    pbar = tqdm(range(total_episodes), desc="Training")
    for episode in pbar:
        state = env.reset()
        episode_reward = 0
        memory.clear()
        current_batch_size = initial_batch_size

        while True:
            if current_batch_size and len(memory.states) >= current_batch_size:
                break
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs, _, _ = ppo.policy_old(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample().to(device)
            logprob = dist.log_prob(action).detach()
            next_state, reward, done, info = env.step(action.item())

            memory.states.append(state_tensor)
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.rewards.append(torch.tensor(reward))  # 存储张量 [[1]]
            memory.masks.append(torch.tensor(1 - done))  # 存储张量 [[6]]

            state = next_state
            episode_reward += reward
            if done:
                retry_count = 0
                success = False
                while retry_count < max_retry and not success:
                    try:
                        memory.cuda(current_batch_size)
                        print(f"Updating with {len(memory.states)} samples")
                        ppo.update(memory)
                        success = True
                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e) and current_batch_size > min_batch_size:
                            current_batch_size = max(current_batch_size // 2, min_batch_size)
                            print(f"Reduce batch_size to {current_batch_size}")
                            memory.cpu()
                            retry_count += 1
                        else:
                            raise e
                if not success:
                    raise RuntimeError("Failed to update after multiple retries.")
                memory.clear()
                initial_batch_size = current_batch_size
                break

        final_balance = info["balance"]
        return_rate = (final_balance - init_balance) / init_balance * 100
        pbar.set_postfix({
            "Reward": f"{episode_reward:.2f}",
            "Return": f"{return_rate:.2f}%",
            "Balance": f"{final_balance:.2f}"
        })
        print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Return: {return_rate:.2f}%")


### 4. 数据预处理与启动训练
if __name__ == "__main__":
    df = pd.read_csv(r'D:\lgb_future2.csv').dropna()
    feature_columns = ['close', 'position', 'return_rate']
    feature_columns = [
        'close',
        'position',
        'return_rate',
        'volume',
        '1min_prev',
        '5min_prev',
        '15min_prev',
        '30min_prev',
        '60min_prev',
        '3h_prev',
        '8h_prev',
        '24h_prev',
        'current_high_rate_15m',
        'current_low_rate_15m',
        'macd1_15m',
        'rsi1_15m',
        'adx1_15m',
        'sar1_15m',
        'bull_upper_15m',
        'bull_lower_15m',
        'vwap'
    ]
    env = TradingEnv(data=df[feature_columns], leverage=10)  # 降低杠杆至5倍 [[8]]
    hyperparams = {
        'lr': 5e-5,  # 降低学习率 [[9]]
        'gamma': 0.95,  # 减少长期负奖励的影响 [[5]]
        'eps_clip': 0.15  # 缩小更新范围 [[7]]
    }
    train_ppo(env, hyperparams, total_episodes=100)
