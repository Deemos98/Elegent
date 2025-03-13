

### 1. 交易环境设计（TradingEnv）

import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data, init_balance=10000):
        super(TradingEnv, self).__init__()
        
        # 数据参数
        self.data = data  # 包含市场特征和标签的DataFrame
        self.current_step = 0
        self.max_steps = len(data)-1
        
        # 账户参数
        self.init_balance = init_balance
        self.balance = init_balance
        self.position = 0  # 0: 空仓, 1: 多仓, -1: 空仓
        self.entry_price = 0.0
        
        # 动作空间：0-保持，1-开多，2-开空，3-平仓
        self.action_space = spaces.Discrete(4)  
        
        # 状态空间设计（示例特征）
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(25,),  # 根据实际特征数量调整
            dtype=np.float32
        )
        
        # 奖励参数
        self.trade_count = 0
        self.max_drawdown = 0
        self.peak_balance = init_balance

    def _get_state(self):
        """获取当前状态"""
        features = self.data.iloc[self.current_step].values
        
        # 拼接账户状态信息
        state = np.concatenate([
            features,
            [self.position, 
             self.balance/self.init_balance, 
             self.entry_price/self.data.iloc[self.current_step]['close']]
        ])
        return state

    def step(self, action):
        done = False
        info = {}
        
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        # 执行动作
        if action == 1 and self.position == 0:  # 开多
            self.position = 1
            self.entry_price = current_price
            self.trade_count += 1
            
        elif action == 2 and self.position == 0:  # 开空
            self.position = -1
            self.entry_price = current_price
            self.trade_count += 1
            
        elif action == 3 and self.position != 0:  # 平仓
            pnl = (current_price/self.entry_price - 1) * self.position
            self.balance *= (1 + pnl)
            reward = pnl * 100  # 放大收益信号
            
            # 计算动态奖励
            drawdown = (self.peak_balance - self.balance)/self.peak_balance
            if drawdown > self.max_drawdown:
                reward -= drawdown * 50  # 惩罚回撤
                
            self.position = 0
            self.entry_price = 0.0
            self.peak_balance = max(self.peak_balance, self.balance)
            
        # 计算持仓浮动盈亏
        if self.position != 0:
            unrealized_pnl = (current_price/self.entry_price - 1) * self.position
            reward += unrealized_pnl * 0.1  # 部分奖励未平仓收益
            
        # 环境终止条件
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            # 终局奖励
            reward += (self.balance/self.init_balance - 1) * 100
            
        # 更新账户信息
        info.update({
            "balance": self.balance,
            "position": self.position,
            "step": self.current_step
        })
        
        return self._get_state(), reward, done, info

    def reset(self):
        self.current_step = 0
        self.balance = self.init_balance
        self.position = 0
        self.entry_price = 0.0
        self.trade_count = 0
        return self._get_state()


### 2. PPO智能体实现

import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        features = self.feature(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def update(self, memory):   
        # 转换数据为tensor
        states = torch.stack(memory.states)
        actions = torch.stack(memory.actions)
        old_logprobs = torch.stack(memory.logprobs)
        rewards = torch.tensor(memory.rewards).float()
        masks = torch.tensor(memory.masks).float()
        
        # 计算折扣奖励
        discounted_rewards = []
        running_reward = 0
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            running_reward = reward + self.gamma * running_reward * mask
            discounted_rewards.insert(0, running_reward)
            
        # 标准化奖励
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)
        
        # 优化策略多次
        for _ in range(5):  # PPO的更新次数
            # 评估旧动作
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(actions)
            
            # 计算比率
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # 计算优势
            advantages = discounted_rewards - state_values.detach()
            
            # 计算损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = self.MseLoss(state_values, discounted_rewards)
            
            # 总损失
            loss = actor_loss + 0.5 * critic_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())


### 3. 训练流程整合

from collections import deque

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.masks = []
        
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.masks[:]

def train_ppo(env, hyperparams, total_episodes=1000):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    ppo = PPO(state_dim, action_dim, 
             lr=hyperparams['lr'],
             gamma=hyperparams['gamma'])
    
    memory = Memory()
    best_reward = -np.inf
    
    for episode in range(total_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            # 转换状态为tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 选择动作
            with torch.no_grad():
                action_probs, _ = ppo.policy_old(state_tensor)
            
            dist = Categorical(action_probs)
            action = dist.sample()
            logprob = dist.log_prob(action)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action.item())
            
            # 存储轨迹
            memory.states.append(state_tensor)
            memory.actions.append(action)
            memory.logprobs.append(logprob)
            memory.rewards.append(reward)
            memory.masks.append(1 - done)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            
            if done:
                # 更新策略
                if len(memory.rewards) > 64:  # 达到最小batch size
                    ppo.update(memory)
                    memory.clear()
                
                # 记录最佳模型
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(ppo.policy.state_dict(), 'best_ppo_trader.pth')
                
                print(f"Episode: {episode}, Reward: {episode_reward:.2f}")
                break


### 4. 策略应用与优化方向

# 加载训练好的模型
policy = ActorCritic(state_dim, action_dim)
policy.load_state_dict(torch.load('best_ppo_trader.pth'))

def ppo_trading_policy(state):
    state_tensor = torch.FloatTensor(state)
    action_probs, _ = policy(state_tensor)
    return torch.argmax(action_probs).item()

# 在交易循环中调用
state = env.reset()
while True:
    action = ppo_trading_policy(state)
    next_state, reward, done, info = env.step(action)
    state = next_state
    if done:
        break
'''
**优化方向：**
1. **状态空间增强：**
   ```python
   # 在_get_state()中添加更多市场特征
   features = np.concatenate([
       self.data.iloc[self.current_step][['close', 'volume', 'rsi', 'macd']].values,
       self.data.iloc[self.current_step-5:self.current_step].mean().values  # 5步均值
   ])
   ```

2. **奖励函数改进：**
   ```python
   # 在step()函数中增加风险调整后的奖励
   sharpe_ratio = calculate_sharpe(returns)  # 需实现夏普率计算
   reward += sharpe_ratio * 0.3
   
   # 惩罚频繁交易
   if self.trade_count > 10:
       reward -= (self.trade_count - 10) * 0.1
   ```

3. **网络结构优化：**
   ```python
   # 使用LSTM处理时序特征
   class ActorCriticLSTM(nn.Module):
       def __init__(self, state_dim, action_dim):
           super().__init__()
           self.lstm = nn.LSTM(state_dim, 128, batch_first=True)
           self.actor = nn.Linear(128, action_dim)
           self.critic = nn.Linear(128, 1)
   ```

4. **多时间尺度集成：**
   ```python
   # 在环境中集成多时间尺度数据
   def __init__(self, data_15m, data_1h):
       self.data_15m = data_15m
       self.data_1h = data_1h
       # 在状态中包含多尺度特征
       self.observation_space = spaces.Box(..., shape=(15m_dim + 1h_dim + 4,))
   ```

'''
