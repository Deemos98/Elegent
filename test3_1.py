
### 1. 交易环境设计（TradingEnv）

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # 导入F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TradingEnv(gym.Env):
    def __init__(self, data,leverage, init_balance=10000):
        super(TradingEnv, self).__init__()

        # 数据参数
        self.data = data  # 包含市场特征和标签的DataFrame
        self.current_step = 0
        self.max_steps = len(data ) -1
        self.feature = data.columns
        # 账户参数
        self.init_balance = init_balance
        self.balance = init_balance
        self.position = 0  # 0: 空仓, 1: 多仓, -1: 空仓
        self.entry_price = 0.0
        self.leverage = leverage  # 新增杠杆属性
        # 动作空间：0-保持，1-开多，2-开空，3-平仓
        self.action_space = spaces.Discrete(4)

        # 状态空间设计（示例特征）
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(29,),  # 根据实际特征数量调整
            dtype=np.float32
        )

        # 奖励参数
        self.trade_count = 0
        self.max_drawdown = 0.0
        self.peak_balance = init_balance

    def _get_state(self):
        current_price = self.data.iloc[self.current_step]['close']
        features = self.data.iloc[self.current_step][self.feature].values  # 假设feature是预定义的特征列

        # 计算实时return_rate（考虑仓位方向）
        if self.position == 1:  # 多仓
            return_rate = (current_price / self.entry_price - 1) * self.leverage
        elif self.position == -1:  # 空仓
            return_rate = (self.entry_price / current_price - 1) * self.leverage
        else:
            return_rate = 0.0

        # 拼接状态向量
        state = np.concatenate([
            features,
            [self.position, return_rate, self.balance / self.init_balance]
        ])
        if np.isnan(state).any() or np.isinf(state).any():
            raise ValueError(f"Invalid state values at step {self.current_step}")
        return state

    def step(self, action):
        done = False
        info = {}

        current_price = self.data.iloc[self.current_step]['close']
        reward = 0

        # 计算当前回撤（全局有效）
        current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
        # 更新最大回撤
        if current_drawdown < self.max_drawdown:
            self.max_drawdown = current_drawdown

        # 执行动作逻辑（开仓/平仓等）...
        if action == 1 and self.position == 0:  # 开多
            self.position = 1
            self.entry_price = current_price
            self.trade_count += 1

        elif action == 2 and self.position == 0:  # 开空
            self.position = -1
            self.entry_price = current_price
            self.trade_count += 1

        elif action == 3 and self.position != 0:  # 平仓
            if self.position == 1:  # 多仓平仓
                pnl = (current_price / self.entry_price - 1) * self.leverage
            else:  # 空仓平仓
                pnl = (self.entry_price / current_price - 1) * self.leverage

            self.balance *= (1 + pnl)
            reward = pnl * 100  # 放大收益信号

            # 动态奖励逻辑保持不变...

        # 计算持仓浮动盈亏...
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (current_price / self.entry_price - 1) * self.leverage
            else:
                unrealized_pnl = (self.entry_price / current_price - 1) * self.leverage
            reward += unrealized_pnl * 0.1

        # 更新peak_balance（无论是否平仓）
        self.peak_balance = max(self.peak_balance, self.balance)

        # 终止条件
        self.current_step += 1
        if self.current_step >= self.max_steps or current_drawdown < -0.7:
            done = True
            # 终局奖励计算保持不变...

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
        # 初始化权重
        for layer in self.feature:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

        # Actor网络（不包含Softmax）
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)  # 输出logits
        )
        # 初始化Actor权重
        nn.init.xavier_normal_(self.actor[-1].weight)
        nn.init.constant_(self.actor[-1].bias, 0.0)

        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.feature(x)

        # 计算动作概率（带双重数值稳定）
        logits = self.actor(features)

        # 保护1：限制logits范围
        logits = torch.clamp(logits, min=-50, max=50)  # 防止极端值

        # 保护2：稳定softmax计算
        max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
        stable_logits = logits - max_logits
        stable_logits = stable_logits - torch.logsumexp(stable_logits, dim=-1, keepdim=True)  # 直接计算log_softmax

        # 使用log_softmax避免数值问题
        action_logprobs = F.log_softmax(stable_logits, dim=-1)
        action_probs = torch.exp(action_logprobs)  # 从log概率恢复概率

        # 保护3：确保概率合法
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0 - 1e-8)

        # 状态价值
        state_value = self.critic(features)
        return action_probs, state_value

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip

        self.policy = ActorCritic(state_dim, action_dim).to(device)  # 移动到GPU
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)  # 移动到GPU
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # 转换数据为tensor
        states = torch.stack(memory.states).to(device)
        actions = torch.stack(memory.actions).to(device)
        old_logprobs = torch.stack(memory.logprobs).to(device)
        rewards = torch.tensor(memory.rewards).float().to(device)
        masks = torch.tensor(memory.masks).float().to(device)

        # 计算折扣奖励
        discounted_rewards = []
        running_reward = 0
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            running_reward = reward + self.gamma * running_reward * mask
            discounted_rewards.insert(0, running_reward)

        # 标准化奖励
        discounted_rewards = torch.tensor(discounted_rewards).to(device)  # [关键修改]
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
            surr2 = torch.clamp(ratios, 1- self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = self.MseLoss(state_values, discounted_rewards)

            # 总损失
            loss = actor_loss + 0.5 * critic_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # 添加梯度裁剪
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        del states, actions, old_logprobs  # 释放大张量
        torch.cuda.empty_cache()  # 强制清理缓存[[3]][[4]]

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
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.masks = []


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
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # 移动到GPU
            with torch.no_grad():
                action_probs, _ = ppo.policy_old(state_tensor)

            dist = Categorical(action_probs)
            action = dist.sample().to(device)  # 采样并转移至GPU
            logprob = dist.log_prob(action).to(device)  # 计算log_prob并转移


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
                if len(memory.rewards) > 16:  # 达到最小batch size
                    ppo.update(memory)
                    memory.clear()

                # 记录最佳模型
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(ppo.policy.state_dict(), 'best_ppo_trader.pth')

                print(f"Episode: {episode}, Reward: {episode_reward:.2f}")
                break

# 导入依赖库
import pandas as pd


# 1. 加载并预处理数据
df = pd.read_csv(r'D:\lgb_future2.csv',encoding='utf-8-sig')
df = df.dropna()
future = ['close','position', 'return_rate', '1min_prev', 'current_last_low_rate_15m', '30min_prev', 'current_last_high_rate_15m', '5min_prev', '15min_prev', 'rsi1_diff_15m', 'rsi1_15m', '60min_prev', 'current_high_rate_15m', 'current_low_rate_15m', 'bull_upper_current_rate_15m', 'vwap_rate', '24h_prev', 'macd1_diff_15m', '1min_momentum', '2h_momentum', 'taker_buy_base_asset_volume', '12h_momentum', 'taker_sell_base_asset_volume', 'current_high_rate_change_15m', 'adx1_15m', '3h_prev']

# 2. 选取特征并验证维度


# 3. 初始化环境
env = TradingEnv(data=df[future], init_balance=100, leverage=10)

# 4. 启动训练流程（调用train_ppo函数）
hyperparams = {'lr': 3e-4, 'gamma': 0.99}
train_ppo(env, hyperparams, total_episodes=1000)
### 4. 策略应用与优化方向

# 加载训练好的模型
# policy = ActorCritic(state_dim, action_dim)
# policy.load_state_dict(torch.load('best_ppo_trader.pth'))
#
#
# def ppo_trading_policy(state):
#     state_tensor = torch.FloatTensor(state)
#     action_probs, _ = policy(state_tensor)
#     return torch.argmax(action_probs).item()
#
#
# # 在交易循环中调用
# state = env.reset()
# while True:
#     action = ppo_trading_policy(state)
#     next_state, reward, done, info = env.step(action)
#     state = next_state
#     if done:
#         break
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
import shap
feature_score = {col: 0 for col in X_train.columns}
for i in tqdm(range(len(df_best_trials))):
    class_weight_1 = int(df_best_trials['class_weight_1'].iloc[i])
    class_weight_0 = int(df_best_trials['class_weight_-1'].iloc[i])
    param = {
    'objective': 'regression',  # 多分类
    'boosting_type': 'gbdt',  # 使用 GOSS 加速树构建
    'learning_rate': df_best_trials['learning_rate'].iloc[i],
    'num_leaves': int(df_best_trials['num_leaves'].iloc[i]),
    'max_depth': int(df_best_trials['max_depth'].iloc[i]),
    'min_child_samples': int(df_best_trials['min_child_samples'].iloc[i]),
    'max_bin': int(df_best_trials['max_bin'].iloc[i]),
    'subsample': df_best_trials['subsample'].iloc[i],
    'lambda_l1': df_best_trials['lambda_l1'].iloc[i],
    'lambda_l2': df_best_trials['lambda_l2'].iloc[i],
    'bagging_fraction': df_best_trials['bagging_fraction'].iloc[i],
    'feature_fraction': df_best_trials['feature_fraction'].iloc[i],
    'random_state': 42,  # 随机种子
    'verbosity':-1
    }

    # 训练 LGBM 模型
    weights = np.ones_like(y_train, dtype=float)  # 初始化所有权重为1
    # 类别不平衡时的权重调整
    weights[y_train == 1] *= class_weight_1
    weights[y_train == -1] *= class_weight_0
    lgb_model = lgb.LGBMRegressor(**param)
    lgb_model.fit(X_train, y_train, sample_weight=weights)
    # 计算SHAP值
    explainer = shap.TreeExplainer(lgb_model)

    # 特征重要性排序
    feature_importance = pd.Series(
        np.abs(explainer.shap_values(X_train.sample(2000))).mean(axis=0),  # 直接计算重要性
        index=X_train.columns
    ).sort_values(ascending=False)

    # 计算当前模型的特征积分（取前30名）
    top30 = feature_importance[:30]
    for rank, (feature, _) in enumerate(top30.items()):
        # 积分规则：30 - 排名（第一名得29分）
        score = 30 - (rank + 1)
        feature_score[feature] += score
# 根据总积分排序并取前24名
final_ranking = sorted(
    feature_score.items(),
    key=lambda x: x[1],
    reverse=True
)[:25]

# 提取最终特征列表
top_features = [item[0] for item in final_ranking]

print("Top 25 Features:", top_features)

# 可视化
shap.summary_plot(top_features, X_train, plot_type="bar")
'''
