import torch
import torch.nn as nn
import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self, x):
        value = self.network(x)
        return value

class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.std_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()
        )
    
    def forward(self, x):
        features = self.shared(x)
        mu = 2.0 * self.mu_head(features)
        std = self.std_head(features) + 1e-6  # 数值稳定性
        return mu, std
    
    


class PPO():
    def __init__(self, state_dim, hidden_dim, action_dim, lr, lmbda, epochs, eps, gamma, device):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.lr = lr
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.gamma = gamma
        self.device = device
        
        self.actor = ActorNet(state_dim=self.state_dim,hidden_dim=self.hidden_dim, action_dim=self.action_dim)
        self.critic = CriticNet(state_dim=self.state_dim, hidden_dim=self.hidden_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-3)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return [action.item()]
    
    def update_network(self, transiton_dict):
        states = torch.tensor(np.array(transiton_dict['states']), dtype=torch.float, device=self.device)
        actions = torch.tensor(np.array(transiton_dict['actions']), dtype=torch.float, device=self.device).view(-1,1)
        rewards = torch.tensor(np.array(transiton_dict['rewards']),dtype=torch.float, device=self.device).view(-1,1)
        next_states = torch.tensor(np.array(transiton_dict['next_states']), dtype=torch.float, device=self.device)
        dones = torch.tensor(np.array(transiton_dict['dones']), dtype=torch.float,device=self.device).view(-1,1)

        #计算优势函数
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_error = td_target - self.critic(states)
        advantage = self.compute_advantage(gamma=self.gamma, lmbda=self.lmbda, td_error=td_error)


        #重要性采样：动作在旧策略的概率
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            new_log_probs = action_dists.log_prob(actions)#同样的动作在更新了参数后的对数概率
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio*advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = nn.MSELoss()(self.critic(states), td_target.detach())
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
      


    def compute_advantage(self, gamma, lmbda, td_error):
        td_error = td_error.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_error[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float, device=self.device)





if __name__ == "__main__":
    lr = 1e-4
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # 连续动作空间
    agent = PPO(state_dim, hidden_dim, action_dim,lr, lmbda, epochs, eps, gamma, device)


    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state,_ = env.reset(seed=42)
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated,_ = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update_network(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)