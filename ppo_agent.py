import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
    
    def sample(self):
        indices = np.random.choice(len(self.states), self.batch_size, replace=False)
        states = [self.states[i] for i in indices]
        actions = [self.actions[i] for i in indices]
        rewards = [self.rewards[i] for i in indices]
        next_states = [self.next_states[i] for i in indices]
        dones = [self.dones[i] for i in indices]
        log_probs = [self.log_probs[i] for i in indices]
        
        return states, actions, rewards, next_states, dones, log_probs
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
    
    def __len__(self):
        return len(self.states)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (Policy network)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic (Value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        features = self.feature_extractor(state)
        
        # Actor output (action mean and standard deviation)
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp()
        
        # Critic output (state value)
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_mean, action_std, _ = self.forward(state)
        
        if deterministic:
            return action_mean.squeeze(0).numpy()
        
        # Create a normal distribution and sample from it
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.squeeze(0).numpy(), log_prob.squeeze(0).item()


class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4, gamma=0.99, clip_ratio=0.2, batch_size=64, epochs=10):
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.memory = Memory(batch_size)
        
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        self.epochs = epochs
    
    def select_action(self, state, deterministic=False):
        if deterministic:
            action = self.actor_critic.get_action(state, deterministic=True)
            return action
        else:
            action, log_prob = self.actor_critic.get_action(state)
            return action, log_prob
    
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.memory.add(state, action, reward, next_state, done, log_prob)
    
    def compute_returns(self, rewards, dones):
        returns = []
        discounted_reward = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
            
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        for _ in range(self.epochs):
            states, actions, rewards, next_states, dones, old_log_probs = self.memory.sample()
            
            states = torch.FloatTensor(np.array(states))
            actions = torch.FloatTensor(np.array(actions))
            old_log_probs = torch.FloatTensor(old_log_probs)
            returns = self.compute_returns(rewards, dones)
            
            # Get current policy and value predictions
            action_means, action_stds, values = self.actor_critic(states)
            
            # Calculate new log probabilities
            dist = torch.distributions.Normal(action_means, action_stds)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Calculate advantage
            advantages = returns - values.squeeze()
            
            # Calculate ratios and surrogate losses
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Update the network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.memory.clear() 