import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, dropout):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action
        self.dropout = nn.Dropout(p=dropout)
        self.max_logstd = 2

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = self.dropout(x)
        x = F.relu(self.l2(x))
        x = self.dropout(x)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=self.max_logstd)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = self.max_action * torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - torch.tanh(x_t).pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mu = self.max_action * torch.tanh(mean)

        return mu, action, log_prob
    
    def setmaxlogstd(self, value):
        self.max_logstd = value

# Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, dropout):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = self.dropout(q1)
        q1 = F.relu(self.l2(q1))
        q1 = self.dropout(q1)
        q1 = self.l3(q1)

        return q1

# Replay buffer
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size):
        self.max_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((buffer_size, state_dim))
        self.action = np.zeros((buffer_size, action_dim))
        self.next_state = np.zeros((buffer_size, state_dim))
        self.reward = np.zeros((buffer_size, 1))
        self.not_done = np.zeros((buffer_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.not_done[ind])
        )

# REDQSAC agent
class REDQSACAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma, tau, lr, batch_size, buffer_size, nr_critics, utd, dropout=0, lam_s=1, lam_t = 1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.nr_critics = nr_critics
        self.utd = utd
        self.dropout = dropout
        self.lam_s = lam_s
        self.lam_t = lam_t
        self.critics = []
        self.t_critics = []
        self.critics_optimizers = []

        self.actor = Actor(state_dim, action_dim, max_action, dropout).to("cpu")

        for _ in range(self.nr_critics):
            critic = Critic(state_dim, action_dim, dropout).to("cpu")
            critic_target = Critic(state_dim, action_dim, dropout).to("cpu")
            critic_target.load_state_dict(critic.state_dict())
            critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
            self.critics.append(critic)
            self.t_critics.append(critic_target)
            self.critics_optimizers.append(critic_optimizer)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

    def update(self):
        for _ in range(self.utd):
            # Sample a batch from the replay buffer
            state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)

            # Update critic
            with torch.no_grad():
                _, next_action, log_prob = self.actor.sample(next_state)
                target_q = self.t_critics[0](next_state, next_action)
                for t_critic in self.t_critics:
                    target_q1 = t_critic(next_state, next_action)
                    target_q = torch.min(target_q, target_q1)
                target_q = target_q - self.log_alpha.exp() * log_prob
                target_q = reward + not_done * self.gamma * target_q

            for i in range(self.nr_critics):
                current_q = self.critics[i](state, action)
                critic_loss = F.mse_loss(current_q, target_q)
                self.critics_optimizers[i].zero_grad()
                critic_loss.backward()
                self.critics_optimizers[i].step()

            # Update critic targets
            for i in range(self.nr_critics):
                for param, target_param in zip(self.critics[i].parameters(), self.t_critics[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
             
            # Update actor
            if self.replay_buffer.size > self.batch_size:
                with torch.no_grad():
                    mu_next, _, _ = self.actor.sample(next_state)
                    spacial_std = torch.full((self.state_dim,), 0.01)
                    mu_bar, _, _ = self.actor.sample(torch.normal(state, spacial_std))
                
                mu, pi, log_prob = self.actor.sample(state)
                q_pi = self.critics[0](state, pi)
                for critic in self.critics:
                    q1_pi = critic(state, pi)
                    q_pi = torch.min(q_pi, q1_pi)
                actor_loss = (self.log_alpha.exp() * log_prob - q_pi).mean()
                #Below is code for CAPS regularisation
                #----------------------------------------------------
                # Temporal smoothness

                temp_loss = self.lam_t*(torch.abs(mu - mu_next)).mean()

                # # Spacial smoothness
                spacial_loss = self.lam_s*(torch.abs(mu - mu_bar)).mean()
                #----------------------------------------------------
                #CAPS ends here
                total_actor_loss = actor_loss + temp_loss + spacial_loss
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                self.actor_optimizer.step()

                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
        return critic_loss.item(), critic_loss.item(), actor_loss.item(), temp_loss.item(), spacial_loss.item()

    def save_weights(self, path_prefix):
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        for i in range(len(self.critics)):
            torch.save(self.critics[i].state_dict(), f"{path_prefix}_critic{i+1}.pth")
    
    def load_weights(self, actor_file, critic_list):
        self.actor.load_state_dict(torch.load(actor_file,weights_only=True))
        self.actor.eval()
        for i in range(len(critic_list)):
            critic = self.critics[i]
            critic.load_state_dict(torch.load(critic_list[i],weights_only=True))
            critic.eval()

    def setlog(self, value):
        self.actor.setmaxlogstd(value)
    
    def set_mode(self, mode="train"):
        if mode == "train":
            self.actor.train()
            for critic in self.critics:
                critic.train()

        elif mode == "eval":
            self.actor.eval()
            for critic in self.critics:
                critic.eval()
    
    def fetchutd(self):
        return self.utd
    
    def fetch_nrcritics(self):
        return self.nr_critics