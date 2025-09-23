import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)
        self.max_action = max_action


    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = self.max_action * torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - torch.tanh(x_t).pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mu = self.max_action * torch.tanh(mean)

        return mu, action, log_prob

# Critic network
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()
        # Q1 network
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

        # Q2 network
        self.l4 = nn.Linear(state_dim + action_dim, 64)
        self.l5 = nn.Linear(64, 64)
        self.l6 = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

# Replay buffer
class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int):
        self.max_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((buffer_size, state_dim))
        self.action = np.zeros((buffer_size, action_dim))
        self.next_state = np.zeros((buffer_size, state_dim))
        self.reward = np.zeros((buffer_size, 1))
        self.not_done = np.zeros((buffer_size, 1))

    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: float, done: bool) -> None:
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.not_done[ind])
        )

# SAC agent
class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, max_action: float, gamma: float, tau: float, lr: float,
                  batch_size: int, buffer_size: int, lam_s: float = 1, lam_t: float = 1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.lam_s = lam_s
        self.lam_t = lam_t
        self.type = "SAC"

        self.actor = Actor(state_dim, action_dim, max_action).to("cpu")
        self.critic = Critic(state_dim, action_dim).to("cpu")
        self.critic_target = Critic(state_dim, action_dim).to("cpu")
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = -action_dim  # Common heuristic is -dim(A)
        self.log_alpha = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

    def update(self) -> tuple[float, float, float, float, float]:
        # Sample a batch from the replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)

        # Update critic network
        with torch.no_grad():
            _, next_action, log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * log_prob
            target_q = reward + not_done * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        c1_loss = F.mse_loss(current_q1, target_q)
        c2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = c1_loss + c2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor network
        if self.replay_buffer.size > self.batch_size:
            with torch.no_grad():
                mu_next, _, _ = self.actor.sample(next_state)
                #Original spacial std
                spacial_std = torch.full((self.state_dim,), 0.01)

                mu_bar, _, _ = self.actor.sample(torch.normal(state, spacial_std))

            mu, pi, log_prob = self.actor.sample(state)
            #print(f"mu is: {mu_bar}")
            q1_pi, q2_pi = self.critic(state, pi)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.log_alpha.exp() * log_prob - q_pi).mean()
            #Below is code for CAPS
            #----------------------------------------------------
            # Temporal smoothness

            temp_loss = self.lam_t*(torch.abs(mu - mu_next)).mean()

            # # Spacial smoothness
            spacial_loss = self.lam_s*(torch.abs(mu - mu_bar)).mean()
            #----------------------------------------------------
            #CAPS stuff ends here
            total_actor_loss = actor_loss + temp_loss + spacial_loss

            if torch.isnan(actor_loss):
                print("NaN detected in actor loss!")

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor_optimizer.step()

            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Update target critic networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return c1_loss.item(), c2_loss.item(), actor_loss.item(), temp_loss.item(), spacial_loss.item()

    def save_weights(self, path_prefix: str) -> None:
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")
    
    def load_weights(self, actor_file: str, critic_file: str) -> None:
        self.actor.load_state_dict(torch.load(actor_file,weights_only=True))
        self.actor.eval()
        self.critic.load_state_dict(torch.load(critic_file,weights_only=True))
        self.critic.eval()