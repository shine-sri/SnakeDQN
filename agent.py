import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from model import SnakeDQN


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class SnakeAgent:
    def __init__(self):
        self.memory   = deque(maxlen=100000)
        self.gamma    = 0.99
        self.epsilon  = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.99995  # hits 0.05 around ep 1500-2000
        self.batch_size    = 256

        self.model        = SnakeDQN().to(device)
        self.target_model = SnakeDQN().to(device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.SmoothL1Loss()

        self.steps_done        = 0
        self.target_update_steps = 500

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # state = (grid, extras) tuple — store as-is
        self.memory.append((state, action, reward, next_state, done))

    def _unpack_batch(self, mini_batch):
        grids      = torch.FloatTensor(np.array([x[0][0] for x in mini_batch])).to(device)
        extras     = torch.FloatTensor(np.array([x[0][1] for x in mini_batch])).to(device)
        actions    = torch.LongTensor([x[1] for x in mini_batch]).unsqueeze(1).to(device)
        rewards    = torch.FloatTensor([x[2] for x in mini_batch]).to(device)
        ng         = torch.FloatTensor(np.array([x[3][0] for x in mini_batch])).to(device)
        ne         = torch.FloatTensor(np.array([x[3][1] for x in mini_batch])).to(device)
        dones      = torch.FloatTensor([float(x[4]) for x in mini_batch]).to(device)
        return grids, extras, actions, rewards, ng, ne, dones

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, 3)

        grid_t   = torch.FloatTensor(state[0]).unsqueeze(0).to(device)
        extras_t = torch.FloatTensor(state[1]).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(grid_t, extras_t)[0]
        return int(q_values.argmax().item())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        self.steps_done += 1
        mini_batch = random.sample(self.memory, self.batch_size)
        grids, extras, actions, rewards, ng, ne, dones = self._unpack_batch(mini_batch)

        current_q = self.model(grids, extras).gather(1, actions).squeeze(1)

        # Double DQN
        with torch.no_grad():
            next_actions = self.model(ng, ne).argmax(1, keepdim=True)
            next_q       = self.target_model(ng, ne).gather(1, next_actions).squeeze(1)
            target_q     = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.steps_done % self.target_update_steps == 0:
            self.update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
