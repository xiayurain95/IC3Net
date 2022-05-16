from trainer import Trainer
import collections
import random
import numpy as np
import torch
from torch import optim
from inspect import getargspec


class DQNTrainer(Trainer):
    def __init__(self, args, policy_net, env):
        super().__init__(args, policy_net, env)
        self.replay_buffer = ReplayBuffer()
        self.epsilon = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.98
        self.target_update = 10
        self.count = 0
        self.minimal_size = 500
        self.batch_size = 64
        
        self.action_dim = args.action_dim  # Action number of the traffic light
        self.car_num = args.nagents
        
        self.is_dqn = args.is_dqn
        self.q_net = Qnet(state_dim=args.state_dim,
                          action_dim=self.action_dim).to(self.device)
        self.target_q_net  = Qnet(state_dim=args.state_dim,
                          action_dim=self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(),
            lr = 0.0001, betas=(0.5, 0.999))

    # 应该返回一个episode和一个stat用于被merge stat
    def get_episode(self, epoch):
        episode_return = 0
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step

        # if should_display:
        self.env.display()
        # while done
        for t in range(self.args.max_steps):
            action_vector, action_scalar = self.take_action(state)
            # lamp action, whether apply dqn, cars_action
            if self.is_dqn:
                car_num = self.env.cars_in_sys
                car_action = [1 for _ in range(self.car_num)]
            next_state, reward, done, info = self.env.step(action_scalar, self.is_dqn, car_action)
            done = done or t == self.args.max_steps - 1
            self.replay_buffer.add(state, action_vector, reward, next_state, done)
            episode_return += reward
            
            # begin training when replay buffer is bigger enough
            if self.replay_buffer.size() > self.minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(self.batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    self.update(transition_dict)

            if should_display:
                self.env.display()
        
        return episode_return / self.args.max_steps
    
    # only used when nprocesses=1
    def train_batch(self, epoch):
        rewards = self.run_batch(epoch)
        return rewards
    
    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0

        rewards = self.get_episode(epoch)
        return rewards
    
    def take_action(self, state):
        # flatten state to a vector
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = state.to(self.device)
            action = torch.softmax(self.q_net(state), dim=1).argmax()
        return action, action
    
    def update(self, transition_dict):
        states = torch.tensor(
            transition_dict['states'],
            dtype=torch.double).to(self.device)
        actions = torch.tensor(
            transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            transition_dict['rewards'],
            dtype=torch.double).view(-1, 1).to(self.device)
        next_states = torch.tensor(
            transition_dict['next_states'],
            dtype=torch.double).to(self.device)
        dones = torch.tensor(
            transition_dict['dones'],
            dtype=torch.double).view(-1, 1).to(self.device)

        # Q value
        q_values = self.q_net(states).gather(1, actions)
        # max Q values in the next state
        max_next_q_values = self.target_q_net(
            next_states).max(1)[0].view(-1, 1)
        # TD error objectives
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones) 
        # MSE Loss
        dqn_loss = torch.mean(
            torch.nn.functional.mse_loss(
                q_values, q_targets))
        # clean the previous grad
        self.optimizer.zero_grad()
        # back propogation
        dqn_loss.backward()
        # update the net
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1
    

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        state = torch.cat(state, dim=0).numpy()
        next_state = torch.cat(next_state, dim=0).numpy()
        return state, action, reward, next_state, done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim=2, action_dim=3, hidden_dim=128):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # x = x.flatten()
        x = torch.nn.functional.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)