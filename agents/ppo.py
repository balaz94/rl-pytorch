import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.stat import write_to_file

class AgentPPO:
    def __init__(self, gamma, actions_count, model, lr = 0.0001, beta_entropy = 0.001, value_loss_coef = 0.5, id=1, name='ppo', epsilon = 0.2, lr_decay = 1e-6):
        self.gamma = gamma
        self.actions_count = actions_count
        self.model = model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.beta_entropy = beta_entropy
        self.value_loss_coef = value_loss_coef

        self.average_score = []
        self.episodes = 0
        self.upper_bound = 1 + epsilon
        self.lower_bound = 1 - epsilon

        self.id = id
        self.name = name

        self.lr = lr
        self.lr_decay = lr_decay

    def choose_action(self, state):
        state = state.unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            logits, _ = self.model(state)
        probs = F.softmax(logits, dim=-1)
        probs = probs.cpu()
        action = probs.multinomial(num_samples=1).detach()
        return action[0].item()

    def learn(self, workers, max_steps, max_iterations, write=True, start_iteration=0, max_epochs = 4, batches_per_epoch = 4):
        len_workers = len(workers)
        observations = []
        for worker in workers:
            observations.append(torch.from_numpy(worker.reset()).float())
        observations = torch.stack(observations)

        self.average_score = []
        self.episodes = 0
        best_avg = -100

        max_steps_epochs = max_steps * batches_per_epoch

        text = 'iteration,episode,score,step'
        iter_step = max_steps_epochs * len_workers
        index_range = torch.arange(0, max_steps_epochs).long()

        for iteration in range(start_iteration, max_iterations):
            mem_values = torch.zeros([max_steps_epochs, len_workers, 1])
            mem_log_probs = torch.zeros([max_steps_epochs, len_workers, 1])
            mem_rewards = torch.zeros([max_steps_epochs, len_workers, 1])
            mem_non_terminals = torch.ones([max_steps_epochs, len_workers, 1])
            mem_actions = torch.zeros(max_steps_epochs, len_workers, 1).long()
            mem_observations = []

            for step in range(max_steps_epochs):
                with torch.no_grad():
                    logits, values = self.model(observations.to(self.device))
                mem_observations.append(observations)

                logits, values = logits.cpu(), values.cpu()
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)

                actions = probs.multinomial(num_samples=1).detach()
                log_policy = log_probs.gather(1, actions)

                rewards = torch.zeros([len_workers, 1])
                non_terminals = torch.ones([len_workers, 1])
                observations = []

                for i in range(len_workers):
                    o, rewards[i, 0], t = workers[i].step(actions[i].item())
                    if t == True:
                        non_terminals[i, 0] = 0
                        o = workers[i].reset()
                    observations.append(torch.from_numpy(o).float())
                observations = torch.stack(observations)

                mem_values[step] = values
                mem_log_probs[step] = log_policy
                mem_rewards[step] = rewards
                mem_non_terminals[step] = non_terminals
                mem_actions[step] = actions

            with torch.no_grad():
                _, R = self.model(observations.to(self.device))
                R = R.detach().cpu()
                #mem_values[max_steps_epochs] = R

            avg = np.average(self.average_score[-100:])
            if avg > best_avg:
                best_avg = avg
                print('saving model, best score is ', best_avg)
                torch.save(self.model.state_dict(), 'models/' + self.name + '_' + str(self.id) + '_ppo.pt')

            if iteration % 25 == 0 and iteration > 0:
                print(iteration, '\tepisodes: ', self.episodes, '\taverage score: ', avg)
                if write:
                    text += '\n' + str(iteration) + ',' + str(self.episodes) + ',' + str(avg) + ',' + str(iter_step * iteration)

                    if iteration % 100 == 0:
                        self.average_score = self.average_score[-100:]
                        write_to_file(text, 'logs/' + self.name + '_' + str(self.id) + '_' + str(iteration) + '_ppo.txt')
                        torch.save(self.model.state_dict(), 'models/' + self.name + '_' + str(self.id) + '_' + str(iteration) + '_ppo.pt')

            '''
            mem_R = torch.zeros([max_steps_epochs, len_workers, 1])
            advantages = torch.zeros([max_steps_epochs, len_workers, 1])
            #returns = torch.zeros([max_steps_epochs, len_workers, 1])
            #gae = torch.zeros(len_workers, 1)

            for step in reversed(range(max_steps_epochs)):
                R = mem_rewards[step] + self.gamma * R * mem_non_terminals[step]
                #delta = mem_rewards[step] + self.gamma * mem_values[step+1] * mem_non_terminals[step] - mem_values[step]
                #gae = gae * self.gamma * 0.95 + delta
                advantage[step] = R - mem_values[step]
            '''
            mem_R = torch.zeros([max_steps_epochs, len_workers, 1])
            advantages = torch.zeros([max_steps_epochs, len_workers, 1])

            for step in reversed(range(max_steps_epochs)):
                R = mem_rewards[step] + self.gamma * R * mem_non_terminals[step]
                mem_R[step] = R
                advantages[step] = R - mem_values[step]

            #mem_R = mem_rewards + self.gamma * mem_values[1:max_steps_epochs+1]
            #advantages = mem_R - mem_values[0:max_steps_epochs]

            #advantages = returns - mem_values[0:max_steps_epochs, :]
            #advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-5)

            mem_observations = torch.stack(mem_observations)

            for epoch in range(max_epochs):
                index = index_range[torch.randperm(max_steps_epochs)].view(-1, max_steps)
                for batch in range(batches_per_epoch):
                    epoch_index = index[epoch]
                    #torch.cuda.empty_cache()
                    states = mem_observations[epoch_index].view(-1, 4, 96, 96)
                    logits, values = self.model(states.to(self.device))

                    logits, values = logits.cpu(), values.cpu()
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)

                    log_new_policy = log_probs.gather(1, mem_actions[epoch_index].view(-1, 1))
                    entropies = (log_probs * probs).sum(1, keepdim=True)

                    advantage = mem_R[epoch_index].view(-1, 1) - values
                    value_loss = (advantage**2).mean()

                    ratio = torch.exp(log_new_policy - mem_log_probs[epoch_index].view(-1, 1))

                    epoch_advangate = copy.deepcopy(advantages[epoch_index].view(-1, 1))

                    surr_policy = ratio * epoch_advangate
                    surr_clip = torch.clamp(ratio, self.lower_bound, self.upper_bound) * epoch_advangate
                    policy_loss = - torch.min(surr_policy, surr_clip).mean()
                    entropy_loss = entropies.mean()

                    self.optimizer.zero_grad()
                    loss = policy_loss + self.value_loss_coef * value_loss + self.beta_entropy * entropy_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                    self.optimizer.step()

            if iteration % 25 == 0 and iteration > 0:
                self.lr = max(self.lr - self.lr_decay, 1e-7)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


    def load_model(self):
        self.model.load_state_dict(torch.load('models/' + self.name))
        self.old_model = copy.deepcopy(self.model)

    def save_model(self):
        torch.save(self.model.state_dict(), 'models/' + self.name + '_' + str(self.id) + '_ppo.pt')

class Worker:
    def __init__(self, id, env, agent, print_score = False):
        self.id = id
        self.env = env

        self.print_score = print_score
        self.episode = 1
        self.state = None
        self.score = 0
        self.agent = agent

    def reset(self):
        if self.print_score and self.episode % 10 == 0:
            print('worker: ', self.id, '\tepisode: ', self.episode, '\tscore: ', self.score)
        self.agent.average_score.append(self.score)
        self.agent.episodes += 1
        self.state = self.env.reset()
        self.episode += 1
        self.score = 0
        return self.state

    def step(self, action):
        self.state, r, t, _ = self.env.step(action)
        self.score += r
        if r > 1.0:
            r = 1.0
        elif r < -1.0:
            r = -1.0
        return self.state, r, t
