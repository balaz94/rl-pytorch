import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.stat import write_to_file

class AgentA2C:
    def __init__(self, gamma, actions_count, model, lr = 0.0001, beta_entropy = 0.001, value_loss_coef = 0.5, id=1, name='a2c'):
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

        self.id = id
        self.name = name

    def choose_action(self, state):
        state = state.unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            logits, _ = self.model(state)
        probs = F.softmax(logits, dim=-1)
        probs = probs.cpu()
        action = probs.multinomial(num_samples=1).detach()
        return action[0].item()

    def learn(self, workers, max_steps, max_iteration, write=True, start_iteration=0):
        len_workers = len(workers)
        observations = []
        for worker in workers:
            observations.append(torch.from_numpy(worker.reset()).float())
        observations = torch.stack(observations).to(self.device)

        self.average_score = []
        best_avg = -100
        self.episodes = 0

        text = text = 'iteration,episode,score,step'
        iter_step = max_steps * len_workers

        for iteration in range(start_iteration, max_iteration):
            mem_values = torch.zeros([max_steps, len_workers, 1])
            mem_log_probs = torch.zeros([max_steps, len_workers, 1])
            mem_entropies = torch.zeros([max_steps, len_workers, 1])

            mem_rewards = torch.zeros([max_steps, len_workers, 1])
            mem_non_terminals = torch.ones([max_steps, len_workers, 1])

            for step in range(max_steps):
                logits, values = self.model(observations)
                logits, values = logits.cpu(), values.cpu()

                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropies = (log_probs * probs).sum(1, keepdim=True)

                actions = probs.multinomial(num_samples=1).detach()
                log_probs_policy = log_probs.gather(1, actions)

                mem_values[step] = values
                mem_entropies[step] = entropies
                mem_log_probs[step] = log_probs_policy

                rewards = torch.zeros([len_workers, 1])
                non_terminals = torch.ones([len_workers, 1], dtype=torch.int8)
                observations = []

                for i in range(len_workers):
                    o, rewards[i, 0], t = workers[i].step(actions[i].item())
                    if t == True:
                        non_terminals[i, 0] = 0
                        o = workers[i].reset()
                    observations.append(torch.from_numpy(o).float())
                observations = torch.stack(observations).to(self.device)

                mem_rewards[step] = rewards
                mem_non_terminals[step] = non_terminals

            with torch.no_grad():
                _, R = self.model(observations)
                R = R.detach().cpu()

            advantages = torch.zeros([max_steps, len_workers, 1])

            for step in reversed(range(max_steps)):
                R = mem_rewards[step] + self.gamma * R * mem_non_terminals[step]
                advantages[step] = R - mem_values[step]

            advantages_detach = advantages.detach()
            #advantages_detach = (advantages_detach - torch.mean(advantages_detach)) / (torch.std(advantages_detach) + 1e-9)

            value_loss = (advantages**2).mean()
            policy_loss = - (mem_log_probs * advantages_detach).mean()
            entropy_loss = mem_entropies.mean()

            self.optimizer.zero_grad()
            loss = policy_loss + self.value_loss_coef * value_loss + self.beta_entropy * entropy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

            if iteration % 25 == 0 and iteration > 0:
                avg = np.average(self.average_score[-100:])
                if avg > best_avg:
                    best_avg = avg
                    print('saving model, best score is ', best_avg)
                    torch.save(self.model.state_dict(), 'models/' + self.name + '_' + str(self.id) + '_a2c.pt')
                print(iteration, '\tepisodes: ', self.episodes, '\taverage score: ', avg)
                if write:
                    text += '\n' + str(iteration) + ',' + str(self.episodes) + ',' + str(avg) + ',' + str(iter_step * iteration)

                    if iteration % 10000 == 0:
                        self.average_score = self.average_score[-100:]
                        write_to_file(text, 'logs/' + self.name + '_' + str(self.id) + '_' + str(iteration) + '_a2c.txt')
                        torch.save(self.model.state_dict(), 'models/' + self.name + '_' + str(self.id) + '_' + str(iteration) + '_a2c.pt')

    def load_model(self):
        self.model.load_state_dict(torch.load('models/' + self.name))

    def save_model(self):
        torch.save(self.model.state_dict(), 'models/' + self.name + '_' + str(self.id) + '_a2c.pt')

def reward_func(r):
    if r > 1:
        return 1
    elif r < -1:
        return -1
    return r

class Worker:
    def __init__(self, id, env, agent, print_score = False, reward_function = reward_func):
        self.id = id
        self.env = env

        self.print_score = print_score
        self.episode = 1
        self.state = None
        self.score = 0
        self.agent = agent
        self.reward_function = reward_function

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

        r = self.reward_function(r)

        return self.state, r, t
