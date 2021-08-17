import numpy as np
from numpy import pi
import cmath
import torch
from torch.autograd import Variable

import importlib

class Agent():
    '''
    Agent class.
    Algorithm is Q-learning(table) or Qdot-learning(table) or
                 Q-learning(RBF) or Qdot-learning(RBF) or
                 Qdot-learning(NN) or Qdot-learning(QC) or
                 Q-learning(LSTM) or Random
    '''

    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.alg_module = importlib.import_module('qfuncs.{}'.format(config.algorithm))
        self.alg_type = config.algorithm.split('_')[0]
        if self.alg_type == 'qdot':
            self.beta = cmath.rect(1, pi/config.rotation_angle)
        if self.config.algorithm.split('_')[-1] == 'lstm':
            if self.config.is_gpu:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    print('GPU is not available')
                    sys.exit()
            else:
                self.device = torch.device("cpu")

    def reset_q_func(self):
        self.q_func = self.alg_module.Qfunc(self.config, self.env)
        if self.config.policy_type == 'epsilon_greedy':
            self.epsilons = np.linspace(self.config.epsilon_start, self.config.epsilon_end, self.config.epsilon_decay_steps)

    def get_policy(self, observation, i_episode):
        if self.alg_type == 'qdot':
            q = self.q_func.get_effective_q(observation, self.ir_value)
        elif self.config.algorithm.split('_')[-1] == 'lstm':
            input_seq = torch.Tensor([[[self.a_history[-1], observation[0]]]], device=self.device)
            with torch.no_grad():
                q = self.q_func.get_effective_q(self.hidden_state, input_seq)[0][0].detach().numpy().copy()
        else:
            q = self.q_func.get_effective_q(observation)
        if self.config.save_log:
            self.q_history.append(q)
            self.ir_history.append(self.ir_value)
        if self.config.policy_type == 'epsilon_greedy':
            epsilon = self.epsilons[min(i_episode, self.config.epsilon_decay_steps-1)]
            if self.config.is_legal_action:
                a_size = len(self.env.legal_action_list[observation])
            else:
                a_size = self.env.action_size
            pi = np.ones(a_size, dtype='float32') * epsilon / a_size
            max_action = np.argmax(q)
            pi[max_action] += 1.0-epsilon
            return pi
        elif self.config.policy_type == 'boltzmann':
            '''
            boltzmann policy
            '''
            if self.config.is_t_change:
                T = self.config.boltzmann_t / (1 + i_episode)
            else:
                T = self.config.boltzmann_t
            Q = [Q / T for Q in q]
            Q_max = np.max(Q)
            e_Q_imp = np.exp(Q-Q_max)
            return e_Q_imp / np.sum(e_Q_imp)

    def init_history(self, observation):
        if self.alg_type == 'qdot':
            # initialize internal reference value
            q = self.q_func.get_q(observation)
            a = np.argmax([abs(Q) for Q in q])
            if self.config.is_legal_action:
                a = self.env.legal_action_list[observation][a]
            self.ir_value = q[a]
        # initialize history
        self.o_history = []
        self.a_history = []
        if self.config.save_log:
            self.q_history = []
            self.ir_history = []
            self.q_func.qdot_history = []
            self.transition_history = []
        if self.config.algorithm.split('_')[-1] == 'lstm':
            self.h_history = []     # hidden_state history
            self.r_history = []     # reward history
            self.d_history = []     # done history
            # initialize hidden_state to zero
            init_hidden_h = torch.zeros(1, 1, self.config.hidden_size).float()
            init_hidden_c = torch.zeros(1, 1, self.config.hidden_size).float()
            self.init_hidden_state = (init_hidden_h, init_hidden_c)
            self.hidden_state = self.init_hidden_state
            # set action to no operation
            self.a_history.append(-1.0)

    def update_history(self, observation, action):
        self.o_history.append(observation)
        self.a_history.append(action)

    def update_ir_value(self, observation, action):
        self.ir_value = self.q_func.get_q_o(observation)[action] / self.beta

    def update_hidden_state(self, hidden_state, action, observation):
        input_seq = torch.Tensor([[[action, observation[0]]]], device=self.device)
        with torch.no_grad():
            self.hidden_state = self.q_func.get_hidden_state(hidden_state, input_seq)

    def update_q_func(self, next_observation, reward, done):
        if self.alg_type == 'qdot':
            effective_q_ = self.q_func.get_effective_q(next_observation, self.ir_value)
            max_index = np.argmax(effective_q_)
            max_q_ = self.q_func.get_q(next_observation)[max_index]
        elif self.config.algorithm.split('_')[-1] == 'lstm':
            target_a_seq = np.array(self.a_history[-(self.config.trace_num+1):]).reshape([-1, 1])
            target_o_seq = np.array(self.o_history[-self.config.trace_num:]+[[next_observation[0]]])
            target_input_seq = torch.Tensor([np.concatenate([target_a_seq, target_o_seq], axis=1)], device=self.device)
            with torch.no_grad():
                q_ = self.q_func.get_effective_q(self.init_hidden_state, target_input_seq)[1:].detach().numpy().copy()
                max_q_ = np.max(q_, 2).reshape(-1)
            reward = np.array(self.r_history[-self.config.trace_num:])
            done = np.array(self.d_history[-self.config.trace_num:])
            input_seq = []
            action_seq = []
        else:
            q_ = self.q_func.get_effective_q(next_observation)
            max_q_ = np.max(q_)
        q_target = reward + self.config.gamma * max_q_ * (1.0-done)
        for k in range(self.config.trace_num):
            if len(self.o_history) < k+1:
                break
            observation = self.o_history[-(k+1)]
            action = self.a_history[-(k+1)]
            if self.alg_type == 'qdot':
                q_target *= self.beta
            if self.config.algorithm.split('_')[-1] != 'lstm':
                self.q_func.update_q(observation, action, q_target)
        if self.config.algorithm.split('_')[-1] == 'lstm':
            input_seq = target_input_seq[:,:-1,:]
            if len(self.a_history) <= self.config.trace_num:
                action_seq = torch.Tensor([np.array(self.a_history[1:]).reshape([-1, 1]).astype('int')], device=self.device).to(torch.long)
            else:
                action_seq = torch.Tensor([np.array(self.a_history[-self.config.trace_num:]).reshape([-1, 1]).astype('int')], device=self.device).to(torch.long)
            target_seq = torch.Tensor([np.array(q_target).reshape([-1, 1])], device=self.device)
            self.q_func.update_q(input_seq, action_seq, target_seq)

    def update(self, observation, action, next_observation, reward, done):
        if self.config.save_log:
            self.transition_history.append([observation, action, next_observation, reward, done])
        # update history
        self.update_history(observation, action)
        if self.config.algorithm.split('_')[-1] == 'lstm':
            self.r_history.append(reward)
            self.d_history.append(done)
        # update Q network
        if self.config.mode == 'learn':
            # update internal reference value
            if self.alg_type == 'qdot':
                self.update_ir_value(observation, action)
            self.update_q_func(next_observation, reward, done)
        # update internal reference valuue
        if self.alg_type == 'qdot':
            # update internal reference valuue
            self.update_ir_value(observation, action)
        # update hidden state
        if self.config.algorithm.split('_')[-1] == 'lstm':
            self.update_hidden_state(self.hidden_state, self.a_history[-2], observation)

    def get_params(self):
        return self.q_func.get_params()

    def set_params(self, params):
        self.q_func.set_params(params)

    def get_info(self):
        ir_value = self.ir_value
        o_history = self.o_history
        a_history = self.a_history
        if self.config.save_log:
            q_history, ir_history, qdot_history, transition_history = self.get_log()
            return [ir_value, o_history, a_history, q_history, ir_history, qdot_history, transition_history]
        else:
            return [ir_value, o_history, a_history]

    def load_info(self, agent_info):
        if self.config.save_log:
            ir_value, o_history, a_history, q_history, ir_history, qdot_history, transition_history = agent_info
        else:
            ir_value, o_history, a_history = agent_info
        self.ir_value = ir_value
        self.o_history = o_history
        self.a_history = a_history
        if self.config.save_log:
            self.q_history = q_history
            self.ir_history = ir_history
            self.q_func.qdot_history = qdot_history
            self.transition_history = transition_history

    def get_log(self):
        q_history = self.q_history
        ir_history = self.ir_history
        qdot_history = self.q_func.qdot_history
        transition_history = self.transition_history
        return [q_history, ir_history, qdot_history, transition_history]
