import numpy as np

class RBF():
    """
    RBF unit class
    """

    def __init__(self, input_size, mu_lr, sigma_lr):
        self.mu = np.zeros(input_size)
        self.sigma = np.ones(input_size)
        self.mu_lr = mu_lr
        self.sigma_lr = sigma_lr

    def outputs(self, input):
        return np.prod(np.exp(-1 * (input - self.mu)**2  / self.sigma**2))

    def update(self, input, td_error, weight):
        delta_mu = self.mu_lr * td_error * weight * ((input - self.mu) / self.sigma) * self.outputs(input)
        delta_sigma = self.sigma_lr * td_error * weight * ((input - self.mu)**2 / self.sigma**3) * self.outputs(input)
        self.mu += delta_mu
        self.sigma += delta_sigma

class RBFNet():
    """
    RBF network class
    Kernel function is RBF
    """

    def __init__(self, input_size, hidden_size, weight_lr, mu_lr, sigma_lr):
        self.hidden_size = hidden_size
        self.hidden_layer = []
        for _ in range(self.hidden_size):
            self.hidden_layer.append(RBF(input_size, mu_lr, sigma_lr))
        self.weight = np.random.randn(hidden_size)
        self.weight_lr = weight_lr

    def hidden_output(self, input):
        return np.array([rbf.outputs(input) for rbf in self.hidden_layer])

    def outputs(self, input):
        hidden_output = self.hidden_output(input)
        return np.dot(hidden_output, self.weight)

    def update(self, input, td_error):
        delta_weight = self.weight_lr * td_error * self.hidden_output(input)
        [rbf.update(input, td_error, self.weight[n]) for n, rbf in enumerate(self.hidden_layer)]
        self.weight += delta_weight

# rbfnet = RBFNet(4, 10, 0.001, 0.001, 0.001)
#
# import gym
#
# env = gym.make('Acrobot-v1')
# env.reset()
# env.env.state = np.array([1.0, 1.0, 1.0, 1.0])
#
# theta1 = env.env.state[0]
# theta2 = env.env.state[1]
# thetadot1 = env.env.state[2]
# thetadot2 = env.env.state[3]
# input = np.array([theta1, theta2, thetadot1, thetadot2])
# a = rbfnet.outputs(input)
# print('weight')
# print(rbfnet.weight)
# print('input')
# print(input)
# print(rbfnet.hidden_output(input))
# print(a)
# print(np.sum(rbfnet.weight) * np.exp(-4))
