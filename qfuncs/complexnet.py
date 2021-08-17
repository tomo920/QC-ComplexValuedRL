import numpy as np

initialization = 'random'
init_max = 0.3
init_min = -0.3

def convert_complex(real_part, imag_part):
    return real_part.astype(np.complex128) + 1j * imag_part.astype(np.complex128)

class Sigmoid():
    '''
    Sigmoid activation function
    '''

    def __init__(self):
        self.output = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def outputs(self, input):
        output_real = self.sigmoid(input.real)
        output_imag = self.sigmoid(input.imag)
        self.output = convert_complex(output_real, output_imag)
        return self.output

    def back_propagate(self, grad):
        back_real = grad.real * (1 - self.output.real) * self.output.real
        back_imag = grad.imag * (1 - self.output.imag) * self.output.imag
        return convert_complex(back_real, back_imag)

class Tanh():
    '''
    Hyperbolic tangent activation function
    '''

    def __init__(self):
        self.output = None

    def tanh(self, x):
        return np.tanh(x)

    def outputs(self, input):
        output_real = self.tanh(input.real)
        output_imag = self.tanh(input.imag)
        self.output = convert_complex(output_real, output_imag)
        return self.output

    def back_propagate(self, grad):
        back_real = grad.real * (1 - self.output.real ** 2)
        back_imag = grad.imag * (1 - self.output.imag ** 2)
        return convert_complex(back_real, back_imag)

class Linear():
    '''
    Linear activation function
    '''

    def __init__(self):
        self.output = None

    def outputs(self, input):
        return input

    def back_propagate(self, grad):
        return grad

class Affine():
    """
    Affine transformation
    y = Wx + b
    """

    def __init__(self, input_size, output_size, lr):
        if initialization == 'random':
            weight_real = (init_max - init_min) * np.random.rand(input_size, output_size) + init_min
            weight_imag = (init_max - init_min) * np.random.rand(input_size, output_size) + init_min
            bias_real = (init_max - init_min) * np.random.rand(output_size) + init_min
            bias_imag = (init_max - init_min) * np.random.rand(output_size) + init_min
        self.weight = convert_complex(weight_real, weight_imag)
        self.bias = convert_complex(bias_real, bias_imag)
        self.lr = lr
        self.input = None
        self.delta_weight = None
        self.delta_bias = None

    def outputs(self, input):
        self.input = input
        return np.dot(input, self.weight) + self.bias

    def back_propagate(self, grad):
        self.delta_bias = -1 * self.lr * grad
        self.delta_weight = np.dot(self.input[np.newaxis, :].conjugate().T, self.delta_bias[np.newaxis, :])
        return np.dot(grad, self.weight.conjugate().T)

    def update(self):
        self.weight += self.delta_weight
        self.bias += self.delta_bias

class Layer():
    '''
    Dense Layer
    '''

    def __init__(self, config):
        self.affine = Affine(config['input_size'], config['output_size'], config['lr'])
        if config['activation'] == 'sigmoid':
            self.f = Sigmoid()
        elif config['activation'] == 'tanh':
            self.f = Tanh()
        elif config['activation'] == 'linear':
            self.f = Linear()
        self.trainable = config['trainable']

    def outputs(self, input):
        return self.f.outputs(self.affine.outputs(input))

    def back_propagate(self, grad):
        grad = self.f.back_propagate(grad)
        return self.affine.back_propagate(grad)

    def update(self):
        if not self.trainable:
            return None
        self.affine.update()

class CNnet():
    '''
    Complex valued neural network
    '''

    def __init__(self, layer_config):
        self.layers = []
        for config in layer_config:
            self.layers.append(Layer(config))

    def outputs(self, input):
        for layer in self.layers:
            input = layer.outputs(input)
        return input

    def train(self, input, target, loss_type):
        pred = self.outputs(input)
        if loss_type == 'square_error':
            grad = pred - target
        for layer in reversed(self.layers):
            grad = layer.back_propagate(grad)
            layer.update()

    def get_params(self):
        params = []
        for layer in self.layers:
            params.append([layer.affine.weight, layer.affine.bias])
        return params

    def set_params(self, params):
        for param, layer in zip(params, self.layers):
            layer.affine.weight = param[0]
            layer.affine.bias = param[1]
