import numpy as np
import math
import cmath

from complexnet import CNnet, convert_complex

lr_h = 0.0001
lr_o = 0.001

hidden_config = {
    'input_size': 2,
    'output_size': 2,
    'activation': 'tanh',
    'lr': lr_h
}

output_config = {
    'input_size': 2,
    'output_size': 1,
    'activation': 'linear',
    'lr': lr_o
}

layer_config = [hidden_config, output_config]

net = CNnet(layer_config)

input = np.array([-0.5, 1.0])
next_input = np.array([-0.45, 1.0])
target = 0.7 * net.outputs(next_input.astype(np.complex128)) * cmath.rect(1, math.pi/180.0)

'''
analytic calculation
'''
td_error = target - net.outputs(input.astype(np.complex128))
h_output = net.layers[0].f.output
v = net.layers[1].affine.weight
analytic_output_delta_bias = lr_o * td_error
analytic_output_delta_weight = np.dot(h_output[np.newaxis, :].conjugate().T, analytic_output_delta_bias[np.newaxis, :])
analytic_hidden_delta_bias = lr_h * convert_complex((1-h_output.real**2) * (td_error.real * v.T[0].real + td_error.imag * v.T[0].imag), (1-h_output.imag**2) * (td_error.imag * v.T[0].real - td_error.real * v.T[0].imag))
analytic_hidden_delta_weight = np.dot(input.astype(np.complex128)[np.newaxis, :].conjugate().T, analytic_hidden_delta_bias[np.newaxis, :])

'''
back propagation calculation
'''
net.train(input.astype(np.complex128), target, 'square_error')
back_propagate_output_delta_bias = net.layers[1].affine.delta_bias
back_propagate_output_delta_weight = net.layers[1].affine.delta_weight
back_propagate_hidden_delta_bias = net.layers[0].affine.delta_bias
back_propagate_hidden_delta_weight = net.layers[0].affine.delta_weight

print('output_delta_bias')
print(analytic_output_delta_bias)
print(back_propagate_output_delta_bias)
print('output_delta_weight')
print(analytic_output_delta_weight)
print(back_propagate_output_delta_weight)
print('hidden_delta_bias')
print(analytic_hidden_delta_bias)
print(back_propagate_hidden_delta_bias)
print('hidden_delta_weight')
print(analytic_hidden_delta_weight)
print(back_propagate_hidden_delta_weight)
