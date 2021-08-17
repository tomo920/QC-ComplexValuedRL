import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__))

from qdot_base import QdotBase
from qcl import QuantumCircuitSimulator, MultiOutputQuantumCircuit

class ComplexQuantumCircuit():
    """
    Quantum Circuit class for representing complex value
    """

    def __init__(self, qubit, input_dim, depth, lr_q, gamma_q, trainable_output, use_qiskit, qiskit_backend, shot_num):
        self.real_part = QuantumCircuitSimulator(qubit, input_dim, depth, lr_q, gamma_q, trainable_output, use_qiskit, qiskit_backend, shot_num, observable='z0')
        self.imaginary_part = QuantumCircuitSimulator(qubit, input_dim, depth, lr_q, gamma_q, trainable_output, use_qiskit, qiskit_backend, shot_num, observable='z0')

    def outputs(self, input):
        real = self.real_part.outputs(input)
        imaginary = self.imaginary_part.outputs(input)
        return real.astype(np.complex128) + 1j * imaginary.astype(np.complex128)

    def update(self, input, target, loss_type):
        target_real = target.real
        target_imag = target.imag
        self.real_part.update(input, target_real, loss_type)
        self.imaginary_part.update(input, target_imag, loss_type)

    def get_params(self):
        real_part_params = self.real_part.get_params()
        imaginary_part_params = self.imaginary_part.get_params()
        return [real_part_params, imaginary_part_params]

    def set_params(self, params):
        real_part_params = params[0]
        imaginary_part_params = params[1]
        self.real_part.set_params(real_part_params)
        self.imaginary_part.set_params(imaginary_part_params)

class ComplexMultiOutputQuantumCircuit():
    """
    Quantum Circuit class for representing complex value
    """

    def __init__(self, qubit, input_dim, depth, lr_q, gamma_q, trainable_output, binary_encoding, output_size, use_qiskit, qiskit_backend, shot_num):
        self.real_part = MultiOutputQuantumCircuit(qubit, input_dim, depth, lr_q, gamma_q, trainable_output, use_qiskit, qiskit_backend, shot_num, binary_encoding, output_size, 'z')
        self.imaginary_part = MultiOutputQuantumCircuit(qubit, input_dim, depth, lr_q, gamma_q, trainable_output, use_qiskit, qiskit_backend, shot_num, binary_encoding, output_size, 'z')

    def outputs(self, input, n):
        real = self.real_part.outputs(input, n)
        imaginary = self.imaginary_part.outputs(input, n)
        return real.astype(np.complex128) + 1j * imaginary.astype(np.complex128)

    def update(self, input, n, target, loss_type):
        target_real = target.real
        target_imag = target.imag
        self.real_part.update(input, n, target_real, loss_type)
        self.imaginary_part.update(input, n, target_imag, loss_type)

    def get_params(self):
        real_part_params = self.real_part.get_params()
        imaginary_part_params = self.imaginary_part.get_params()
        return [real_part_params, imaginary_part_params]

    def set_params(self, params):
        real_part_params = params[0]
        imaginary_part_params = params[1]
        self.real_part.set_params(real_part_params)
        self.imaginary_part.set_params(imaginary_part_params)

class Qfunc(QdotBase):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.action_size = env.action_size
        self.action_list = env.action_list
        if config.binary_encoding:
            if config.is_continuous:
                print('binary encoding can only be used in discrete environment')
                sys.exit()
            if config.action_encoding:
                print('binary encoding cannot be used with encoding action')
                sys.exit()
        if config.use_qiskit and config.mode == 'learn':
            print('learning in circuit by qiskit is not supported now')
            sys.exit()
        if config.action_encoding:
            if config.divide_complex_number:
                self.qc = ComplexQuantumCircuit(config.qubit, 2, config.depth, config.lr_q, config.gamma_q,
                                                not config.not_trainable_output_layer, config.use_qiskit, qiskit_backend, shot_num)
            else:
                self.qc = QuantumCircuitSimulator(config.qubit, 2, config.depth, config.lr_q, config.gamma_q,
                                                  not config.not_trainable_output_layer, config.use_qiskit, config.qiskit_backend, config.shot_num)
        else:
            if config.divide_complex_number:
                self.qc = ComplexMultiOutputQuantumCircuit(config.qubit, 1, config.depth, config.lr_q, config.gamma_q,
                                                           not config.not_trainable_output_layer, config.binary_encoding, self.action_size, config.use_qiskit, qiskit_backend, shot_num)
            else:
                self.qc = MultiOutputQuantumCircuit(config.qubit, 1, config.depth, config.lr_q, config.gamma_q,
                                                    not config.not_trainable_output_layer, config.use_qiskit, config.qiskit_backend, config.shot_num, config.binary_encoding, self.action_size, 'u')

    def get_q_o_a(self, observation, action):
        if self.config.action_encoding:
            action = np.array([self.action_list[action]])
            input = np.concatenate([observation, action])
            return self.qc.outputs(input)[0]
        else:
            return self.qc.outputs(observation[0], action)[0]

    def get_q_o(self, observation):
        return np.concatenate([self.get_q_o_a(observation, action) for action in range(self.action_size)])

    def update_q(self, observation, action, q_target):
        if self.config.action_encoding:
            action = np.array([self.action_list[action]])
            input = np.concatenate([observation, action])
            self.qc.update(input, q_target, 'square_error')
        else:
            self.qc.update(observation[0], action, q_target, 'square_error')

    def get_params(self):
        return self.qc.get_params()

    def set_params(self, params):
        self.qc.set_params(params)
