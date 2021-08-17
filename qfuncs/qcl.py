import numpy as np
from string import ascii_lowercase
import sys

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, execute, Aer, IBMQ

Z = np.array([[1,0],[0, -1]])
I = np.array([[1,0],[0, 1]]).astype(np.complex128)

def gate_base2(gate_matrix, ctl, tgt, bn):
    num_state = 2 ** bn
    gate = np.zeros((num_state, num_state), dtype=complex)
    for i in range(num_state):
        bit_c = int(int(format(i, 'b')) / 10**(bn - ctl - 1) % 2)
        if bit_c == 1:
            bit_t = int(int(format(i, 'b')) / 10**(bn - tgt - 1) % 2)
            bit_list = list(format(i, '0' + str(bn) + 'b'))
            bit_list[tgt] = '1' if bit_t == 0 else '0'
            idx = int("".join(bit_list), 2)
            if i < idx:
                gate[i, i] = gate_matrix[bit_t, 0]
                gate[i, idx] = gate_matrix[bit_t, 1]
            else:
                gate[i, i] = gate_matrix[bit_t, 1]
                gate[i, idx] = gate_matrix[bit_t, 0]
        else:
            gate[i, i] = 1.
    return gate

def get_gate_matrix(gatename, theta=None):
    if gatename=='rx':
        gate = np.array([[np.cos(theta / 2), -1.0j * np.sin(theta / 2)],
                         [-1.0j * np.sin(theta / 2), np.cos(theta / 2)]]).astype(np.complex128)
    elif gatename=='ry':
        gate = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                         [np.sin(theta / 2), np.cos(theta / 2)]]).astype(np.complex128)
    elif gatename=='rz':
        gate = np.array([[np.exp(-1j * theta / 2), 0],
                         [0, np.exp(1j * theta / 2)]])
    elif gatename=='cz':
        gate = np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., -1.]]).reshape(4 * [2]).astype(np.complex128)
    elif gatename=='z':
        gate = Z.astype(np.complex128)

    return gate

def get_gate_diff(gatename, theta):
    if gatename=='ry':
        gate = np.array([[-np.sin(theta / 2), -np.cos(theta / 2)],
                                   [np.cos(theta / 2), -np.sin(theta / 2)]]).astype(np.complex128) / 2
    elif gatename=='rz':
        gate = np.array([[-1j*np.exp(-theta*1j/2)/2, 0], [0, 1j*np.exp(theta*1j/2)/2]])
    return gate

def einsum_string(bn, *args):
    v_start = ascii_lowercase[:bn]
    v_end = list(v_start)
    m, v = '', ''
    for i in range(len(args)):
        m += ascii_lowercase[-1 - i]
#             v += v_start[-1 - args[i]]
        v += v_start[args[i]]
#             v_end[-1 - args[i]] = ascii_lowercase[-1 - i]
        v_end[args[i]] = ascii_lowercase[-1 - i]
    v_end = ''.join(v_end)
    return "{0}{1}, {2}->{3}".format(m, v, v_start, v_end)

# def convert_matrix(gate, num, bn):
#     if num == 0:
#         matrix = gate
#     else:
#         matrix = I
#     for i in range(1, bn):
#         if i == num:
#             matrix = np.kron(matrix, gate)
#         else:
#             matrix = np.kron(matrix, I)
#     return matrix

# def convert_matrix(gate, num, bn):
#     if num == 0:
#         matrix = gate
#     else:
#         matrix = I
#     n = 2
#     for i in range(1, bn):
#         n *= 2
#         if i == num:
#             matrix = np.einsum('ik,jl', matrix, gate).reshape(n, n)
#         else:
#             matrix = np.einsum('ik,jl', matrix, I).reshape(n, n)
#     return matrix

def convert_matrix(gate, num, bn):
    if num == 0 and bn == 1:
        matrix = gate
    elif num == 0 and bn != 1:
        matrix = np.einsum('ik,jl', gate, np.identity(2**(bn-1))).reshape(2**bn, 2**bn)
    elif num == 1 and bn == 2:
        matrix = np.einsum('ik,jl', np.identity(2), gate).reshape(4, 4)
    elif num == bn - 1:
        matrix = np.einsum('ik,jl', np.identity(2**(bn-1)), gate).reshape(2**bn, 2**bn)
    else:
        matrix = np.einsum('ik,jl', np.identity(2**num), gate).reshape(2**(num+1), 2**(num+1))
        matrix = np.einsum('ik,jl', matrix, np.identity(2**(bn-num-1))).reshape(2**bn, 2**bn)
    return matrix

class Gate():
    def __init__(self, gatename, num, bn, lr):
        self.gatename = gatename
        self.num = num
        self.bn = bn
        if self.gatename=='cz':
            self.ein_str = einsum_string(self.bn, self.num[0], self.num[1])
        else:
            self.ein_str = einsum_string(self.bn, self.num)
        if self.gatename!='cz' and self.gatename!='z':
            self.theta = 2 * np.pi * np.random.rand() - np.pi
            self.lr = lr

    def get_gate(self):
        if self.gatename=='cz' or self.gatename=='z':
            self.gate = get_gate_matrix(self.gatename)
        else:
            self.gate = get_gate_matrix(self.gatename, self.theta)

    def get_dw_dtheta(self):
        gate = get_gate_diff(self.gatename, self.theta)
        self.dw_dtheta = convert_matrix(gate, self.num, self.bn)

    def outputs(self, input):
        self.input = input
        self.get_gate()
        input = input.reshape(self.bn * [2])
        return np.einsum(self.ein_str, self.gate, input, dtype=complex, casting='no').reshape((2**self.bn, 1))

    def back_propagate(self, grad):
        if self.gatename!='cz' and self.gatename!='z':
            self.get_dw_dtheta()
            d_w = np.dot(grad, self.input.conjugate().T)
            self.d_theta = np.sum(d_w * self.dw_dtheta.conjugate()).real
        grad = grad.reshape(self.bn * [2])
        return np.einsum(self.ein_str, self.gate.conjugate().T, grad, dtype=complex, casting='no').reshape((2**self.bn, 1))

    def update(self):
        if self.gatename!='cz' and self.gatename!='z':
            self.theta += -1 * self.lr * self.d_theta

class Measurement():
    def __init__(self, observable, bn, lr, trainable, use_qiskit):
        self.gatename = observable
        self.bn = bn
        self.trainable = trainable
        if use_qiskit:
            self.uo_gates = []
        else:
            self.layers = []
        if observable[0] == 'u' or observable[0] == 'U':
            for i in range(self.bn):
                if observable[-1] == 'u' or observable[-1] == 'U' or str(i) in observable:
                    if use_qiskit:
                        self.uo_gates.append(['rz', i, 2 * np.pi * np.random.rand() - np.pi])
                        self.uo_gates.append(['ry', i, 2 * np.pi * np.random.rand() - np.pi])
                        self.uo_gates.append(['rz', i, 2 * np.pi * np.random.rand() - np.pi])
                    else:
                        self.layers.append(Gate('rz', i, self.bn, lr))
                        self.layers.append(Gate('ry', i, self.bn, lr))
                        self.layers.append(Gate('rz', i, self.bn, lr))
        elif observable[0] == 'z' or observable[0] == 'Z':
            for i in range(self.bn):
                if observable[-1] == 'z' or observable[-1] == 'Z' or str(i) in observable:
                    if use_qiskit:
                        self.uo_gates.append(['z', i])
                    else:
                        self.layers.append(Gate('z', i, self.bn, lr))
    '''
    def get_weight(self):
        if self.gatename[0] == 'z' or self.gatename[0] == 'Z':
            gate = np.array([[1, 0],
                             [0, -1]]).astype(np.complex128)
        elif self.gatename[0] == 'y' or self.gatename[0] == 'Y':
            gate = np.array([[0, -1j],
                             [1j, 0]])
        elif self.gatename[0] == 'u' or self.gatename[0] == 'U':
            theta1 = self.theta
            theta2 = self.theta * 2
            theta3 = self.theta * 3
            gate = np.dot(get_gate_matrix('ry', theta2), get_gate_matrix('rz', theta1))
            gate = np.dot(get_gate_matrix('rz', theta3), gate)
        if self.num == 0:
            self.weight = convert_matrix(gate, self.num, self.bn)
        else:
    '''

    def outputs(self, input):
        self.input = input
        state = input
        for layer in self.layers:
            state = layer.outputs(state)
        self.w_phi = state
        if self.gatename[0] == 'z' or self.gatename[0] == 'Z':
            return np.dot(np.conj(input.T), state).real
        else:
            return np.dot(np.conj(input.T), state)

    def back_propagate(self, grad):
        g = np.dot(self.input, grad)
        for layer in reversed(self.layers):
            g = layer.back_propagate(g)
        g_conj = np.dot(grad, np.conj(self.w_phi.T)).T
        return g + (g_conj.real - 1j * g_conj.imag)

    def update(self):
        if not self.trainable:
            return None
        for layer in self.layers:
            layer.update()

class QuantumCircuitSimulator():
    def __init__(self, qubit, input_dim, depth, lr, gamma, trainable_output, use_qiskit, qiskit_backend, shot_num, observable='u', binary_encoding=False):
        self.bn = qubit
        self.input_dim = input_dim
        self.depth = depth
        self.gamma = gamma
        self.binary_encoding = binary_encoding
        self.use_qiskit = use_qiskit
        # preparation
        if self.use_qiskit:
            if qiskit_backend == 'qasm_simulator':
                self.qiskit_backend = Aer.get_backend(qiskit_backend)
            else:
                IBMQ.load_account()
                provider = IBMQ.get_provider(group='your group name')
                self.qiskit_backend = provider.backends(qiskit_backend)[0]
            self.shot_num = shot_num
            self.uw_gates = []
        else:
            self.init_state = np.zeros(2 ** self.bn).reshape(2 ** self.bn, 1).astype(np.complex128)
            self.init_state[0] = 1
            self.layers = []
        # set parameter and entangle gates
        for i in range(self.bn):
            if self.use_qiskit:
                self.uw_gates.append(['ry', i, 2 * np.pi * np.random.rand() - np.pi])
                self.uw_gates.append(['rz', i, 2 * np.pi * np.random.rand() - np.pi])
            else:
                self.layers.append(Gate('ry', i, self.bn, lr))
                self.layers.append(Gate('rz', i, self.bn, lr))
        for _ in range(1, depth):
            for i in range(self.bn):
                if self.use_qiskit:
                    self.uw_gates.append(['cz', [i, (i+1)%self.bn]])
                else:
                    self.layers.append(Gate('cz', [i, (i+1)%self.bn], self.bn, lr))
            for i in range(self.bn):
                if self.use_qiskit:
                    self.uw_gates.append(['ry', i, 2 * np.pi * np.random.rand() - np.pi])
                    self.uw_gates.append(['rz', i, 2 * np.pi * np.random.rand() - np.pi])
                else:
                    self.layers.append(Gate('ry', i, self.bn, lr))
                    self.layers.append(Gate('rz', i, self.bn, lr))
        # set output gates
        measurement = Measurement(observable, self.bn, lr, trainable_output, use_qiskit)
        if self.use_qiskit:
            self.uo_gates = measurement.uo_gates
        else:
            self.layers.append(measurement)

    def encode_input(self, input):
        if self.binary_encoding:
            binary_input = bin(input)[2:].zfill(self.bn)
            if not self.use_qiskit:
                state = self.init_state.reshape(self.bn * [2])
            gate_x_0 = 0.0 if self.use_qiskit else get_gate_matrix('rx', 0.0)
            gate_z_0 = 0.0 if self.use_qiskit else get_gate_matrix('rz', 0.0)
            gate_x_pi = np.pi if self.use_qiskit else get_gate_matrix('rx', np.pi)
            gate_z_pi = np.pi if self.use_qiskit else get_gate_matrix('rz', np.pi)
            for i in range(self.bn):
                if binary_input[i] == '1':
                    gate_x = gate_x_pi
                    gate_z = gate_z_pi
                else:
                    gate_x = gate_x_0
                    gate_z = gate_z_0
                if self.use_qiskit:
                    self.qc.rx(gate_x, self.qr[i])
                    self.qc.rz(gate_z, self.qr[i])
                else:
                    ein_str = einsum_string(self.bn, i)
                    state = np.einsum(ein_str, gate_x, state, dtype=complex, casting='no')
                    state = np.einsum(ein_str, gate_z, state, dtype=complex, casting='no')
            if not self.use_qiskit:
                return state.reshape((2**self.bn, 1))
        else:
            if self.input_dim == 1:
                angle_y = np.arcsin(input)
                angle_z = np.arccos(input**2)
            elif self.input_dim == 2:
                angle_y = np.arcsin(input[0])
                angle_z = np.arccos(input[1])
            if not self.use_qiskit:
                gate_y = get_gate_matrix('ry', angle_y)
                gate_z = get_gate_matrix('rz', angle_z)
                state = self.init_state.reshape(self.bn * [2])
            for i in range(self.bn):
                if self.use_qiskit:
                    self.qc.ry(angle_y, self.qr[i])
                    self.qc.rz(angle_z, self.qr[i])
                else:
                    ein_str = einsum_string(self.bn, i)
                    state = np.einsum(ein_str, gate_y, state, dtype=complex, casting='no')
                    state = np.einsum(ein_str, gate_z, state, dtype=complex, casting='no')
            if not self.use_qiskit:
                return state.reshape((2**self.bn, 1))

    def get_output_state(self):
        for uw_gate in self.uw_gates:
            if uw_gate[0] == 'ry':
                self.qc.ry(uw_gate[2], self.qr[uw_gate[1]])
            elif uw_gate[0] == 'rz':
                self.qc.rz(uw_gate[2], self.qr[uw_gate[1]])
            elif uw_gate[0] == 'cz':
                self.qc.cz(self.qr[uw_gate[1][0]], self.qr[uw_gate[1][1]])

    def hadamard_test(self, part, qr_had):
        self.qc.h(qr_had[0])
        if part == 'imag':
            self.qc.s(qr_had[0])
        for uo_gate in self.uo_gates:
            if uo_gate[0] == 'ry':
                self.qc.cry(uo_gate[2], qr_had[0], self.qr[uo_gate[1]])
            elif uo_gate[0] == 'rz':
                self.qc.crz(uo_gate[2], qr_had[0], self.qr[uo_gate[1]])
        self.qc.h(qr_had[0])
        self.qc.measure(qr_had[0], self.cr[0])

    def get_exp_val(self, input, part):
        self.qr = QuantumRegister(self.bn)
        # for Hadamard test
        qr_had = QuantumRegister(1)
        self.cr = ClassicalRegister(1)
        self.qc = QuantumCircuit(self.qr, qr_had, self.cr)
        # encode input
        self.encode_input(input)
        # get output state
        self.get_output_state()
        # get real or imaginary part
        self.hadamard_test(part, qr_had)
        # run circuit
        result = execute(self.qc, self.qiskit_backend, shots=self.shot_num).result()
        # get counts of 0 or 1
        counts = result.get_counts(self.qc)
        # calculate probability of 0 or 1
        if '1' not in counts:
            p_1 = 0.
        else:
            p_1 = counts['1'] / self.shot_num
        if '0' not in counts:
            p_0 = 0.
        else:
            p_0 = counts['0'] / self.shot_num
        # get real or imaginary part of expectation value
        if part == 'real':
            return p_0 - p_1
        elif part == 'imag':
            return p_1 - p_0

    def get_output(self, input_state):
        state = input_state
        for layer in self.layers:
            state = layer.outputs(state)
        return state

    def outputs(self, input):
        if self.use_qiskit:
            real_part = self.get_exp_val(input, 'real')
            imag_part = self.get_exp_val(input, 'imag')
            return [[(real_part + 1j * imag_part) * self.gamma]]
        else:
            input_state = self.encode_input(input)
            return self.get_output(input_state) * self.gamma


    def update(self, input, target, loss_type):
        pred = self.outputs(input)
        if loss_type == 'square_error':
            grad = (pred - target) * self.gamma
        for layer in reversed(self.layers):
            grad = layer.back_propagate(grad)
            layer.update()

    def get_params(self):
        theta = []
        if self.use_qiskit:
            for uw_gate in self.uw_gates:
                if uw_gate[0] != "cz":
                    theta.append(uw_gate[2])
        else:
            for g in self.layers[:-1]:
                if g.gatename != "cz":
                    theta.append(g.theta)
        theta_o = []
        if self.use_qiskit:
            for uo_gate in self.uo_gates:
                if uo_gate[0] != "z":
                    theta_o.append(uo_gate[2])
        else:
            for g in self.layers[-1].layers:
                if g.gatename != "z":
                    theta_o.append(g.theta)
        return [theta, theta_o]

    def set_params(self, params):
        theta = params[0]
        theta_o = params[1]
        n = 0
        if self.use_qiskit:
            for uw_gate in self.uw_gates:
                if uw_gate[0] != "cz":
                    uw_gate[2] = theta[n]
                    n += 1
        else:
            for g in self.layers[:-1]:
                if g.gatename != "cz":
                    g.theta = theta[n]
                    n += 1
        n = 0
        if self.use_qiskit:
            for uo_gate in self.uo_gates:
                if uo_gate[0] != "z":
                    uo_gate[2] = theta_o[n]
                    n += 1
        else:
            for g in self.layers[-1].layers:
                if g.gatename != "z":
                    g.theta = theta_o[n]
                    n += 1

class MultiOutputQuantumCircuit(QuantumCircuitSimulator):
    """
    Multiple outputs Quantum Circuit class
    """

    def __init__(self, qubit, input_dim, depth, lr, gamma, trainable_output, use_qiskit, qiskit_backend, shot_num, binary_encoding, output_size, observable):
        super().__init__(qubit, input_dim, depth, lr, gamma, trainable_output, use_qiskit, qiskit_backend, shot_num, binary_encoding=binary_encoding)
        if self.use_qiskit:
            self.uo_gates_list = []
        else:
            self.delete_output_layer()
            self.output_layers = []
        if observable == 'z' or observable == 'Z':
            if qubit < output_size:
                print('the number of qubit must be greater than or equal to output size')
                sys.exit()
            for i in range(output_size):
                observable = 'z{}'.format(i)
                measurement = Measurement(observable, self.bn, lr, trainable_output, use_qiskit)
                if self.use_qiskit:
                    self.uo_gates_list.append(measurement.uo_gates)
                else:
                    self.output_layers.append(measurement)
        elif observable == 'u' or observable == 'U':
            if qubit < output_size + 1:
                print('the number of qubit must be greater than or equal to output size + 1')
                sys.exit()
            for i in range(1, output_size + 1):
                observable = 'u0{}'.format(i)
                measurement = Measurement(observable, self.bn, lr, trainable_output, use_qiskit)
                if self.use_qiskit:
                    self.uo_gates_list.append(measurement.uo_gates)
                else:
                    self.output_layers.append(measurement)

    def set_output_layer(self, n):
        if self.use_qiskit:
            self.uo_gates = self.uo_gates_list[n]
        else:
            self.layers.append(self.output_layers[n])

    def delete_output_layer(self):
        self.layers.pop()

    def outputs(self, input, n):
        self.set_output_layer(n)
        output = super().outputs(input)
        if not self.use_qiskit:
            self.delete_output_layer()
        return output

    def update(self, input, n, target, loss_type):
        self.set_output_layer(n)
        pred = super().outputs(input)
        if loss_type == 'square_error':
            grad = (pred - target) * self.gamma
        for layer in reversed(self.layers):
            grad = layer.back_propagate(grad)
            layer.update()
        self.delete_output_layer()

    def get_params(self):
        theta = []
        if self.use_qiskit:
            for uw_gate in self.uw_gates:
                if uw_gate[0] != "cz":
                    theta.append(uw_gate[2])
        else:
            for g in self.layers:
                if g.gatename != "cz":
                    theta.append(g.theta)
        theta_o = []
        if self.use_qiskit:
            for uo_gates in self.uo_gates_list:
                for uo_gate in uo_gates:
                    if uo_gate[0] != "z":
                        theta_o.append(uo_gate[2])
        else:
            for o in self.output_layers:
                for g in o.layers:
                    if g.gatename != "z":
                        theta_o.append(g.theta)
        return [theta, theta_o]

    def set_params(self, params):
        theta = params[0]
        theta_o = params[1]
        n = 0
        if self.use_qiskit:
            for uw_gate in self.uw_gates:
                if uw_gate[0] != "cz":
                    uw_gate[2] = theta[n]
                    n += 1
        else:
            for g in self.layers:
                if g.gatename != "cz":
                    g.theta = theta[n]
                    n += 1
        n = 0
        if self.use_qiskit:
            for uo_gates in self.uo_gates_list:
                for uo_gate in uo_gates:
                    if uo_gate[0] != "z":
                        uo_gate[2] = theta_o[n]
                        n += 1
        else:
            for o in self.output_layers:
                for g in o.layers:
                    if g.gatename != "z":
                        g.theta = theta_o[n]
                        n += 1
