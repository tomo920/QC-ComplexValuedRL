# Complex Valued Reinforcement Learning with Quantum Circuit
The implementation of complex valued reinforcement learning method using quantum circuit.

This repository is the code for the paper "Variational quantum circuit based reinforcement learning for POMDP and experimental implementation".

If you have any comments, feel free to contact me by email tomoakikimura95710920@gmail.com.

# Requirements
This repository uses these libraries.

* numpy
* qiskit
* pytorch
* gym

# Methods
This repository implements these methods for POMDP problems.

* Complex-valued reinforcement learning
    * Qdot Table
    * Qdot Neural Network
    * Qdot Quantum Circuit
* RNN
## Complex-valued reinforcement learning
Value function is complex value and updated by learning algorithm that extends normal Q-learning in consideration of phase. This complex-valued value function is represented by below methods.
* Table
* Neural Network
* Quantum Circuit
## RNN
Value function is represented by recurrent neural network to use history.
