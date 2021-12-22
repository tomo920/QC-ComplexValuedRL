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
# Run
To run Complex-valued reinforcement learning:
* Table

```
python main.py po_maze qdot_learning 0.9 1000 "save_dir_name" --rotation_angle 6 --alpha 0.25 --n_episodes 5000
```

* Neural Network

```
python main.py po_maze qdot_learning_nn 0.7 1000 "save_dir_name" --rotation_angle 180 --is_continuous --action_encoding --n_episodes 5000
```

* Quantum Circuit

```
python main.py po_maze qdot_learning_qc 0.7 1000 "save_dir_name" --rotation_angle 180 --binary_encoding --qubit 6 --depth 3 --gamma_q 6 --lr_q 0.001 --n_episodes 5000
```

To run RNN:

```
python main.py po_maze q_learning_lstm 0.9 1000 "save_dir_name" --is_continuous --n_episodes 5000 --lr_o 0.00001
```
