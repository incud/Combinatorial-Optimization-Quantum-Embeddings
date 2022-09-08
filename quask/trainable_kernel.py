import copy

import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from quask.template_pennylane import ltfim_ansatz
from quask.combinatorial_kernel import *


class TrainableKernel:

    def __init__(self, X_train, y_train, X_validation, y_validation, n_layers, initial_solution):

        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_validation
        self.y_validation = y_validation
        self.n_qubits = X_train.shape[1]
        self.n_layers = n_layers
        self.n_gates = 2 * self.n_qubits * self.n_layers
        paulis = initial_solution[:, 0].ravel()
        operations = jnp.arange(self.n_gates).ravel()
        self.initial_solution = jnp.stack([paulis, operations]).T
        self.state = jnp.array(np.random.normal(size=(self.n_qubits * self.n_layers,)))
        self.trainable_kernel = self.create_pennylane_function()
        self.history_losses = []
        self.history_grads = []
        self.history_params = []
        self.gate_removed = []

    # def get_embedding(self, x, weights, bandwidth):
    #     # assert len(weights) == 3 * self.trainable_layers
    #     for i in range(self.trainable_layers):
    #         # encode trainable parameters
    #         ltfim_ansatz(bandwidth * weights[i * 3:(i + 1) * 3], wires=range(self.n_qubits))
    #         # encode data
    #         for j in range(self.n_qubits // 3):
    #             ltfim_ansatz(bandwidth * x[j * 3, (j + 1) * 3], wires=range(self.n_qubits))
    #         remaining = self.n_qubits % 3
    #         if remaining != 0:
    #             params = jnp.pad(bandwidth * x[self.n_qubits - remaining], (0, remaining), 'constant')
    #             ltfim_ansatz(params, wires=range(self.n_qubits))

    def overlap_parameters(self, x, weights):
        params = []
        for i in range(self.n_layers):
            params.append(weights[i * self.n_qubits : (i + 1) * self.n_qubits].ravel())
            params.append(x.ravel())
        return jnp.concatenate(params)

    def create_pennylane_function(self):
        # define function to compile
        def trainable_kernel_wrapper(x1, x2, weights, bandwidth):
            device = qml.device("default.qubit.jax", wires=self.n_qubits)

            # create projector (measures probability of having all "00...0")
            projector = np.zeros((2 ** self.n_qubits, 2 ** self.n_qubits))
            projector[0, 0] = 1

            # define the circuit for the quantum kernel ("overlap test" circuit)
            @qml.qnode(device, interface='jax')
            def trainable_kernel():
                CombinatorialFeatureMap(self.overlap_parameters(x1, weights), self.n_qubits, self.n_layers,
                                        self.initial_solution, bandwidth)
                qml.adjoint(CombinatorialFeatureMap)(self.overlap_parameters(x2, weights), self.n_qubits, self.n_layers,
                                        self.initial_solution, bandwidth)
                return qml.expval(qml.Hermitian(projector, wires=range(self.n_qubits)))

            return trainable_kernel()

        return jax.jit(trainable_kernel_wrapper)

    def jnp_to_np(self, value):
        try:
            value_numpy = np.array(value.primal)
            return value_numpy
        except:
            pass
        try:
            value_numpy = np.array(value.primal.aval)
            return value_numpy
        except:
            pass
        try:
            value_numpy = np.array(value)
            return value_numpy
        except:
            raise ValueError(f"Cannot convert to numpy value {value}")

    def estimate_mse(self, weights=None, X_test=None, y_test=None):
        X_test = self.X_validation if X_test is None else X_test
        y_test = self.y_validation if y_test is None else y_test
        training_gram = self.get_kernel_values(self.X_train, weights=weights)
        training_gram = self.jnp_to_np(training_gram)
        testing_gram = self.get_kernel_values(X_test, self.X_train, weights=weights)
        testing_gram = self.jnp_to_np(testing_gram)
        return self.estimate_mse_svr(training_gram, self.y_train, testing_gram, y_test)

    def estimate_mse_svr(self, gram_train, y_train, gram_test, y_test):
        svr = SVR()
        svr.fit(gram_train, y_train.ravel())
        y_pred = svr.predict(gram_test)
        return mean_squared_error(y_test.ravel(), y_pred.ravel())

    def get_kernel_values(self, X1, X2=None, weights=None, bandwidth=None):
        weights = self.state if weights is None else weights
        bandwidth = 1.0 if bandwidth is None else bandwidth
        X1 = jnp.array(X1)
        if X2 is None:
            m = X1.shape[0]
            kernel_gram = jnp.array([[
                self.trainable_kernel(X1[i], X1[j], weights, bandwidth) if i < j else 0.0
                for j in range(m)]
                for i in range(m)
            ])
            return kernel_gram + kernel_gram.T + jnp.eye(m)
        else:
            X2 = jnp.array(X2)
            kernel_gram = jnp.array([[
                self.trainable_kernel(X1[i], X2[j], weights, bandwidth)
                for j in range(len(X2))]
                for i in range(len(X1))
            ])
        return kernel_gram

    def train(self, epochs, lr):
        opt = optax.adam(learning_rate=lr)
        opt_state = opt.init(self.state)
        for epoch in range(epochs):
            mse, grad_mse = jax.value_and_grad(self.estimate_mse)(self.state)
            updates, opt_state = opt.update(grad_mse, opt_state)
            self.state = optax.apply_updates(self.state, updates).reshape((self.n_qubits * self.n_layers,))
            print(".", end="", flush=True)
            self.history_losses.append(mse)
            self.history_grads.append(grad_mse)
            self.history_params.append(copy.deepcopy(self.state))
            if jnp.linalg.norm(grad_mse) < 0.0001:
                # gate_to_remove = np.random.choice(self.n_gates)
                # self.gate_removed.append(gate_to_remove)
                # self.state = self.state.at[gate_to_remove][0].set(0)  # identity gate
                print(f"{epoch=} Detected barren plateau!")
                return
