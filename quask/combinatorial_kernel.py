import pennylane as qml
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm


sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_id = np.eye(2)
pauli_vector = jnp.array([sigma_id, sigma_x, sigma_y, sigma_z])


def create_operation(n_qubits, n_layers, index, pauli, angle):
    assert 0 <= index < n_qubits * 2 * n_layers
    # assert 0 <= pauli < 16
    index_within_layer = index % n_layers
    if index_within_layer < n_qubits:  # single qubit operation
        unitary = expm(-1j * angle * pauli_vector[pauli % 4])
        qml.QubitUnitary(unitary, wires=index_within_layer)
    else:  # two qubits operation
        unitary = expm(-1j * angle * np.kron(pauli_vector[pauli % 4], pauli_vector[pauli // 4]))
        qml.QubitUnitary(unitary, wires=(index_within_layer, (index_within_layer + 1) % n_qubits))


def create_identity_combinatorial_kernel(n_qubits, n_layers):
    n_operations = n_qubits * 2 * n_layers
    return np.zeros(shape=(n_operations, 2))


def create_random_combinatorial_kernel(n_qubits, n_layers, n_operations):
    n_gates = n_qubits * 2 * n_layers
    return np.stack((
        np.random.randint(0, 16, size=(n_gates,)),
        np.random.randint(0, n_operations, size=(n_gates,))
    ))


def CombinatorialFeatureMap(x, n_qubits, n_layers, solution, bandwidth):

    n_gates = n_qubits * 2 * n_layers
    assert solution.shape == (n_gates, 2)
    for index in range(n_gates):
        pauli = solution[index][0]
        operation_idx = solution[index][1]
        angle = bandwidth * x[operation_idx]
        create_operation(n_qubits, n_layers, index, pauli, angle)
