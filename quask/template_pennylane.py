"""
Module dedicated to define Templates for Pennylane quantum circuits.
See https://pennylane.readthedocs.io/en/stable/introduction/templates.html for details.
"""
import sys

import jax
from jax.config import config
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

import quask.metrics

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pennylane as qml
import numpy as np
import optax
import pygad
from .metrics import (
    calculate_kernel_target_alignment,
    calculate_generalization_accuracy,
    calculate_geometric_difference,
    calculate_model_complexity,
)
from scipy.linalg import expm
import bisect
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time


def rx_embedding(x, wires):
    """
    Encode the data with one rotation on sigma_x per qubit per feature

    Args:
        x: feature vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    qml.AngleEmbedding(x, wires=wires, rotation="X")


def ry_embedding(x, wires):
    """
    Encode the data with one rotation on sigma_y per qubit per feature

    Args:
        x: feature vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    qml.AngleEmbedding(x, wires=wires, rotation="Y")


def rz_embedding(x, wires):
    """
    Encode the data with one hadamard then one rotation on sigma_y per qubit per feature

    Args:
        x: feature vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    qml.Hadamard(wires=wires)
    qml.AngleEmbedding(x, wires=wires, rotation="Z")


def zz_fullentanglement_embedding(x, wires):
    """
    Encode the data with the ZZ Feature Map (https://qiskit.org/documentation/stubs/qiskit.circuit.library.ZZFeatureMap.html)

    Args:
        x: feature vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    N = len(wires)
    for i in range(N):
        qml.Hadamard(wires=i)
        qml.RZ(2 * x[i], wires=i)
    for i in range(N):
        for j in range(i + 1, N):
            qml.CRZ(2 * (np.pi - x[i]) * (np.pi - x[j]), wires=[i, j])


def hardware_efficient_ansatz(theta, wires):
    """
    Hardware efficient ansatz

    Args:
        theta: parameter vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    N = len(wires)
    assert len(theta) == 2 * N
    for i in range(N):
        qml.RX(theta[2 * i], wires=wires[i])
        qml.RY(theta[2 * i + 1], wires=wires[i])
    for i in range(N - 1):
        qml.CZ(wires=[wires[i], wires[i + 1]])


def tfim_ansatz(theta, wires):
    """
    Transverse Field Ising Model
    Figure 6a (left) in https://arxiv.org/pdf/2105.14377.pdf

    Args:
        theta: parameter vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    N = len(wires)
    assert len(theta) == 2
    for i in range(N // 2):
        qml.MultiRZ(theta[0], wires=[wires[2 * i], wires[2 * i + 1]])
    for i in range(N // 2 - 1):
        qml.MultiRZ(theta[0], wires=[wires[2 * i + 1], wires[2 * i + 2]])
    for i in range(N):
        qml.RX(theta[1], wires=wires[i])


def ltfim_ansatz(theta, wires):
    """
    Transverse Field Ising Model with additional sigma_z rotations.
    Figure 6a (right) in https://arxiv.org/pdf/2105.14377.pdf

    Args:
        theta: parameter vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    N = len(wires)
    assert len(theta) == 3
    tfim_ansatz(theta[:2], wires)
    for i in range(N):
        qml.RZ(theta[2], wires=wires[i])


def zz_rx_ansatz(theta, wires):
    """
    ZZX Model
    Figure 7a in https://arxiv.org/pdf/2109.11676.pdf

    Args:
        theta: parameter vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)

    Returns:
        None
    """
    N = len(wires)
    assert len(theta) == 2
    for i in range(N // 2):
        qml.MultiRZ(theta[0], wires=[wires[2 * i], wires[2 * i + 1]])
    for i in range(N // 2 - 1):
        qml.MultiRZ(theta[0], wires=[wires[2 * i + 1], wires[2 * i + 2]])
    for i in range(N):
        qml.RX(theta[1], wires=wires[i])


def random_qnn_encoding(x, wires, trotter_number=10):
    """
    This function creates and appends a quantum neural network to the selected
    encoding. It follows formula S(116) in the Supplementary.

    Args:
        x: feature vector (list or numpy array of floats)
        wires: wires of the circuit (list of int)
        trotter_number: number of repetitions (int)

    Returns:
        None
    """
    assert len(x) == len(wires)
    # embedding
    ry_embedding(x, wires)
    # random rotations
    for _ in range(trotter_number):
        for i in range(len(wires) - 1):
            angle = np.random.normal()
            qml.RXX(angle, wires=[wires[i], wires[i + 1]])
            qml.RYY(angle, wires=[wires[i], wires[i + 1]])
            qml.RZZ(angle, wires=[wires[i], wires[i + 1]])


def projected_xyz_embedding(embedding, X):
    """
    Create a Quantum Kernel given the template written in Pennylane framework

    Args:
        embedding: Pennylane template for the quantum feature map
        X: feature data (matrix)

    Returns:
        projected quantum feature map X
    """
    N = X.shape[1]

    # create device using JAX
    device = qml.device("default.qubit.jax", wires=N)

    # define the circuit for the quantum kernel ("overlap test" circuit)
    @qml.qnode(device, interface='jax')
    def proj_feature_map(x):
        embedding(x, wires=range(N))
        return (
            [qml.expval(qml.PauliX(i)) for i in range(N)]
            + [qml.expval(qml.PauliY(i)) for i in range(N)]
            + [qml.expval(qml.PauliZ(i)) for i in range(N)]
        )

    # build the gram matrix
    X_proj = np.array([proj_feature_map(x) for x in X])

    return X_proj


def pennylane_quantum_kernel(feature_map, X_1, X_2=None):
    """
    Create a Quantum Kernel given the template written in Pennylane framework

    Args:
        feature_map: Pennylane template for the quantum feature map
        X_1: First dataset
        X_2: Second dataset

    Returns:
        Gram matrix
    """
    if X_2 is None:
        X_2 = X_1  # Training Gram matrix
    assert (
        X_1.shape[1] == X_2.shape[1]
    ), "The training and testing data must have the same dimensionality"
    N = X_1.shape[1]

    # create device using JAX
    def get_kernel_value(x1, x2):
        device = qml.device("default.qubit.jax", wires=N)

        # create projector (measures probability of having all "00...0")
        projector = np.zeros((2**N, 2**N))
        projector[0, 0] = 1

        # define the circuit for the quantum kernel ("overlap test" circuit)
        @qml.qnode(device, interface='jax')
        def kernel():
            feature_map(x1, wires=range(N))
            qml.adjoint(feature_map)(x2, wires=range(N))
            return qml.expval(qml.Hermitian(projector, wires=range(N)))

        return kernel()

    # build the gram matrix
    gram = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
    for i in range(X_1.shape[0]):
        for j in range(i, X_2.shape[0]):
            value = get_kernel_value(X_1[i], X_2[j])
            gram[i][j] = value
            gram[j][i] = gram[i][j]

    return gram


def pennylane_projected_quantum_kernel(feature_map, X_1, X_2=None, params=[1.0]):
    """
    Create a Quantum Kernel given the template written in Pennylane framework.

    Args:
        feature_map: Pennylane template for the quantum feature map
        X_1: First dataset
        X_2: Second dataset
        params: List of one single parameter representing the constant in the exponentiation

    Returns:
        Gram matrix
    """
    if X_2 is None:
        X_2 = X_1  # Training Gram matrix
    assert (
        X_1.shape[1] == X_2.shape[1]
    ), "The training and testing data must have the same dimensionality"

    X_1_proj = projected_xyz_embedding(feature_map, X_1)
    X_2_proj = projected_xyz_embedding(feature_map, X_2)

    # build the gram matrix
    gamma = params[0]

    gram = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
    for i in range(X_1_proj.shape[0]):
        for j in range(X_2_proj.shape[0]):
            value = np.exp(-gamma * ((X_1_proj[i] - X_2_proj[j]) ** 2).sum())
            gram[i][j] = value

    return gram


class PennylaneTrainableKernel:
    """
    Create a trainable kernel using Pennylane framework.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        embedding,
        var_form,
        layers,
        optimizer,
        metric,
        seed,
        keep_intermediate=True,
    ):
        """
        Init method.

        Args:
            X_train: training set feature vector
            y_train: training set label vector
            X_test: testing set feature vector
            y_test: testing set label vector
            embedding: one of the following list: "rx", "ry", "rz", "zz"
            var_form: one of the following list: "hardware_efficient", "tfim", "ltfim", "zz_rx"
            layers: number of ansatz repetition
            optimizer: one of the following list: "adam", "grid"
            metric: one of the following list: "kernel-target-alignment", "accuracy", "geometric-difference", "model-complexity"
            seed: random seed (int)
            keep_intermediate: True if you want to keep the intermediate results of the optimization (bool)

        Returns:
            None
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        assert embedding in ["rx", "ry", "rz", "zz"]
        self.embedding = embedding
        assert var_form in ["hardware_efficient", "tfim", "ltfim", "zz_rx"]
        self.var_form = var_form
        assert 1 <= layers < 1000
        self.layers = layers
        assert optimizer in ["adam", "grid"]
        self.optimizer = optimizer
        assert metric in [
            "kernel-target-alignment",
            "accuracy",
            "geometric-difference",
            "model-complexity",
        ]
        self.metric = metric
        self.seed = seed
        self.circuit = None
        self.params = None
        self.create_circuit()
        self.intermediate_params = []
        self.intermediate_grams = []
        self.keep_intermediate = keep_intermediate

    @staticmethod
    def jnp_to_np(value):
        """
        Convert jax numpy value to numpy

        Args:
            value: jax value

        Returns:
            numpy value
        """
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

    def get_embedding(self):
        """
        Convert the embedding into its function pointer

        Returns:
            None
        """
        if self.embedding == "rx":
            return rx_embedding
        elif self.embedding == "ry":
            return ry_embedding
        elif self.embedding == "rz":
            return rz_embedding
        elif self.embedding == "zz":
            return zz_fullentanglement_embedding
        else:
            raise ValueError(f"Unknown embedding {self.embedding}")

    def get_var_form(self, n_qubits):
        """
        Convert the variational form into its function pointer

        Args:
            n_qubits: Number of qubits of the variational form

        Returns:
            (fn, n) tuple of function and integer, the former representing the ansatz and the latter the number of parameters
        """
        if self.var_form == "hardware_efficient":
            return hardware_efficient_ansatz, 2 * n_qubits
        elif self.var_form == "tfim":
            return tfim_ansatz, 2
        elif self.var_form == "ltfim":
            return ltfim_ansatz, 3
        elif self.var_form == "zz_rx":
            return zz_rx_ansatz, 2
        else:
            raise ValueError(f"Unknown var_form {self.var_form}")

    def create_circuit(self):
        """
        Creates the quantum circuit to be simulated with jax.

        Returns:
            None
        """
        N = self.X_train.shape[1]
        device = qml.device("default.qubit.jax", wires=N)
        embedding_fn = self.get_embedding()
        var_form_fn, params_per_layer = self.get_var_form(N)

        @jax.jit
        @qml.qnode(device, interface="jax")
        def circuit(x, theta):
            embedding_fn(x, wires=range(N))
            for i in range(self.layers):
                var_form_fn(
                    theta[i * params_per_layer : (i + 1) * params_per_layer],
                    wires=range(N),
                )
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(N)]

        self.circuit = circuit
        self.params = jax.random.normal(
            jax.random.PRNGKey(self.seed), shape=(self.layers * params_per_layer,)
        )

    def get_gram_matrix(self, X_1, X_2, theta):
        """
        Get the gram matrix given the actual parameters.

        Args:
            X_1: first set (testing)
            X_2: second set (training)
            theta: parameters

        Returns:
            Gram matrix
        """
        X_proj_1 = jnp.array([self.circuit(x, theta) for x in X_1])
        X_proj_2 = jnp.array([self.circuit(x, theta) for x in X_2])
        gamma = 1.0

        gram = np.zeros(shape=(X_1.shape[0], X_2.shape[0]))
        for i in range(X_proj_1.shape[0]):
            for j in range(X_proj_2.shape[0]):
                value = jnp.exp(-gamma * ((X_proj_1[i] - X_proj_2[j]) ** 2).sum())
                gram[i][j] = PennylaneTrainableKernel.jnp_to_np(value)
        return gram

    def get_loss(self, theta):
        """
        Get loss according to the wanted metric.

        Args:
            theta: parameter vector

        Returns:
            loss (float)
        """
        theta_numpy = PennylaneTrainableKernel.jnp_to_np(theta)
        training_gram = self.get_gram_matrix(self.X_train, self.X_train, theta)
        if self.keep_intermediate:
            self.intermediate_params.append(theta_numpy)
            self.intermediate_grams.append(training_gram)
        if self.metric == "kernel-target-alignment":
            return 1 / calculate_kernel_target_alignment(training_gram, self.y_train)
        elif self.metric == "accuracy":
            return 1 / calculate_generalization_accuracy(
                training_gram, self.y_train, training_gram, self.y_train
            )
        elif self.metric == "geometric-difference":
            comparison_gram = np.outer(self.X_train, self.X_train)
            return 1 / calculate_geometric_difference(training_gram, comparison_gram)
        elif self.metric == "model-complexity":
            return 1 / calculate_model_complexity(training_gram, self.y_train)
        else:
            raise ValueError(f"Unknown metric {self.metric} for loss function")

    def get_optimizer(self):
        """
        Convert the optimizer from string to object

        Returns:
            optimizer object
        """
        if self.optimizer == "adam":
            return optax.adam(learning_rate=0.1)
        elif self.optimizer == "grid":
            return "grid"
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")

    def optimize_circuit(self):
        """
        Run optimization of the circuit

        Returns:
            None
        """
        optimizer = self.get_optimizer()
        if optimizer == "grid":
            raise ValueError("Not implemented yet")
        else:
            opt_state = optimizer.init(self.params)
            epochs = 2
            for epoch in range(epochs):
                cost, grad_circuit = jax.value_and_grad(
                    lambda theta: self.get_loss(theta)
                )(self.params)
                updates, opt_state = optimizer.update(grad_circuit, opt_state)
                self.params = optax.apply_updates(self.params, updates)
                print(".", end="", flush=True)

    def get_optimized_gram_matrices(self):
        """
        Get optimized gram matrices

        Returns:
            (tr,te) tuple of training and testing gram matrices
        """
        training_gram = self.get_gram_matrix(self.X_train, self.X_train, self.params)
        testing_gram = self.get_gram_matrix(self.X_test, self.X_train, self.params)
        return training_gram, testing_gram


class GeneticEmbedding:

    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_id = np.eye(2)
    PAULIS = [sigma_id, sigma_x, sigma_y, sigma_z]
    N_PAULIS = 4

    def __init__(self, X, y, n_qubits, layers, kernel_concentration_threshold,
                 bandwidth=1.0,
                 num_generations=50,
                 num_parents_mating=4,
                 solution_per_population=4,
                 parent_selection_type="sss",
                 crossover_type="single_point",
                 mutation_type="random",
                 mutation_percent_genes=10,
                 fitness_mode='kta',
                 validation_X = None,
                 validation_y = None,
                 initial_population = None,
                 verbose = True):
        self.X = X
        self.y = y
        self.n_features = len(X[0])
        self.operations = self.create_operations()
        self.range_operation = len(self.operations)
        self.range_gene = self.N_PAULIS * self.N_PAULIS * len(self.operations)
        self.bandwidth = bandwidth
        self.n_qubits = n_qubits
        self.fitness_mode = fitness_mode
        self.kernel_concentration_threshold = kernel_concentration_threshold
        self.layers = layers
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.solution_per_population = solution_per_population
        self.validation_X = validation_X
        self.validation_y = validation_y
        self.variance_idxs = [[],[]]
        self.low_variance_list = []
        self.initial_population = initial_population
        self.verbose = verbose
        self.start = time.process_time()
        self.count = 0

        def prep_variance_computation():
            n = np.shape(self.X)[0]
            self.variance_idxs = [[],[]]
            idxs = np.random.choice(range(int((n*n-1)/2)), int(np.log2(n*n)), replace = False)
            for i in idxs:
                row = 0
                column = n - 1
                c = column
                while i > c:
                    column -= 1
                    row += 1
                    c += column
                self.variance_idxs[0].append(row)
                self.variance_idxs[1].append(n + i - c - 1)
            self.low_variance_list.append([])

        def on_start(ga_instance):
            prep_variance_computation()
            self.start = time.process_time()
            if self.verbose == True:
                print('S', end='\n', flush=True)

        def on_fitness(ga_instance, population_fitness):
            prep_variance_computation()
            end = time.process_time()
            self.count += 1
            if self.verbose == True:
                print(self.low_variance_list[len(self.low_variance_list) - 2])
                print(f'F:{np.min(population_fitness)}-{np.max(population_fitness)} ', end='', flush=True)
            elif self.verbose == 'minimal':
                sys.stdout.write('\033[K' + 'Remaining generation: ' + str(self.num_generations - self.count) +
                                 ' --- Max Fitness: ' + str(np.max(population_fitness)) +
                                 ' --- Estimated time left: ' + str(timedelta(seconds=(self.num_generations - self.count) * (end - self.start) / self.count)) + ' ')

        def on_parents(ga_instance, selected_parents):
            if self.verbose == True:
                print('P', end='', flush=True)
            elif self.verbose == 'minimal':
                sys.stdout.write('P')

        def on_crossover(ga_instance, offspring_crossover):
            if self.verbose == True:
                print('C', end='', flush=True)
            elif self.verbose == 'minimal':
                sys.stdout.write('C')

        def on_mutation(ga_instance, offspring_mutation):
            if self.verbose == True:
                print('M', end='', flush=True)
            elif self.verbose == 'minimal':
                sys.stdout.write('M')

        def on_generation(ga_instance):
            if self.verbose == True:
                print('G', end='\n', flush=True)
            elif self.verbose == 'minimal':
                sys.stdout.write('G\r')

        def on_stop(ga_instance, last_population_fitness):
            self.count = 0
            if self.verbose == True:
                print('X', end='\n', flush=True)

        self.ga = pygad.GA(
            fitness_func=lambda sol, sol_idx: self.fitness(sol, sol_idx),
            num_genes=self.get_genes(),
            gene_type=int,
            gene_space=range(self.get_range_gene()),
            init_range_low=0,
            init_range_high=self.get_range_gene() - 1,
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            sol_per_pop=self.solution_per_population,
            parent_selection_type=parent_selection_type,
            keep_parents=-1,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            mutation_percent_genes=mutation_percent_genes,
            save_solutions=True,
            save_best_solutions=True,
            initial_population=initial_population,
            on_start=on_start,
            on_fitness=on_fitness,
            on_parents=on_parents,
            on_crossover=on_crossover,
            on_mutation=on_mutation,
            on_generation=on_generation,
            on_stop=on_stop
        )

    def create_operations(self, include_inverse=False, constant_ticks=4):
        operations = []
        # constants
        for i in range(constant_ticks):
            operations.append(lambda _: np.pi * (i + 1) / constant_ticks)
        # first degree operations
        for i in range(self.n_features):
            operations.append(lambda x: x[i])
            if include_inverse: operations.append(lambda x: np.pi - x[i])
        # second degree operations
        for i in range(self.n_features):
            for j in range(self.n_features):
                operations.append(lambda x: x[i] * x[j])
                if include_inverse:  operations.append(lambda x: (np.pi - x[i]) * (np.pi - x[j]))
        return operations

    def get_genes(self):
        return self.get_genes_per_layer() * self.layers

    def get_genes_per_layer(self):
        return self.n_qubits * 2

    def get_range_gene(self):
        return self.range_gene

    def get_range_operation(self):
        return self.range_operation

    def fitness(self, solution, solution_idx):

        # compute accuracy for regression tasks
        def accuracy_svr(gram, gram_test, y, y_test):
            svm = SVR(kernel='precomputed').fit(gram, y)
            y_predict = svm.predict(gram_test)
            return -mean_squared_error(y_test, y_predict)

        # compute variance of pre-selected gram matrix entries
        def compute_variance(feat_map, X_1, X_2, params=[1.0]):
            var_list = []
            X_1_proj = projected_xyz_embedding(feat_map, X_1)
            X_2_proj = projected_xyz_embedding(feat_map, X_2)
            gamma = params[0]
            for i in range(X_1_proj.shape[0]):
                value = np.exp(-gamma * ((X_1_proj[i] - X_2_proj[i]) ** 2).sum())
                var_list.append(value)
            return np.var(var_list)

        feature_map = lambda x, wires: self.transform_solution_to_embedding(x, solution)
        X_batch = self.X
        y_batch = self.y

        variance = compute_variance(feature_map, X_batch[self.variance_idxs[0]], X_batch[self.variance_idxs[1]])
        if variance < self.kernel_concentration_threshold:
            self.low_variance_list[len(self.low_variance_list) - 1].append(variance)
            return -np.inf

        gram_matrix = pennylane_projected_quantum_kernel(feature_map, X_batch)

        if self.fitness_mode == 'mse':
            if self.validation_X is None or self.validation_y is None:
                fitness = accuracy_svr(gram_matrix, gram_matrix, y_batch, y_batch)
            else:
                gram_matrix_validation = pennylane_projected_quantum_kernel(feature_map, self.validation_X, X_batch)
                fitness = accuracy_svr(gram_matrix, gram_matrix_validation, y_batch, self.validation_y)
        else:
            fitness = quask.metrics.calculate_kernel_target_alignment(gram_matrix, y_batch)

        return fitness

    def run(self):
        self.ga.run()
        self.ga.plot_fitness()
        best_solution, best_solution_fitness, _ = self.ga.best_solution()
        if self.verbose == True: print(f"Best solution: {best_solution} fitness={best_solution_fitness}")
        plt.savefig(f'fitness_{datetime.now().strftime("%y%m%d_%H%M%S")}.png')
        plt.clf()

    def transform_solution_to_embedding(self, x, solution):
        for i, gene in enumerate(solution):
            n_operation = i % self.get_genes_per_layer()
            is_single_qubit = n_operation < self.n_qubits
            wires = [(n_operation + i) % self.n_qubits for i in range(2)]
            self.apply_operation(gene, x, is_single_qubit, wires=(wires[0] if is_single_qubit else wires))

    def unpack_gene(self, gene):
        paulis = gene % (self.N_PAULIS ** 2)
        pauli_1 = paulis // self.N_PAULIS
        pauli_2 = paulis % self.N_PAULIS
        operation_index = gene // (self.N_PAULIS ** 2)
        return pauli_1, pauli_2, operation_index

    def apply_operation(self, gene, x, is_single_qubit, wires):

        pauli_1, pauli_2, operation_index = self.unpack_gene(gene)

        # calculate angle from data
        # print(f"Gene={gene} (max {self.get_range_gene()} | Operation index={operation_index} (max {self.get_range_operation()})")
        angle = self.bandwidth * self.operations[operation_index](x)

        if is_single_qubit:
            if pauli_1 == 0: qml.Identity(wires=wires)
            elif pauli_1 == 1: qml.RX(angle, wires=wires)
            elif pauli_1 == 2: qml.RY(angle, wires=wires)
            elif pauli_1 == 3: qml.RZ(angle, wires=wires)
        else:
            unitary = expm(-1j * angle * np.kron(self.PAULIS[pauli_1], self.PAULIS[pauli_2]))
            assert self.is_unitary(unitary)
            qml.QubitUnitary(unitary, wires=wires)

    @staticmethod
    def is_unitary(m):
        """
        Check if the matrix is unitary (up to a small numerical error).

        Args:
            m: square numpy 2-dimensional array of floats

        Returns:
            True if the matrix is unitary, False otherwise
        """
        return np.allclose(np.eye(m.shape[0]), m.dot(m.conj().T))
