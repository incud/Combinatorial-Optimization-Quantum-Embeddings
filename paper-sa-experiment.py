import random

from quask.combinatorial_kernel import *
from quask.combinatorial_kernel_optimization_simanneal import *
from quask.datasets import *

random.seed(12345)
np.random.seed(12345)

dataset = load_who_life_expectancy_dataset()
dataset['X'] = dataset['X'][:10]
dataset['y'] = dataset['y'][:10]
n_components = min(dataset['X'].shape[0], 3)
n_elements = 50
X_train, X_other, y_train, y_other = process_regression_dataset(dataset, n_components, n_elements)
X_validation, X_test, y_validation, y_test = process_regression_dataset({'X': X_other, 'y': y_other}, n_components)

n_qubits = n_components
n_layers = 3
initial_solution = create_identity_combinatorial_kernel(n_qubits, n_layers)
operation_table = [(lambda x: x[i]) for i in range(n_components)]

sa = CombinatorialKernelSimulatedAnnealingTraining(n_components, n_layers, initial_solution, operation_table,
                                                   X_train, y_train, X_validation, y_validation)
sa.Tmax = 25000.0
sa.Tmin = 2.5
sa.steps = 1000
sa.updates = 100
best_state, best_energy = sa.anneal()

