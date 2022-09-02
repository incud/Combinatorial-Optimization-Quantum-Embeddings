import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
from quask.combinatorial_kernel_optimization_simanneal import *
from quask.combinatorial_kernel_mcts import *
from quask.random_kernel import *
from quask.trainable_kernel import *
from quask.datasets import *

def run_experiment(X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, repetitions, save_path):

    n_features = X_train.shape[1]

    # for i in range(repetitions):
    #     if not Path(f"{save_path}/rk_weights_{i}.npy").exists():
    #         rk = RandomKernel(X_train, y_train, X_validation, y_validation, n_layers=n_layers, seed=344143 + i)
    #         print(f"\nRandom kernel {i} loss: ", rk.estimate_mse(X_test=X_test, y_test=y_test))
    #         np.save(f"{save_path}/rk_weights_{i}.npy", rk.state)

    for i in range(repetitions):
        print("\n\n")

        if not Path(f"{save_path}/initial_solution_{i}.npy").exists():
            initial_solution = create_random_combinatorial_kernel(
                n_qubits=n_features,
                n_layers=n_layers,
                n_operations=n_features).astype(int)
            np.save(f"{save_path}/initial_solution_{i}.npy", initial_solution)
        else:
            initial_solution = jnp.array(np.load(f"{save_path}/initial_solution_{i}.npy").astype(int))

        # print(f"TrainableKernel {i} loss: ", tk.estimate_mse(X_test=X_test, y_test=y_test))
        #     if not Path(f"{save_path}/tk_weights_{i}.npy").exists():
        #         tk = TrainableKernel(X_train, y_train, X_validation, y_validation, n_layers, initial_solution)
        #         print("Start training ", datetime.datetime.now().strftime("%H:%M:%S"))
        #         tk.train(1000, lr=0.01)
        #         print("End training ", datetime.datetime.now().strftime("%H:%M:%S"))
        #
        #     np.save(f"{save_path}/tk_weights_{i}.npy", tk.state)

        if not Path(f"{save_path}/ck_solution_{i}.npy").exists():
            ck = CombinatorialKernelSimulatedAnnealingTraining(
                n_components, n_layers, initial_solution, n_components,
                X_train, y_train, X_validation, y_validation)
            print("Start annealing ", datetime.datetime.now().strftime("%H:%M:%S"))
            ck.steps = 1000
            best_solution, best_energy = ck.anneal()
            print("End annealing ", datetime.datetime.now().strftime("%H:%M:%S"))
            print(f"Combinatorial Kernel SA {i} loss: ", ck.estimate_mse(X_test=X_test, y_test=y_test))
            np.save(f"{save_path}/ck_solution_{i}.npy", best_solution)


random.seed(12345)
np.random.seed(12345)

dataset = load_who_life_expectancy_dataset()
dataset['X'] = dataset['X']
dataset['y'] = dataset['y']
n_components = min(dataset['X'].shape[0], 3)
n_elements = 10
X_train, X_other, y_train, y_other = process_regression_dataset(dataset, n_components, n_elements)
X_validation, X_test, y_validation, y_test = process_regression_dataset({'X': X_other, 'y': y_other}, n_components)

n_layers = 2
repetitions = 10
save_path = "new results 2"

initial_solution = create_random_combinatorial_kernel(
                n_qubits=n_components,
                n_layers=n_layers,
                n_operations=n_components).astype(int)

ck = CombinatorialKernelMtcs(initial_solution, n_components, n_layers, n_components, X_train, y_train, X_validation, y_validation)
print(datetime.datetime.now().strftime("%H:%M:%S"))
res = ck.search()
print(datetime.datetime.now().strftime("%H:%M:%S"))