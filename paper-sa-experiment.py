import random
import numpy as np
import matplotlib.pyplot as plt
from quask.combinatorial_kernel_optimization_simanneal import *
from quask.random_kernel import *
from quask.trainable_kernel import *
from quask.datasets import *


def load_losses(dataset_directory, repetitions):
    labels = ["Linear SVM", "Gaussian SVM", "Neural Network", "NTK (init) + SVM", "NTK (end) + SVM", "PK + SVM", "Decorrelated PK + SVM"]
    loss_files = ["linear_loss.npy", "gaussian_loss.npy", "predictor_loss.npy", "tang_init_loss.npy", "tang_end_loss.npy", "path_loss.npy", "dec_path_loss.npy"]
    loss_data = [
        [np.load(f"{dataset_directory}_{i}/{file}") for i in range(1, repetitions+1)]
        for file in loss_files
    ]
    return np.array(loss_data).astype(float), labels


def run_experiment(X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, repetitions, save_path):

    n_features = X_train.shape[1]

    for i in range(repetitions):
        if not Path(f"{save_path}/rk_weights_{i}.npy").exists():
            rk = RandomKernel(X_train, y_train, X_validation, y_validation, n_layers=n_layers, seed=344143 + i)
            print(f"\nRandom kernel {i} loss: ", rk.estimate_mse(X_test=X_test, y_test=y_test))
            np.save(f"{save_path}/rk_weights_{i}.npy", rk.state)

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

        if not Path(f"{save_path}/tk_weights_{i}.npy").exists():
            tk = TrainableKernel(X_train, y_train, X_validation, y_validation, n_layers, initial_solution)
            print("Start training ", datetime.datetime.now().strftime("%H:%M:%S"))
            tk.train(1000, lr=0.01)
            print("End training ", datetime.datetime.now().strftime("%H:%M:%S"))
            print(f"TrainableKernel {i} loss: ", tk.estimate_mse(X_test=X_test, y_test=y_test))
            np.save(f"{save_path}/tk_weights_{i}.npy", tk.state)

        if not Path(f"{save_path}/ck_solution_{i}.npy").exists():
            operation_table = [(lambda x: x[i]) for i in range(n_components)]
            ck = CombinatorialKernelSimulatedAnnealingTraining(
                n_components, n_layers, initial_solution, operation_table,
                X_train, y_train, X_validation, y_validation)
            print("Start annealing ", datetime.datetime.now().strftime("%H:%M:%S"))
            ck.steps = 1000
            best_solution, best_energy = ck.anneal()
            print("End annealing ", datetime.datetime.now().strftime("%H:%M:%S"))
            print(f"Combinatorial Kernel SA {i} loss: ", ck.estimate_mse(X_test=X_test, y_test=y_test))
            np.save(f"{save_path}/ck_solution_{i}.npy", best_solution)


def plot_experiments(X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, repetitions, load_path):
    rk_weights = [np.load(f"{load_path}/rk_weights_{i}.npy") for i in range(repetitions)]
    init_solutions = [np.load(f"{load_path}/initial_solution_{i}.npy") for i in range(repetitions)]
    tk_weights = [np.load(f"{load_path}/tk_weights_{i}.npy") for i in range(repetitions)]
    ck_solutions = [np.load(f"{load_path}/ck_solution_{i}.npy") for i in range(repetitions)]

    rk_mse = []
    for i, rk_weight in enumerate(rk_weights):
        rk = RandomKernel(X_train, y_train, X_validation, y_validation, n_layers=n_layers, seed=344143 + 1)
        this_mse = rk.estimate_mse(weights=rk_weight, X_test=X_test, y_test=y_test)
        rk_mse.append(this_mse)
        print(f"RK MSE {i} = {this_mse:0.3f}")

    tk_mse = []
    for i, (initial_solution, tk_weight) in enumerate(zip(init_solutions, tk_weights)):
        tk = TrainableKernel(X_train, y_train, X_validation, y_validation, n_layers, initial_solution)
        this_mse = tk.estimate_mse(weights=tk_weight, X_test=X_test, y_test=y_test)
        tk_mse.append(this_mse)
        print(f"TK MSE {i} = {this_mse:0.3f}")

    ck_mse = []
    for i, ck_solution in enumerate(ck_solutions):
        operation_table = [(lambda x: x[i]) for i in range(n_components)]
        ck = CombinatorialKernelSimulatedAnnealingTraining(
            n_components, n_layers, initial_solution, operation_table,
            X_train, y_train, X_validation, y_validation)
        this_mse = ck.estimate_mse(solution=ck_solution, X_test=X_test, y_test=y_test)
        ck_mse.append(this_mse)
        print(f"CK MSE {i} = {this_mse:0.3f}")

    plt.figure()
    plt.violinplot(
        range(3),
        [rk_mse, tk_mse, ck_mse],
        widths=0.3,
        showmeans=True,
        showextrema=True,
        showmedians=True
    )
    plt.ylabel("MSE (lower = better)")
    plt.xticks(range(3), ['RK', 'TK', 'CK'], rotation=45)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(f"{load_path}/violin.png")
    plt.close()


random.seed(12345)
np.random.seed(12345)

dataset = load_who_life_expectancy_dataset()
dataset['X'] = dataset['X']
dataset['y'] = dataset['y']
n_components = min(dataset['X'].shape[0], 5)
n_elements = 100
X_train, X_other, y_train, y_other = process_regression_dataset(dataset, n_components, n_elements)
X_validation, X_test, y_validation, y_test = process_regression_dataset({'X': X_other, 'y': y_other}, n_components)

n_layers = 3
repetitions = 10
save_path = "new results"
run_experiment(X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, repetitions, save_path)
# plot_experiments(X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, repetitions, save_path)
