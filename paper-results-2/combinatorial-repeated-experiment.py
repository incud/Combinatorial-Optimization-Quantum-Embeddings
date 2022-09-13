import datetime
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from quask.datasets import *
from quask.random_kernel import RandomKernel
from quask.trainable_kernel import TrainableKernel
from quask.scrambled_kernel import ScrambledKernel
from quask.combinatorial_kernel import *
from quask.combinatorial_kernel_simanneal import CombinatorialKernelSimulatedAnnealingTraining
from quask.combinatorial_kernel_greedy import CombinatorialKernelGreedySearch
from quask.combinatorial_kernel_genetic import CombinatorialKernelGenetic


# =====================================================================================
# 0. UTILITY FUNCTIONS ================================================================
# =====================================================================================


def reset_seed(the_seed):
    random.seed(the_seed)
    np.random.seed(the_seed)


reset_seed(43843933)
# do not move the generation of the seeds! Always add the the bottom
SIMULATION_SEED = np.random.randint(100000, size=(1000,))
SCRAMBLED_4_SEED = np.random.randint(100000, size=(1000,))
SCRAMBLED_3_SEED = np.random.randint(100000, size=(1000,))
GENETIC_SEED = np.random.randint(100000, size=(1000,))
simulation_seed_index = 0
scrambled_3_seed_index = 0
scrambled_4_seed_index = 0
genetic_seed_index = 0


def get_next_seed():
    global SIMULATION_SEED, simulation_seed_index
    ret = SIMULATION_SEED[simulation_seed_index]
    simulation_seed_index += 1
    return int(ret)


DATASET_PATH = "paper-results-2/datasets"
INTERMEDIATE_PATH = "paper-results-2/intermediate"
PLOT_PATH = "paper-results-2/plots"
np.save(f"{INTERMEDIATE_PATH}/SIMULATION_SEED.npy", SIMULATION_SEED)
REFRESH_DATASET = False
REFRESH_INTERMEDIATE = False

# =====================================================================================
# 1. GENERATE DATASETS ================================================================
# =====================================================================================
# Load each dataset and split into three equal-sized parts. You can repetately test
# any model by selecting one part as train, one as validation and the last one as test sets.
# You can more accurately estimate the generalization of your model using k-fold, meaning
# you swap the three parts and retrain the model and finally averages the performances.


def save_datasets(dataset_path, dataset, n_components, n_elements):
    assert Path(dataset_path).exists() and Path(dataset_path).is_dir()
    if Path(f"{dataset_path}/X_train.npy").exists() and not REFRESH_DATASET:
        print(f"{dataset_path} is non-empty, skipping generation of this dataset")
        return
    X_train, X_val_test, y_train, y_val_test = process_regression_dataset(dataset, n_components=n_components, n_elements=n_elements, test_size=0.66)
    X_validation, X_test, y_validation, y_test = process_regression_dataset({'X': X_val_test, 'y': y_val_test}, n_components=n_components, test_size=0.50)
    np.save(f"{dataset_path}/X_train.npy",      X_train)
    np.save(f"{dataset_path}/X_validation.npy", X_validation)
    np.save(f"{dataset_path}/X_test.npy",       X_test)
    np.save(f"{dataset_path}/y_train.npy",      y_train)
    np.save(f"{dataset_path}/y_validation.npy", y_validation)
    np.save(f"{dataset_path}/y_test.npy",       y_test)


dataset = load_fish_market_dataset()
reset_seed(12345)
print(f"Dataset 'fish market' has {dataset['X'].shape[0]} elements with {dataset['X'].shape[1]} features")
save_datasets(f"{DATASET_PATH}/fish_market", dataset, 5, 150)
# save_datasets(f"{DATASET_PATH}/fish_market", dataset, 3, 15)

dataset = load_function_approximation_sin_squared()
reset_seed(54321)
print(f"Dataset 'function approximation sin squared' has {dataset['X'].shape[0]} elements with {dataset['X'].shape[1]} features")
save_datasets(f"{DATASET_PATH}/function_approximation_sin_squared", dataset, 2, 90)
# save_datasets(f"{DATASET_PATH}/function_approximation_sin_squared", dataset, 2, 15)

dataset = load_function_approximation_step()
reset_seed(10293)
print(f"Dataset 'function approximation step' has {dataset['X'].shape[0]} elements with {dataset['X'].shape[1]} features")
save_datasets(f"{DATASET_PATH}/function_approximation_step", dataset, 2, 90)
# save_datasets(f"{DATASET_PATH}/function_approximation_step", dataset, 2, 15)

dataset = load_function_approximation_meyer_wavelet()
reset_seed(19283)
print(f"Dataset 'function approximation meyer wavelet' has {dataset['X'].shape[0]} elements with {dataset['X'].shape[1]} features")
save_datasets(f"{DATASET_PATH}/function_approximation_meyer_wavelet", dataset, 2, 90)
# save_datasets(f"{DATASET_PATH}/function_approximation_meyer_wavelet", dataset, 2, 15)

dataset = load_who_life_expectancy_dataset()
reset_seed(54562)
print(f"Dataset 'life expectancy' has {dataset['X'].shape[0]} elements with {dataset['X'].shape[1]} features")
save_datasets(f"{DATASET_PATH}/life_expectancy", dataset, 5, 150)
# save_datasets(f"{DATASET_PATH}/life_expectancy", dataset, 3, 15)

dataset = load_medical_bill_dataset()
reset_seed(39284)
print(f"Dataset 'medical bill' has {dataset['X'].shape[0]} elements with {dataset['X'].shape[1]} features")
save_datasets(f"{DATASET_PATH}/medical_bill", dataset, 5, 150)
# save_datasets(f"{DATASET_PATH}/medical_bill", dataset, 3, 15)

dataset = load_ols_cancer_dataset()
reset_seed(19475)
print(f"Dataset 'ols cancer' has {dataset['X'].shape[0]} elements with {dataset['X'].shape[1]} features")
save_datasets(f"{DATASET_PATH}/ols_cancer", dataset, 5, 150)
# save_datasets(f"{DATASET_PATH}/ols_cancer", dataset, 3, 15)

dataset = load_real_estate_dataset()
reset_seed(24298)
print(f"Dataset 'real estate' has {dataset['X'].shape[0]} elements with {dataset['X'].shape[1]} features")
save_datasets(f"{DATASET_PATH}/real_estate", dataset, 5, 150)
# save_datasets(f"{DATASET_PATH}/real_estate", dataset, 3, 15)


# =====================================================================================
# 2. GENERATE INTERMEDIATE RESULTS ====================================================
# =====================================================================================

def run_scrambled_kernel(save_path, X_train, X_validation, X_test, y_train, y_validation, y_test, n_qubits):
    if Path(f"{save_path}/mse.npy").exists() and not REFRESH_INTERMEDIATE:
        print(f"{save_path} is non-empty, skipping this scrambled kernel")
        return
    sk = ScrambledKernel(X_train, y_train, X_validation, y_validation, n_qubits=n_qubits)
    gram_train = sk.get_kernel_values(X_train)
    gram_test = sk.get_kernel_values(X_test, X_train)
    np.save(f"{save_path}/state.npy", sk.state)
    np.save(f"{save_path}/gram_train.npy", gram_train)
    np.save(f"{save_path}/gram_test.npy", gram_test)
    try:
        mse = sk.estimate_mse_svr(gram_train, y_train, gram_test, y_test)
    except:
        print("Cannot calculate MSE due to faulty gram matrix")
        mse = np.array(np.inf)

    np.save(f"{save_path}/mse.npy", np.array(mse))


def run_random_kernel(save_path, X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, seed):
    if Path(f"{save_path}/mse.npy").exists() and not REFRESH_INTERMEDIATE:
        print(f"{save_path} is non-empty, skipping this random kernel")
        return
    rk = RandomKernel(X_train, y_train, X_validation, y_validation, n_layers, seed=seed)
    gram_train = rk.get_kernel_values(X_train)
    gram_test = rk.get_kernel_values(X_test, X_train)
    np.save(f"{save_path}/n_layers.npy", np.array(n_layers))
    np.save(f"{save_path}/seed.npy", np.array(seed))
    np.save(f"{save_path}/state.npy", rk.state)
    np.save(f"{save_path}/gram_train.npy", gram_train)
    np.save(f"{save_path}/gram_test.npy", gram_test)
    try:
        mse = rk.estimate_mse_svr(gram_train, y_train, gram_test, y_test)
    except:
        print("Cannot calculate MSE due to faulty gram matrix")
        mse = np.array(np.inf)

    np.save(f"{save_path}/mse.npy", np.array(mse))


def run_trainable_kernel(save_path, X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, initial_solution, epochs, lr):
    if Path(f"{save_path}/mse.npy").exists() and not REFRESH_INTERMEDIATE:
        print(f"{save_path}is non-empty, skipping this trainable kernel")
        return
    np.save(f"{save_path}/initial_solution.npy", np.array(initial_solution))
    tk = TrainableKernel(X_train, y_train, X_validation, y_validation, n_layers, initial_solution)
    tk.train(epochs, lr)
    np.save(f"{save_path}/history_losses.npy", np.array(tk.history_losses))
    np.save(f"{save_path}/history_grads_norm.npy", np.linalg.norm(np.array(tk.history_grads)))
    np.save(f"{save_path}/gate_removed.npy", np.array(tk.gate_removed))
    gram_train = tk.get_kernel_values(X_train)
    gram_test = tk.get_kernel_values(X_test, X_train)
    try:
        mse = tk.estimate_mse(X_test=X_test, y_test=y_test)
    except:
        print("Cannot calculate MSE due to faulty gram matrix")
        mse = np.array(np.inf)
    np.save(f"{save_path}/gram_train.npy", gram_train)
    np.save(f"{save_path}/gram_test.npy", gram_test)
    np.save(f"{save_path}/mse.npy", np.array(mse))


def run_combinatorial_sa_kernel(save_path, X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, initial_solution, steps):
    if Path(f"{save_path}/mse.npy").exists() and not REFRESH_INTERMEDIATE:
        print(f"{save_path} is non-empty, skipping this combinatorial (sa) kernel")
        return
    np.save(f"{save_path}/initial_solution.npy", np.array(initial_solution))
    ck = CombinatorialKernelSimulatedAnnealingTraining(
        X_train.shape[1], n_layers, initial_solution, X_train.shape[1],
        X_train, y_train, X_validation, y_validation)
    ck.steps = steps
    best_solution, best_energy = ck.anneal()
    gram_train = ck.get_kernel_values(X_train)
    gram_test = ck.get_kernel_values(X_test, X_train)
    mse = ck.estimate_mse(X_test=X_test, y_test=y_test)
    np.save(f"{save_path}/final_solution.npy", ck.state)
    np.save(f"{save_path}/energy_calculation_performed.npy", np.array(ck.energy_calculation_performed))
    np.save(f"{save_path}/energy_calculation_discarded.npy", np.array(ck.energy_calculation_discarded))
    np.save(f"{save_path}/gram_train.npy", gram_train)
    np.save(f"{save_path}/gram_test.npy", gram_test)
    np.save(f"{save_path}/mse.npy", np.array(mse))


def run_combinatorial_greedy_kernel(save_path, X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, initial_solution):
    if Path(f"{save_path}/mse.npy").exists() and not REFRESH_INTERMEDIATE:
        print(f"{save_path} is non-empty, skipping this combinatorial (greedy) kernel")
        return
    np.save(f"{save_path}/initial_solution.npy", np.array(initial_solution))
    ck = CombinatorialKernelGreedySearch(initial_solution, X_train.shape[1], n_layers, X_train.shape[1],
                                         X_train, y_train, X_validation, y_validation)
    ck.search()
    gram_train = ck.get_kernel_values(X_train)
    gram_test = ck.get_kernel_values(X_test, X_train)
    mse = ck.estimate_mse(X_test=X_test, y_test=y_test)
    np.save(f"{save_path}/final_solution.npy", ck.solution)
    np.save(f"{save_path}/energy_calculation_performed.npy", np.array(ck.energy_calculation_performed))
    np.save(f"{save_path}/energy_calculation_discarded.npy", np.array(ck.energy_calculation_discarded))
    np.save(f"{save_path}/gram_train.npy", gram_train)
    np.save(f"{save_path}/gram_test.npy", gram_test)
    np.save(f"{save_path}/mse.npy", np.array(mse))


def run_combinatorial_genetic_kernel(save_path, X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, ck):
    if Path(f"{save_path}/mse.npy").exists() and not REFRESH_INTERMEDIATE:
        print(f"{save_path} is non-empty, skipping this combinatorial (greedy) kernel")
        return

    # X_train, y_train, X_validation, y_validation, ck, n_qubits, n_layers
    n_qubits = X_train.shape[1]
    ck = CombinatorialKernelGenetic(X_train, y_train, X_validation, y_validation, ck, n_qubits, n_layers)
    ck.generate_initial_population()
    np.save(f"{save_path}/initial_items.npy", np.array([item[0] for item in ck.initial_population]))
    np.save(f"{save_path}/initial_cost.npy", np.array([item[1] for item in ck.initial_population]))
    ck.run_genetic_optimization()
    np.save(f"{save_path}/final_items.npy", np.array([item[0] for item in ck.current_population]))
    np.save(f"{save_path}/final_cost.npy", np.array([item[1] for item in ck.current_population]))
    gram_train = ck.get_kernel_values(X_train)
    gram_test = ck.get_kernel_values(X_test, X_train)
    mse = ck.estimate_mse(X_test=X_test, y_test=y_test)
    np.save(f"{save_path}/gram_train.npy", gram_train)
    np.save(f"{save_path}/gram_test.npy", gram_test)
    np.save(f"{save_path}/energy_calculation_performed.npy", np.array(ck.energy_calculation_performed))
    np.save(f"{save_path}/energy_calculation_discarded.npy", np.array(ck.energy_calculation_discarded))
    np.save(f"{save_path}/mse.npy", np.array(mse))


def run_simulations(n_layers, epochs, lr, repetitions=10):
    global SCRAMBLED_3_SEED, SCRAMBLED_4_SEED, GENETIC_SEED, scrambled_4_seed_index, scrambled_3_seed_index, genetic_seed_index

    DATASETS = ["fish_market", "function_approximation_meyer_wavelet", "function_approximation_sin_squared",
                "function_approximation_step", "life_expectancy",
                "medical_bill", "ols_cancer", "real_estate"]

    combinatorial_kernel_2 = CombinatorialKernel(2, n_layers)
    combinatorial_kernel_5 = CombinatorialKernel(5, n_layers)

    for dataset in DATASETS:

        X_train         = np.load(f"{DATASET_PATH}/{dataset}/X_train.npy")
        X_validation    = np.load(f"{DATASET_PATH}/{dataset}/X_validation.npy")
        X_test          = np.load(f"{DATASET_PATH}/{dataset}/X_test.npy")
        y_train         = np.load(f"{DATASET_PATH}/{dataset}/y_train.npy")
        y_validation    = np.load(f"{DATASET_PATH}/{dataset}/y_validation.npy")
        y_test          = np.load(f"{DATASET_PATH}/{dataset}/y_test.npy")

        seed = get_next_seed()
        reset_seed(seed)
        initial_solution = create_random_combinatorial_kernel(n_qubits=X_train.shape[1], n_layers=n_layers,
                                                              n_operations=X_train.shape[1]).astype(int)

        for repetition in range(repetitions):

            print(f"========================= {dataset} {repetition} ========================= ")
            timing = {}
            ck = combinatorial_kernel_5 if X_train.shape[1] == 5 else combinatorial_kernel_2

            # SCRAMBLED 3 QUBITS =================================================================
            timing['scrambled_kernel_start-3'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            print(f"Scrambled kernel starts at {timing['scrambled_kernel_start-3']}")
            reset_seed(int(SCRAMBLED_3_SEED[scrambled_3_seed_index]))
            scrambled_3_seed_index += 1
            Path(f"{INTERMEDIATE_PATH}/{dataset}/scrambled_kernel-3/{repetition}").mkdir(exist_ok=True)
            run_scrambled_kernel(f"{INTERMEDIATE_PATH}/{dataset}/scrambled_kernel-3/{repetition}", X_train,
                                 X_validation,
                                 X_test, y_train, y_validation, y_test, n_qubits=3)
            timing['scrambled_kernel_end-3'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            # SCRAMBLED 4 QUBITS =================================================================
            timing['scrambled_kernel_start-4'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            print(f"Scrambled kernel starts at {timing['scrambled_kernel_start-4']}")
            reset_seed(int(SCRAMBLED_4_SEED[scrambled_4_seed_index]))
            scrambled_4_seed_index += 1
            Path(f"{INTERMEDIATE_PATH}/{dataset}/scrambled_kernel-4/{repetition}").mkdir(exist_ok=True)
            run_scrambled_kernel(f"{INTERMEDIATE_PATH}/{dataset}/scrambled_kernel-4/{repetition}", X_train, X_validation,
                              X_test, y_train, y_validation, y_test, n_qubits=4)
            timing['scrambled_kernel_end-4'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            # RANDOM =======================================================================
            timing['random_kernel_start'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            print(f"Random kernel starts at {timing['random_kernel_start']}")
            seed = get_next_seed()
            reset_seed(seed)
            Path(f"{INTERMEDIATE_PATH}/{dataset}/random_kernel/{repetition}").mkdir(exist_ok=True)
            run_random_kernel(f"{INTERMEDIATE_PATH}/{dataset}/random_kernel/{repetition}", X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, seed)
            timing['random_kernel_end'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            # COMBINATORIAL SA =============================================================
            timing['combinatorial_sa_kernel_start'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            print(f"Combinatorial (sa) kernel starts at {timing['combinatorial_sa_kernel_start']}")
            seed = get_next_seed()
            reset_seed(seed)
            Path(f"{INTERMEDIATE_PATH}/{dataset}/combinatorial_sa_kernel/{repetition}").mkdir(exist_ok=True)
            run_combinatorial_sa_kernel(f"{INTERMEDIATE_PATH}/{dataset}/combinatorial_sa_kernel/{repetition}",
                X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, initial_solution, epochs)
            timing['combinatorial_sa_kernel_end'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            # COMBINATORIAL GREEDY =============================================================
            timing['combinatorial_greedy_kernel_start'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            print(f"Combinatorial (greedy) kernel starts at {timing['combinatorial_greedy_kernel_start']}")
            seed = get_next_seed()
            reset_seed(seed)
            Path(f"{INTERMEDIATE_PATH}/{dataset}/combinatorial_greedy_kernel/{repetition}").mkdir(exist_ok=True)
            run_combinatorial_greedy_kernel(f"{INTERMEDIATE_PATH}/{dataset}/combinatorial_greedy_kernel/{repetition}",
                X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, initial_solution)
            timing['combinatorial_greedy_kernel_end'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            # COMBINATORIAL GENETIC =============================================================
            timing['combinatorial_genetic_kernel_start'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            print(f"Combinatorial (genetic) kernel starts at {timing['combinatorial_genetic_kernel_start']}")
            reset_seed(int(GENETIC_SEED[genetic_seed_index]))
            genetic_seed_index += 1
            Path(f"{INTERMEDIATE_PATH}/{dataset}/combinatorial_genetic_kernel/{repetition}").mkdir(exist_ok=True)
            run_combinatorial_genetic_kernel(f"{INTERMEDIATE_PATH}/{dataset}/combinatorial_genetic_kernel/{repetition}",
                                            X_train, X_validation, X_test, y_train, y_validation, y_test,
                                            n_layers, ck)
            timing['combinatorial_genetic_kernel_end'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            # TODO ADD INIT OF THE TRAINING

            json.dump(timing, open(f"{INTERMEDIATE_PATH}/{dataset}/timing.json", "w"))


# run_simulations(3, 1000, 0.01, 10)


# =====================================================================================
# 3. GENERATE PLOTS ===================================================================
# =====================================================================================


def plot_mse(the_plot_path, the_mse_data, dataset_name, title=None):
    mse_items = list(the_mse_data.items())
    mse_items = [(k, v) for (k, v) in mse_items if len(v) > 0]
    if not mse_items:
        print(f"Warning: {dataset_name=} has not be processed at all")
        return

    # plot graph
    plt.figure()
    plt.violinplot([v for k, v in mse_items], range(len(mse_items)),
        widths=0.3, showmeans=True, showextrema=True, showmedians=True)
    plt.ylabel("MSE (lower is better)")
    plt.xticks(range(len(mse_items)), [k for k, v in mse_items], rotation=45)
    plt.subplots_adjust(bottom=0.25)
    plt.title(f"MSE for {dataset_name=}" if title is None else title)
    plt.savefig(f"{the_plot_path}/{dataset_name}.png")
    plt.close()


def plot_eigvals(the_plot_path, the_eigvals_data, dataset_name, title=None, ylim=None):
    eigvals_items = list(the_eigvals_data.items())
    eigvals_items = [(k, v) for (k, v) in eigvals_items if len(v) > 0]
    if not eigvals_items:
        print(f"Warning: {dataset_name=} has not be processed at all")
        return

    # plot graph
    plt.figure()
    for i in range(len(eigvals_items[0][1])):
        plt.violinplot([v[i] for k, v in eigvals_items], range(len(eigvals_items)),
            widths=0.3, showmeans=True, showextrema=True, showmedians=True)
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel("Eigenvalue")
    plt.xticks(range(len(eigvals_items)), [k for k, v in eigvals_items], rotation=45)
    plt.subplots_adjust(bottom=0.25)
    plt.title(f"Eigenvalue distribution for for {dataset_name=}" if title is None else title)
    plt.savefig(f"{the_plot_path}/{dataset_name}.png")
    plt.close()


def plot_coefficients(the_plot_path, the_coeffs_data, dataset_name, title=None):
    coeffs_items = list(the_coeffs_data.items())
    coeffs_items = [(k, v) for (k, v) in coeffs_items if len(v) > 0]
    if not coeffs_items:
        print(f"Warning: {dataset_name=} has not be processed at all")
        return

    # plot graph
    plt.figure()
    for i in range(len(coeffs_items[0][1])):
        parts = plt.violinplot([v[i] for k, v in coeffs_items], range(len(coeffs_items)),
            widths=0.3, showmeans=True, showextrema=True, showmedians=True)
        for j in range(len(parts['bodies'])):
            parts['bodies'][j].set_facecolor("None")  # "None" = transparent
            parts['bodies'][j].set_edgecolor('#000000')
            parts['bodies'][j].set_linewidth(0.75)
            parts['bodies'][j].set_alpha(0.50)
    plt.ylabel("Coefficient value")
    plt.xticks(range(len(coeffs_items)), [k for k, v in coeffs_items], rotation=45)
    plt.subplots_adjust(bottom=0.25)
    plt.title(f"Coefficient distributions for {dataset_name=}" if title is None else title)
    plt.savefig(f"{the_plot_path}/{dataset_name}.png")
    plt.close()


def plot_mse_variance(the_plot_path, the_mse_data, the_coeffs_data, dataset_name, title=None):
    mse_items = list(the_mse_data.items())
    mse_items = [(k, v) for (k, v) in mse_items if len(v) > 0]
    if not mse_items:
        print(f"Warning: {dataset_name=} has not be processed at all")
        return
    coeffs_items = list(the_coeffs_data.items())
    coeffs_items = [(k, v) for (k, v) in coeffs_items if len(v) > 0]
    if not coeffs_items:
        print(f"Warning: {dataset_name=} has not be processed at all")
        return

    assert set([k for (k, v) in mse_items]) == set([k for (k, v) in coeffs_items])

    # plot graph
    plt.figure()
    for k in the_mse_data.keys():
        assert k in the_coeffs_data.keys()
        x = the_mse_data[k]
        y = [np.var(cs) for cs in the_coeffs_data[k]]
        if len(x) != len(y):
            print(f"{dataset_name} {k} has {x} and {y}")
        plt.scatter(x, y, label=k)

    plt.ylabel("Variance")
    plt.xlabel("MSE (lower is better)")
    if dataset_name.endswith("sin_squared"):
        plt.xlim((0.6, 1.2))
    if dataset_name.endswith("step"):
        plt.xlim((1, 2))
    if dataset_name.endswith("wavelet"):
        plt.xlim((0, 0.2))
    plt.legend()
    plt.subplots_adjust(bottom=0.25)
    plt.title(f"MSE for {dataset_name=}" if title is None else title)
    plt.savefig(f"{the_plot_path}/{dataset_name}.png")
    plt.close()


def plot_mse_eigvals(the_plot_path, the_mse_data, the_eigvals_data, dataset_name, title=None):
    mse_items = list(the_mse_data.items())
    mse_items = [(k, v) for (k, v) in mse_items if len(v) > 0]
    if not mse_items:
        print(f"Warning: {dataset_name=} has not be processed at all")
        return
    coeffs_items = list(the_eigvals_data.items())
    coeffs_items = [(k, v) for (k, v) in coeffs_items if len(v) > 0]
    if not coeffs_items:
        print(f"Warning: {dataset_name=} has not be processed at all")
        return

    assert set([k for (k, v) in mse_items]) == set([k for (k, v) in coeffs_items])

    # plot graph
    plt.figure()
    for k in the_mse_data.keys():
        assert k in the_eigvals_data.keys()
        x = the_mse_data[k]
        y = [np.unique(np.abs(cs))[-2] for cs in the_eigvals_data[k]]
        if len(x) != len(y):
            print(f"{dataset_name} {k} has {x} and {y}")
        plt.scatter(x, y, label=k)

    plt.ylabel("2nd largest eigval")
    plt.xlabel("MSE (lower is better)")
    if dataset_name.endswith("sin_squared"):
        plt.xlim((0.6, 1.2))
    if dataset_name.endswith("step"):
        plt.xlim((1, 2))
    if dataset_name.endswith("wavelet"):
        plt.xlim((0, 0.2))
    plt.legend()
    plt.subplots_adjust(bottom=0.25)
    plt.title(f"MSE-2nd eigval for {dataset_name=}" if title is None else title)
    plt.savefig(f"{the_plot_path}/{dataset_name}.png")
    plt.close()


def generate_plots(intermediate_path, plot_path, repetitions, allow_partial_fold=True):

    DATASETS = ["fish_market", "life_expectancy", "medical_bill", "ols_cancer", "real_estate",
                "function_approximation_meyer_wavelet", "function_approximation_sin_squared",
                "function_approximation_step"]
    TECHNIQUES = ['random_kernel', 'scrambled_kernel-4', 'scrambled_kernel-3', 'combinatorial_sa_kernel',
                  'combinatorial_genetic_kernel', 'combinatorial_greedy_kernel']

    for dataset in DATASETS:
        mse_data = {technique: [] for technique in TECHNIQUES}
        eigvals_data = {technique: [] for technique in TECHNIQUES}
        coeffs_data = {technique: [] for technique in TECHNIQUES}

        # re-load data from folders
        for technique in TECHNIQUES:
            n = 0
            for rep in range(repetitions):
                if Path(f"{intermediate_path}/{dataset}/{technique}/{rep}/mse.npy").exists():
                    # load mse
                    mse = np.load(f"{intermediate_path}/{dataset}/{technique}/{rep}/mse.npy")
                    mse_data[technique].append(mse)
                    # load training gram matrix & calculate eigvals and variance
                    gram_train = np.load(f"{intermediate_path}/{dataset}/{technique}/{rep}/gram_train.npy")
                    train_eigs = np.linalg.eigvals(gram_train)
                    train_coeffs = gram_train[np.triu_indices(gram_train.shape[0], k=1)]
                    eigvals_data[technique].append(train_eigs)
                    coeffs_data[technique].append(train_coeffs)
                    n = gram_train.shape[0]

                elif allow_partial_fold:
                    print(f"Warning: dataset {dataset} has missing {technique=} {rep=}")
                else:
                    assert False, f"Warning: dataset {dataset} has missing {technique=} {rep=}"
            mse_data[technique] = np.array(mse_data[technique])
            mse_data[technique][mse_data[technique] == np.inf] = 2*n

        # plot_mse(f"{plot_path}/mse", mse_data, dataset_name=dataset, title=f"MSE {dataset}")
        # plot_eigvals(f"{plot_path}/eigenvalue", eigvals_data, dataset_name=dataset, title=f"EIGVALS {dataset}")
        # plot_eigvals(f"{plot_path}/eigenvalue", eigvals_data, dataset_name=f"{dataset}_zoom", title=f"EIGVALS {dataset}", ylim=(5, 25))
        # plot_coefficients(f"{plot_path}/variance", coeffs_data, dataset_name=dataset, title=f"VARIANCE {dataset}")
        # plot_mse_variance(f"{plot_path}/mse-variance", mse_data, coeffs_data, dataset_name=dataset)
        plot_mse_eigvals(f"{plot_path}/mse-eigenvalue", mse_data, eigvals_data, dataset_name=dataset)

# generate_plots(INTERMEDIATE_PATH, PLOT_PATH, repetitions=10)
