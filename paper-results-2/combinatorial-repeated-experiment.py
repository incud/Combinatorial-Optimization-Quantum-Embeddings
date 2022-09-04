import datetime
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from quask.datasets import *
from quask.random_kernel import RandomKernel
from quask.trainable_kernel import TrainableKernel
from quask.combinatorial_kernel import *
from quask.combinatorial_kernel_optimization_simanneal import CombinatorialKernelSimulatedAnnealingTraining
from quask.combinatorial_kernel_greedy import CombinatorialKernelGreedySearch


# =====================================================================================
# 0. UTILITY FUNCTIONS ================================================================
# =====================================================================================


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


reset_seed(43843933)
SIMULATION_SEED = np.random.randint(100000, size=(1000,))
simulation_seed_index = 0


def get_next_seed():
    global SIMULATION_SEED, simulation_seed_index
    ret = SIMULATION_SEED[simulation_seed_index]
    simulation_seed_index += 1
    return int(ret)


DATASET_PATH = "paper-results/datasets"
INTERMEDIATE_PATH = "paper-results/intermediate"
PLOT_PATH = "paper-results/plots"
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
save_datasets(f"{DATASET_PATH}/life_expectancy", dataset, 5, 180)
# save_datasets(f"{DATASET_PATH}/life_expectancy", dataset, 3, 15)

dataset = load_medical_bill_dataset()
reset_seed(39284)
print(f"Dataset 'medical bill' has {dataset['X'].shape[0]} elements with {dataset['X'].shape[1]} features")
save_datasets(f"{DATASET_PATH}/medical_bill", dataset, 5, 180)
# save_datasets(f"{DATASET_PATH}/medical_bill", dataset, 3, 15)

dataset = load_ols_cancer_dataset()
reset_seed(19475)
print(f"Dataset 'ols cancer' has {dataset['X'].shape[0]} elements with {dataset['X'].shape[1]} features")
save_datasets(f"{DATASET_PATH}/ols_cancer", dataset, 5, 180)
# save_datasets(f"{DATASET_PATH}/ols_cancer", dataset, 3, 15)

dataset = load_real_estate_dataset()
reset_seed(24298)
print(f"Dataset 'real estate' has {dataset['X'].shape[0]} elements with {dataset['X'].shape[1]} features")
save_datasets(f"{DATASET_PATH}/real_estate", dataset, 5, 180)
# save_datasets(f"{DATASET_PATH}/real_estate", dataset, 3, 15)


# =====================================================================================
# 2. GENERATE INTERMEDIATE RESULTS ====================================================
# =====================================================================================

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
    np.save(f"{save_path}/gram_train.npy", gram_train)
    np.save(f"{save_path}/gram_test.npy", gram_test)
    np.save(f"{save_path}/mse.npy", np.array(mse))


def run_simulations(n_layers, epochs, lr, repetitions=10):

    DATASETS = ["fish_market", "function_approximation_meyer_wavelet", "function_approximation_sin_squared",
                "function_approximation_step", "life_expectancy",
                "medical_bill", "ols_cancer", "real_estate"]

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

            timing['random_kernel_start'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            print(f"Random kernel starts at {timing['random_kernel_start']}")
            seed = get_next_seed()
            reset_seed(seed)
            run_random_kernel(f"{INTERMEDIATE_PATH}/{dataset}/random_kernel/{repetition}", X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, seed)
            timing['random_kernel_end'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NOT FIXED PATH
            # timing['trainable_kernel_start'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            # print(f"Trainable kernel starts at {timing['trainable_kernel_start']}")
            # seed = get_next_seed()
            # reset_seed(seed)
            # run_trainable_kernel(save_path + "/trainable_kernel", X_train, X_validation, X_test, y_train, y_validation,
            #                      y_test, n_layers, initial_solution, epochs, lr)
            # timing['trainable_kernel_end'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            timing['combinatorial_sa_kernel_start'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            print(f"Combinatorial (sa) kernel starts at {timing['combinatorial_sa_kernel_start']}")
            seed = get_next_seed()
            reset_seed(seed)
            run_combinatorial_sa_kernel(f"{INTERMEDIATE_PATH}/{dataset}/combinatorial_sa_kernel/{repetition}",
                X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, initial_solution, epochs)
            timing['combinatorial_sa_kernel_end'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            timing['combinatorial_greedy_kernel_start'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            print(f"Combinatorial (greedy) kernel starts at {timing['combinatorial_greedy_kernel_start']}")
            seed = get_next_seed()
            reset_seed(seed)
            run_combinatorial_greedy_kernel(f"{INTERMEDIATE_PATH}/{dataset}/combinatorial_greedy_kernel/{repetition}",
                X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, initial_solution)
            timing['combinatorial_greedy_kernel_end'] = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            json.dump(timing, open(f"{INTERMEDIATE_PATH}/{dataset}/timing.json", "w"))


run_simulations(3, 1000, 0.01, 10)


# =====================================================================================
# 3. GENERATE PLOTS ===================================================================
# =====================================================================================

# FIX PATH!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
def generate_plots(intermediate_path, plot_path, allow_partial_fold=True):

    FOLDS = ["TR1-V2-TE3", "TR1-V3-TE2", "TR2-V1-TE3", "TR2-V3-TE1", "TR3-V1-TE2", "TR3-V2-TE1"]
    DATASETS = ["fish_market", "life_expectancy", "medical_bill", "ols_cancer", "real_estate",
                "function_approximation_meyer_wavelet", "function_approximation_sin_squared",
                "function_approximation_step"]
    TECHNIQUES = ['random_kernel', 'trainable_kernel', 'combinatorial_sa_kernel', 'combinatorial_greedy_kernel']

    for dataset in DATASETS:
        mse_dataset = {technique: [] for technique in TECHNIQUES}

        # load data
        for fold in FOLDS:
            for technique in TECHNIQUES:
                if Path(f"{intermediate_path}/{dataset}/{fold}/{technique}/mse.npy").exists():
                    mse = np.load(f"{intermediate_path}/{dataset}/{fold}/{technique}/mse.npy")
                    mse_dataset[technique].append(mse)
                elif allow_partial_fold:
                    print(f"Warning: dataset {dataset} has missing {fold=} {technique=}")
                else:
                    assert False, f"Warning: dataset {dataset} has missing {fold=} {technique=}"

        mse_items = list(mse_dataset.items())
        mse_items = [(k, v) for (k, v) in mse_items if len(v) > 0]
        if not mse_items:
            print(f"Warining: {dataset=} has not be processed at all")
            continue

        # plot graph
        plt.figure()
        plt.violinplot(
            [v for k, v in mse_items],
            range(len(mse_items)),
            widths=0.3,
            showmeans=True,
            showextrema=True,
            showmedians=True
        )
        plt.ylabel("MSE (lower is better)")
        plt.xticks(range(len(mse_items)), [k for k, v in mse_items], rotation=45)
        plt.subplots_adjust(bottom=0.25)
        plt.title(f"MSE for {dataset=}")
        plt.savefig(f"{plot_path}/{dataset}.png")
        plt.close()


# generate_plots(INTERMEDIATE_PATH, PLOT_PATH)
