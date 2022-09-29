import datetime
import json
import pathlib
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns

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


DATASET_PATH = "datasets"
INTERMEDIATE_PATH = "intermediate"
PLOT_PATH = "plots"
np.save(f"{INTERMEDIATE_PATH}/SIMULATION_SEED.npy", SIMULATION_SEED)
REFRESH_DATASET = False
REFRESH_INTERMEDIATE = False

def get_kernel_values(ck, solution, X1, X2=None, bandwidth=1.0):
    if X2 is None:
        m = X1.shape[0]
        kernel_gram = np.eye(m)
        for i in range(m):
            for j in range(i + 1, m):
                value = ck(X1[i], X1[j], solution, bandwidth)
                kernel_gram[i][j] = value
                kernel_gram[j][i] = value
                print(".", end="")
    else:
        kernel_gram = np.zeros(shape=(len(X1), len(X2)))
        for i in range(len(X1)):
            for j in range(len(X2)):
                kernel_gram[i][j] = ck(X1[i], X2[j], solution, bandwidth)
                print(".", end="")
    return kernel_gram


def estimate_mse_from_gram(gram_train, gram_test, y_train, y_test):
    svr = SVR()
    svr.fit(gram_train, y_train.ravel())
    y_pred = svr.predict(gram_test)
    return mean_squared_error(y_test.ravel(), y_pred.ravel())


def estimate_mse(ck, solution, X_train, X_test, y_train, y_test, save_path=None):
    training_gram = get_kernel_values(ck, solution, X_train)
    testing_gram = get_kernel_values(ck, solution, X_test, X_train)
    if save_path:
        if issubclass(type(save_path), pathlib.Path):
            np.save(save_path / "training_gram.npy", training_gram)
            np.save(save_path / "testing_gram.npy", testing_gram)
        else:
            np.save(f"{save_path}/training_gram.npy", training_gram)
            np.save(f"{save_path}/testing_gram.npy", testing_gram)
    try:
        return estimate_mse_from_gram(training_gram, testing_gram, y_train, y_test)
    except:
        training_gram = training_gram + np.eye(training_gram.shape[0])
        return estimate_mse_from_gram(training_gram, testing_gram, y_train, y_test)

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


def run_combinatorial_initialization_kernels(dataset_path, repetition, X_train, X_validation, X_test, y_train, y_validation, y_test, n_layers, ck):

    TECHNIQUES = ['combinatorial_genetic_kernel', 'combinatorial_sa_kernel', 'combinatorial_greedy_kernel']
    for technique in TECHNIQUES:

        optimized_folder = Path(f"{dataset_path}/{technique}/{repetition}")
        initialization_parent_folder = Path(f"{dataset_path}/init_{technique}")
        initialization_folder = initialization_parent_folder / str(repetition)

        if not optimized_folder.exists():
            print(f"Skipping {dataset_path=} {technique=} {repetition=} (MISSING OPTIMIZED PART)")
            continue

        if (initialization_folder / "mse.npy").exists():
            print(f"Skipping {dataset_path=} {technique=} {repetition=} (ALREADY EXISTS)")
            continue

        initialization_parent_folder.mkdir(exist_ok=True)
        initialization_folder.mkdir(exist_ok=True)

        # start with initial solution and see the gram matrices and mse
        if (optimized_folder / "initial_solution.npy").exists():

            # create kernel using initialization solution
            initial_solution = np.load(optimized_folder / "initial_solution.npy")
            np.save(initialization_folder / "initial_solution.npy", initial_solution)
            mse = estimate_mse(ck, initial_solution, X_train, X_test, y_train, y_test, initialization_folder)
            np.save(initialization_folder / "mse.npy", np.array(mse))

        elif (optimized_folder / "initial_items.npy").exists():

            initial_items = np.load(optimized_folder / "initial_items.npy")
            np.save(initialization_folder / "initial_items.npy", initial_items)

            mse_list = []
            for initial_solution in initial_items:
                mse = estimate_mse(ck, initial_solution, X_train, X_test, y_train, y_test, initialization_folder)
                mse_list.append(mse)

            np.save(initialization_folder / "mse.npy", np.array(mse_list))

        else:
            raise ValueError("Missing initialization info")

the_n_layers = 3
combinatorial_kernel_2 = CombinatorialKernel(2, the_n_layers)
combinatorial_kernel_5 = CombinatorialKernel(5, the_n_layers)

def run_simulations(n_layers, epochs, lr, repetitions=10):

    global SCRAMBLED_3_SEED, SCRAMBLED_4_SEED, GENETIC_SEED, scrambled_4_seed_index, scrambled_3_seed_index, genetic_seed_index
    global combinatorial_kernel_2, combinatorial_kernel_5

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

            # DIFFERENCE OF COMBINATORIAL KERNELS BEFORE AND AFTER THE INITIALIZATION
            run_combinatorial_initialization_kernels(f"{INTERMEDIATE_PATH}/{dataset}", repetition,
                                                     X_train, X_validation, X_test, y_train, y_validation, y_test,
                                                     n_layers, ck)

            json.dump(timing, open(f"{INTERMEDIATE_PATH}/{dataset}/timing.json", "w"))


# run_simulations(the_n_layers, 1000, 0.01, 10)


# =====================================================================================
# 3. GENERATE PLOTS ===================================================================
# =====================================================================================

def load_mse_data(reps):

    mse_data = pd.DataFrame(columns=['mse_change', 'mse_pre', 'mse_post', 'dataset', 'technique',
                                     'pre_gram', 'post_gram', 'pre_gram_test', 'post_gram_test'])

    DATASETS = ["fish_market", "function_approximation_meyer_wavelet", "function_approximation_sin_squared",
                "function_approximation_step", "life_expectancy",
                "medical_bill", "ols_cancer", "real_estate"]

    TECHNIQUES = ['combinatorial_genetic_kernel', 'combinatorial_sa_kernel', 'combinatorial_greedy_kernel']

    for dataset in DATASETS:
        for technique in TECHNIQUES:
            print(f"\n{dataset=} {technique=}")
            if technique.startswith("combinatorial_genetic"):
                for i in range(reps):
                    pre_opt_mse = np.load(f"{INTERMEDIATE_PATH}/{dataset}/init_{technique}/{i}/mse.npy")
                    pre_opt_mse = np.min(pre_opt_mse).item()
                    post_opt_mse = np.load(f"{INTERMEDIATE_PATH}/{dataset}/{technique}/{i}/mse.npy")
                    relative_change = (pre_opt_mse - post_opt_mse) / np.abs(pre_opt_mse)
                    print(f"Before optimization {pre_opt_mse}, after optimization {post_opt_mse}, {relative_change=}")
                    pre_gram = np.load(f"{INTERMEDIATE_PATH}/{dataset}/init_{technique}/{i}/training_gram.npy")
                    post_gram = np.load(f"{INTERMEDIATE_PATH}/{dataset}/{technique}/{i}/gram_train.npy")
                    pre_gram_test = np.load(f"{INTERMEDIATE_PATH}/{dataset}/init_{technique}/{i}/testing_gram.npy")
                    post_gram_test = np.load(f"{INTERMEDIATE_PATH}/{dataset}/{technique}/{i}/gram_test.npy")
                    mse_data.loc[len(mse_data)] = {'mse_change': relative_change,
                                                   'mse_pre': pre_opt_mse,
                                                   'mse_post': post_opt_mse,
                                                   'dataset': dataset,
                                                   'technique': technique,
                                                   'pre_gram': pre_gram,
                                                   'post_gram': post_gram,
                                                   'pre_gram_test': pre_gram_test,
                                                   'post_gram_test': post_gram_test}
            else:
                for i in range(reps):
                    pre_opt_mse = np.load(f"{INTERMEDIATE_PATH}/{dataset}/init_{technique}/{i}/mse.npy")
                    post_opt_mse = np.load(f"{INTERMEDIATE_PATH}/{dataset}/{technique}/{i}/mse.npy")
                    relative_change = (pre_opt_mse - post_opt_mse) / np.abs(pre_opt_mse)
                    print(f"Before optimization {pre_opt_mse}, after optimization {post_opt_mse}, {relative_change=}")
                    pre_gram = np.load(f"{INTERMEDIATE_PATH}/{dataset}/init_{technique}/{i}/training_gram.npy")
                    post_gram = np.load(f"{INTERMEDIATE_PATH}/{dataset}/{technique}/{i}/gram_train.npy")
                    pre_gram_test = np.load(f"{INTERMEDIATE_PATH}/{dataset}/init_{technique}/{i}/testing_gram.npy")
                    post_gram_test = np.load(f"{INTERMEDIATE_PATH}/{dataset}/{technique}/{i}/gram_test.npy")
                    mse_data.loc[len(mse_data)] = {'mse_change': relative_change,
                                                   'mse_pre': pre_opt_mse,
                                                   'mse_post': post_opt_mse,
                                                   'dataset': dataset,
                                                   'technique': technique,
                                                   'pre_gram': pre_gram,
                                                   'post_gram': post_gram,
                                                   'pre_gram_test': pre_gram_test,
                                                   'post_gram_test': post_gram_test}

    mse_data['dataset'] = mse_data['dataset'].replace(
        {
            "fish_market": 'FM',
            "function_approximation_meyer_wavelet": 'XM',
            "function_approximation_sin_squared": 'XS',
            "function_approximation_step": 'XT',
            "life_expectancy": 'LE',
            "medical_bill": 'MB',
            "ols_cancer": 'OC',
            "real_estate": 'RE'
        }
    )
    mse_data['technique'] = mse_data['technique'].replace(
        {
            'combinatorial_genetic_kernel': 'Genetic',
            'combinatorial_sa_kernel': 'Sim. Annealing',
            'combinatorial_greedy_kernel': 'Greedy'
        }
    )
    return mse_data


def print_table_data(mse_data):
    table_data = mse_data.drop(columns=['pre_gram', 'post_gram', 'pre_gram_test', 'post_gram_test'], axis=1,
                             inplace=False)
    table_data = table_data.groupby(['dataset', 'technique']).agg({'mse_change': 'max', 'mse_pre': 'min', 'mse_post': 'min'})[
        ['mse_change', 'mse_pre', 'mse_post']].reset_index()
    print(table_data.to_string())


def create_increment_performances_plot(mse_data):

    order = ['XS', 'XT', 'XM', 'OC', 'RE', 'FM', 'MB', 'LE']
    hue_order = ['Genetic', 'Greedy', 'Sim. Annealing']
    # baseline
    plt.hlines(0.0, colors='red', xmin=-0.5, xmax=7.5, zorder=-1)
    # plot distributions of solutions
    sns.stripplot(x="dataset", y="mse_change", hue="technique", data=mse_data, dodge=True, alpha=.25, zorder=1, order=order, hue_order=hue_order)
    # plot maximum
    max_data = mse_data.drop(columns=['pre_gram', 'post_gram', 'pre_gram_test', 'post_gram_test'], axis=1, inplace=False)
    max_data = max_data.groupby(['dataset', 'technique']).max()
    print(max_data)
    max_data = max_data.reset_index()
    ax = sns.pointplot(x="dataset", y="mse_change", hue="technique", data=max_data, join=False, dodge=0.8 - 0.8/3, order=order, hue_order=hue_order)
    handles, labels = ax.get_legend_handles_labels()
    plt.ylabel('$\\frac{mse\\,init - mse\\,opt}{mse\\,init}$ (higher is better)')
    plt.legend(handles[0:3], labels[0:3], loc='lower right', borderaxespad=0.2)
    plt.ylim((-0.5, 0.75))
    plt.xlim((-0.5, 7.5))
    plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/sns-mse/mse_change.png")
    plt.close('all')


def create_eigenvalue_single_comparison(mse_data, dataset, technique, reps):
    sub_mse_data = mse_data[(mse_data['dataset'] == dataset) & (mse_data['technique'] == technique)]
    the_eigvals_data = pd.DataFrame(columns=['ith', 'value', 'type'])
    for rep in range(reps):
        row = sub_mse_data.iloc[rep]
        pre_eigvals = np.sort(np.linalg.eigvals(row['pre_gram']).real)[::-1]
        post_eigvals = np.sort(np.linalg.eigvals(row['post_gram']).real)[::-1]
        for i in range(10):
            the_eigvals_data.loc[len(the_eigvals_data)] = {'ith': i, 'value': pre_eigvals[i], 'type': 'pre'}
            the_eigvals_data.loc[len(the_eigvals_data)] = {'ith': i, 'value': post_eigvals[i], 'type': 'post'}
    sns.lineplot(data=the_eigvals_data, x='ith', y='value', hue='type')


def create_eigenvalue_technique_comparison(mse_data, dataset, reps):
    sub_mse_data = mse_data[(mse_data['dataset'] == dataset)]
    the_eigvals_data = pd.DataFrame(columns=['ith', 'value', 'type', 'technique'])
    for technique in ['Genetic', 'Greedy', 'Sim. Annealing']:
        for rep in range(reps):
            row = sub_mse_data[sub_mse_data['technique'] == technique].iloc[rep]
            pre_eigvals = np.sort(np.linalg.eigvals(row['pre_gram']).real)[::-1]
            post_eigvals = np.sort(np.linalg.eigvals(row['post_gram']).real)[::-1]
            for i in range(10):
                the_eigvals_data.loc[len(the_eigvals_data)] = {'ith': i, 'technique': technique, 'value': pre_eigvals[i], 'type': 'pre'}
                the_eigvals_data.loc[len(the_eigvals_data)] = {'ith': i, 'technique': technique, 'value': post_eigvals[i], 'type': 'post'}
    sns.lineplot(data=the_eigvals_data, x='ith', y='value', style='technique', hue='type')


def create_eigenvalue_change_plot(mse_data, reps):
    for dataset in ['XS', 'XT', 'XM', 'OC', 'RE', 'FM', 'MB']:
        create_eigenvalue_technique_comparison(mse_data, dataset, reps)
        plt.tight_layout()
        plt.savefig(f"{PLOT_PATH}/sns-mse/eigvals_{dataset}.png")
        plt.close('all')


def create_variance_single_change_plot(mse_data, dataset, technique, reps):
    sub_mse_data = mse_data[(mse_data['dataset'] == dataset) & (mse_data['technique'] == technique)]
    the_variance_data = pd.DataFrame(columns=['mse', 'variance', 'type'])
    for rep in range(reps):
        row = sub_mse_data.iloc[rep]
        gram_pre = row['pre_gram']
        gram_post = row['post_gram']
        var_pre = np.var(gram_pre[np.triu_indices(gram_pre.shape[0], k=1)]).item()
        var_post = np.var(gram_post[np.triu_indices(gram_post.shape[0], k=1)]).item()
        for i in range(10):
            the_variance_data.loc[len(the_variance_data)] = {'mse': row['mse_pre'], 'variance': var_pre, 'type': 'pre'}
            the_variance_data.loc[len(the_variance_data)] = {'mse': row['mse_post'], 'variance': var_post, 'type': 'post'}
    the_variance_data['type'] = the_variance_data['type'].astype('category')
    the_variance_data['mse'] = the_variance_data['mse'].astype(float)
    the_variance_data['variance'] = the_variance_data['variance'].astype(float)
    print(the_variance_data)
    sns.jointplot(data=the_variance_data, x='mse', y='variance', hue='type')
    plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/sns-mse/variance_{dataset}_{technique}.png")
    plt.close('all')


def create_variance_technique_change_plot(mse_data, dataset, reps):
    techniques = ['Genetic', 'Greedy', 'Sim. Annealing']
    the_variance_data = pd.DataFrame(columns=['mse', 'variance', 'type', 'technique'])
    for technique in techniques:
        sub_mse_data = mse_data[(mse_data['dataset'] == dataset) & (mse_data['technique'] == technique)]
        for rep in range(reps):
            row = sub_mse_data.iloc[rep]
            gram_pre = row['pre_gram']
            gram_post = row['post_gram']
            var_pre = np.var(gram_pre[np.triu_indices(gram_pre.shape[0], k=1)]).item()
            var_post = np.var(gram_post[np.triu_indices(gram_post.shape[0], k=1)]).item()
            assert var_pre >= 0 and var_post >= 0
            for i in range(10):
                the_variance_data.loc[len(the_variance_data)] = {'technique': technique, 'mse': row['mse_pre'], 'variance': var_pre, 'type': 'pre'}
                the_variance_data.loc[len(the_variance_data)] = {'technique': technique, 'mse': row['mse_post'], 'variance': var_post, 'type': 'post'}
        the_variance_data['type'] = the_variance_data['type'].astype('category')
        the_variance_data['mse'] = the_variance_data['mse'].astype(float)
        the_variance_data['variance'] = the_variance_data['variance'].astype(float)
    print(the_variance_data)
    # generate plot
    sns.relplot(data=the_variance_data, x='mse', y='variance', hue='type', style='technique')
    plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/sns-mse/variance_{dataset}.png")
    plt.close('all')


def emulate_shots(p, shots):
    assert 0 <= p <= 1
    counts = 0
    for _ in range(shots):
        if np.random.uniform() < p:
            counts += 1
    return counts / shots


def approximate_gram(gram, shots):
    gram = gram.copy()
    for i in range(gram.shape[0]):
        for j in range(gram.shape[1]):
            gram[i][j] = emulate_shots(gram[i][j], shots)
    return gram


def create_variance_technique_approximated_change_plot(mse_data, dataset, y_train, y_test, reps, shots):
    techniques = ['Genetic', 'Greedy', 'Sim. Annealing']
    the_variance_data = pd.DataFrame(columns=['mse', 'variance', 'type', 'technique'])
    for technique in techniques:
        sub_mse_data = mse_data[(mse_data['dataset'] == dataset) & (mse_data['technique'] == technique)]
        for rep in range(reps):
            print(f"Processing {technique=} {rep=}")
            row = sub_mse_data.iloc[rep]
            gram_pre = approximate_gram(row['pre_gram'], shots)
            gram_post = approximate_gram(row['post_gram'], shots)
            gram_pre_test = approximate_gram(row['pre_gram_test'], shots)
            gram_post_test = approximate_gram(row['post_gram_test'], shots)
            var_pre = np.var(gram_pre[np.triu_indices(gram_pre.shape[0], k=1)]).item()
            var_post = np.var(gram_post[np.triu_indices(gram_post.shape[0], k=1)]).item()
            mse_pre = estimate_mse_from_gram(gram_pre, gram_pre_test, y_train, y_test)
            mse_post = estimate_mse_from_gram(gram_post, gram_post_test, y_train, y_test)
            assert var_pre >= 0 and var_post >= 0
            for i in range(10):
                the_variance_data.loc[len(the_variance_data)] = {'technique': technique, 'mse': mse_pre, 'variance': var_pre, 'type': 'pre'}
                the_variance_data.loc[len(the_variance_data)] = {'technique': technique, 'mse': mse_post, 'variance': var_post, 'type': 'post'}
        the_variance_data['type'] = the_variance_data['type'].astype('category')
        the_variance_data['mse'] = the_variance_data['mse'].astype(float)
        the_variance_data['variance'] = the_variance_data['variance'].astype(float)
    print(the_variance_data)
    # generate plot
    sns.relplot(data=the_variance_data, x='mse', y='variance', hue='type', style='technique')
    plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/sns-mse/variance_{dataset}_{shots}.png")
    plt.close('all')


def create_variance_technique_approximated_all_change_plot(mse_data, dataset, y_train, y_test, reps):
    techniques = ['Genetic', 'Greedy', 'Sim. Annealing']
    the_variance_data = pd.DataFrame(columns=['mse', 'variance', 'type', 'technique', 'shots'])
    # non approximated
    for technique in techniques:
        sub_mse_data = mse_data[(mse_data['dataset'] == dataset) & (mse_data['technique'] == technique)]
        for rep in range(reps):
            row = sub_mse_data.iloc[rep]
            gram_pre = row['pre_gram']
            gram_post = row['post_gram']
            var_pre = np.var(gram_pre[np.triu_indices(gram_pre.shape[0], k=1)]).item()
            var_post = np.var(gram_post[np.triu_indices(gram_post.shape[0], k=1)]).item()
            assert var_pre >= 0 and var_post >= 0
            for i in range(10):
                the_variance_data.loc[len(the_variance_data)] = {'shots': 0, 'technique': technique, 'mse': row['mse_pre'], 'variance': var_pre, 'type': 'pre'}
                the_variance_data.loc[len(the_variance_data)] = {'shots': 0, 'technique': technique, 'mse': row['mse_post'], 'variance': var_post, 'type': 'post'}
        the_variance_data['type'] = the_variance_data['type'].astype('category')
        the_variance_data['mse'] = the_variance_data['mse'].astype(float)
        the_variance_data['variance'] = the_variance_data['variance'].astype(float)
    # approximated
    for shots in [16, 128, 1024]:
        for technique in techniques:
            sub_mse_data = mse_data[(mse_data['dataset'] == dataset) & (mse_data['technique'] == technique)]
            for rep in range(reps):
                print(f"Processing {technique=} {rep=}")
                row = sub_mse_data.iloc[rep]
                gram_pre = approximate_gram(row['pre_gram'], shots)
                gram_post = approximate_gram(row['post_gram'], shots)
                gram_pre_test = approximate_gram(row['pre_gram_test'], shots)
                gram_post_test = approximate_gram(row['post_gram_test'], shots)
                var_pre = np.var(gram_pre[np.triu_indices(gram_pre.shape[0], k=1)]).item()
                var_post = np.var(gram_post[np.triu_indices(gram_post.shape[0], k=1)]).item()
                mse_pre = estimate_mse_from_gram(gram_pre, gram_pre_test, y_train, y_test)
                mse_post = estimate_mse_from_gram(gram_post, gram_post_test, y_train, y_test)
                assert var_pre >= 0 and var_post >= 0
                for i in range(10):
                    the_variance_data.loc[len(the_variance_data)] = {'shots': shots, 'technique': technique, 'mse': mse_pre, 'variance': var_pre, 'type': 'pre'}
                    the_variance_data.loc[len(the_variance_data)] = {'shots': shots, 'technique': technique, 'mse': mse_post, 'variance': var_post, 'type': 'post'}
            the_variance_data['type'] = the_variance_data['type'].astype('category')
            the_variance_data['mse'] = the_variance_data['mse'].astype(float)
            the_variance_data['variance'] = the_variance_data['variance'].astype(float)
    return the_variance_data


def plot_variance_technique_approximated_all_change_plot(variance_data, dataset):
    sns.relplot(data=variance_data, x='mse', y='variance', hue='shots', style='type')
    plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/sns-mse/variance_{dataset}.png")
    plt.close('all')


def create_variance_change_plot(mse_data, reps):
    for dataset in ['XS', 'XT', 'XM', 'OC', 'RE', 'FM', 'MB']:
        create_variance_technique_change_plot(mse_data, dataset, reps)


def create_variance_shots_change_plot(mse_data, dataset, reps):
    for dataset in ['XS', 'XT', 'XM', 'OC', 'RE', 'FM', 'MB']:
        create_variance_technique_change_plot(mse_data, dataset, reps)




the_mse_data = load_mse_data(10)
print_table_data(the_mse_data)
# create_increment_performances_plot(the_mse_data)
# # create_eigenvalue_change_plot(the_mse_data, 10)
# y_train = np.load(f"{DATASET_PATH}/fish_market/y_train.npy")
# y_test = np.load(f"{DATASET_PATH}/fish_market/y_test.npy")
# # create_variance_technique_approximated_change_plot(the_mse_data, 'FM', y_train, y_test, 10, 16)
# # create_variance_technique_approximated_change_plot(the_mse_data, 'FM', y_train, y_test, 10, 128)
# # create_variance_technique_approximated_change_plot(the_mse_data, 'FM', y_train, y_test, 10, 1024)
# # create_variance_technique_change_plot(the_mse_data, 'FM', 10)
# the_variance_data = create_variance_technique_approximated_all_change_plot(the_mse_data, 'FM', y_train, y_test, 10)
# plot_variance_technique_approximated_all_change_plot(the_variance_data, 'FM')
# # create_variance_change_plot(the_mse_data, 10)

# =====================================================================================
# 4. GENERATE DETAILED PLOTS ==========================================================
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
    plt.xlabel("Approach")
    plt.ylabel("MSE (lower is better)")
    plt.savefig(f"{the_plot_path}/{dataset_name}.png")
    plt.close()


def plot_eigvals(the_plot_path, the_eigvals_data, dataset_name, title=None, ylim=None):
    eigvals_items = list(the_eigvals_data.items())
    eigvals_items = [(k, v) for (k, v) in eigvals_items if len(v) > 0]
    if not eigvals_items:
        print(f"Warning: {dataset_name=} has not be processed at all")
        return

    # plot graph
    try:
        plt.figure()
        for i in range(len(eigvals_items[0][1])):
            violins = [v[i] for k, v in eigvals_items]
            # print(f"{len(eigvals_items[0][1])=} {len(eigvals_items)=} {len(violins)=}")
            plt.violinplot(violins, range(len(eigvals_items)),
                           widths=0.3, showmeans=True, showextrema=True, showmedians=True)
        if ylim is not None:
            plt.ylim(ylim)
        plt.ylabel("Eigenvalue")
        plt.xticks(range(len(eigvals_items)), [k for k, v in eigvals_items], rotation=45)
        plt.subplots_adjust(bottom=0.25)
        plt.title(f"Eigenvalue distribution for for {dataset_name=}" if title is None else title)
        plt.savefig(f"{the_plot_path}/{dataset_name}.png")
        plt.close()
    except:
        print("ERROR! THE SIMULATION HAS BEEN STOPPED MIDWAY")


def plot_coefficients(the_plot_path, the_coeffs_data, dataset_name, title=None):
    coeffs_items = list(the_coeffs_data.items())
    coeffs_items = [(k, v) for (k, v) in coeffs_items if len(v) > 0]
    if not coeffs_items:
        print(f"Warning: {dataset_name=} has not be processed at all")
        return

    try:
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
    except:
        print("ERROR! THE SIMULATION HAS BEEN STOPPED MIDWAY")


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


# def generate_plots(intermediate_path, plot_path, repetitions, allow_partial_fold=True):
#
#     DATASETS = ["fish_market", "life_expectancy", "medical_bill", "ols_cancer", "real_estate",
#                 "function_approximation_meyer_wavelet", "function_approximation_sin_squared",
#                 "function_approximation_step"]
#     TECHNIQUES = ['random_kernel', 'scrambled_kernel-4', 'scrambled_kernel-3', 'combinatorial_sa_kernel',
#                   'combinatorial_genetic_kernel', 'combinatorial_greedy_kernel']
#
#     for dataset in DATASETS:
#         mse_data = {technique: [] for technique in TECHNIQUES}
#         eigvals_data = {technique: [] for technique in TECHNIQUES}
#         coeffs_data = {technique: [] for technique in TECHNIQUES}
#
#         # re-load data from folders
#         for technique in TECHNIQUES:
#             n = 0
#             for rep in range(repetitions):
#                 if Path(f"{intermediate_path}/{dataset}/{technique}/{rep}/mse.npy").exists():
#                     # load mse
#                     mse = np.load(f"{intermediate_path}/{dataset}/{technique}/{rep}/mse.npy")
#                     mse_data[technique].append(mse)
#                     # load training gram matrix & calculate eigvals and variance
#                     gram_train = np.load(f"{intermediate_path}/{dataset}/{technique}/{rep}/gram_train.npy")
#                     train_eigs = np.linalg.eigvals(gram_train)
#                     train_coeffs = gram_train[np.triu_indices(gram_train.shape[0], k=1)]
#                     eigvals_data[technique].append(train_eigs)
#                     coeffs_data[technique].append(train_coeffs)
#                     n = gram_train.shape[0]
#
#                 elif allow_partial_fold:
#                     print(f"Warning: dataset {dataset} has missing {technique=} {rep=}")
#                 else:
#                     assert False, f"Warning: dataset {dataset} has missing {technique=} {rep=}"
#             mse_data[technique] = np.array(mse_data[technique])
#             mse_data[technique][mse_data[technique] == np.inf] = 2*n
#
#         plot_mse(f"{plot_path}/mse", mse_data, dataset_name=dataset, title=f"MSE {dataset}")
#         # plot_eigvals(f"{plot_path}/eigenvalue", eigvals_data, dataset_name=dataset, title=f"EIGVALS {dataset}")
#         # plot_eigvals(f"{plot_path}/eigenvalue", eigvals_data, dataset_name=f"{dataset}_zoom", title=f"EIGVALS {dataset}", ylim=(5, 25))
#         # plot_coefficients(f"{plot_path}/variance", coeffs_data, dataset_name=dataset, title=f"VARIANCE {dataset}")
#         # plot_mse_variance(f"{plot_path}/mse-variance", mse_data, coeffs_data, dataset_name=dataset)
#         # plot_mse_eigvals(f"{plot_path}/mse-eigenvalue", mse_data, eigvals_data, dataset_name=dataset)


def new_generate_plots(intermediate_path, plot_path, repetitions, allow_partial_fold=True):

    the_mse_df = pd.DataFrame(columns=['technique', 'mse', 'dataset'])

    DATASETS = ["fish_market", "life_expectancy", "medical_bill", "ols_cancer", "real_estate",
                "function_approximation_meyer_wavelet", "function_approximation_sin_squared",
                "function_approximation_step"]
    TECHNIQUES = ['random_kernel', 'scrambled_kernel-4', 'scrambled_kernel-3', 'combinatorial_sa_kernel',
                  'combinatorial_genetic_kernel', 'combinatorial_greedy_kernel']

    for dataset in DATASETS:
         for technique in TECHNIQUES:
             for rep in range(repetitions):
                 if Path(f"{intermediate_path}/{dataset}/{technique}/{rep}/mse.npy").exists():
                     # load mse
                     mse = np.load(f"{intermediate_path}/{dataset}/{technique}/{rep}/mse.npy")
                     if mse >= np.inf: mse = 5.0
                     assert mse >= 0
                     the_mse_df.loc[len(the_mse_df)] = {'mse': float(mse), 'technique': str(technique), 'dataset': str(dataset)}

    the_mse_df['dataset'] = the_mse_df['dataset'].replace(
        {
            "fish_market": 'FM',
            "function_approximation_meyer_wavelet": 'XM',
            "function_approximation_sin_squared": 'XS',
            "function_approximation_step": 'XT',
            "life_expectancy": 'LE',
            "medical_bill": 'MB',
            "ols_cancer": 'OC',
            "real_estate": 'RE'
        }
    )
    the_mse_df['dataset'] = the_mse_df['dataset'].astype('category')
    the_mse_df_technique = the_mse_df['technique'].copy()
    the_mse_df['technique'] = the_mse_df['technique'].replace(
        {
            'combinatorial_genetic_kernel': 'C',
            'combinatorial_sa_kernel': 'C',
            'combinatorial_greedy_kernel': 'C',
            'scrambled_kernel-4': 'HE3',
            'scrambled_kernel-3': 'HE4',
            'random_kernel': 'R'
        }
    )
    the_mse_df['technique'] = the_mse_df['technique'].astype('category')
    sns.barplot(data=the_mse_df, x="dataset", y="mse", hue="technique",
                order=['OC', 'RE', 'FM', 'MB', 'LE', 'XM', 'XS', 'XT'],
                hue_order=['C', 'HE3', 'HE4', 'R']) # , order=['HE4', 'HE3', 'R', 'C'])
    plt.xlabel("Dataset")
    plt.ylabel("MSE (lower is better)")
    plt.ylim((0, 0.50))
    plt.savefig(f"{plot_path}/mse_approaches.png")
    plt.close()

    the_mse_df['technique'] = the_mse_df_technique.replace(
        {
            'combinatorial_genetic_kernel': 'CGEN',
            'combinatorial_sa_kernel': 'CSA',
            'combinatorial_greedy_kernel': 'CGLS',
            'scrambled_kernel-4': 'HE3',
            'scrambled_kernel-3': 'HE4',
            'random_kernel': 'R'
        }
    )
    for dataset in ['OC', 'RE', 'FM', 'MB', 'LE', 'XM', 'XS', 'XT']:
        sns.barplot(data=the_mse_df[the_mse_df['dataset'] == dataset], x="technique", y="mse",
                    order=['CGEN', 'CSA', 'CGLS', 'HE3', 'HE4', 'R'],
                    capsize=0.4, linewidth=3)
        plt.xlabel("Technique")
        plt.ylabel("MSE (lower is better)")
        # plt.ylim((0, 0.50))
        plt.savefig(f"{plot_path}/mse_{dataset}.png")
        plt.close()


new_generate_plots(INTERMEDIATE_PATH, PLOT_PATH, repetitions=10)

