import jax
import os
from jax.config import config
config.update("jax_enable_x64", True)
import pennylane as qml
import numpy as np
import jax.numpy as jnp
import optax
from quask.template_pennylane import pennylane_projected_quantum_kernel, hardware_efficient_ansatz, GeneticEmbedding
from pathlib import Path
import quask
from utils import *



# main function
def train_kernel(k, path, dataname):

    name = ''
    for key in k.keys():
        name = name + str(k[key]) + '_'
    name = name[:-1]

    datatype = dataname.split('_')[0]
    splitteddata = load_dataset(path + '/' + dataname + '.npy', datatype)
    n = int(dataname.split('_')[1])

    if k['type'] == 'random':
        train_random(splitteddata, path + '/kernels/', name, k['seed'])
    elif k['type'] == 'trainable':
        train_trainable(splitteddata, k['epochs'], k['metric'], path + '/kernels/', name, k['seed'])
    elif k['type'] == 'genetic':
        pass
    else:
        print('Invalid kernel type: no data will be generated for this configuration.')
        return False, ''
    return True, name



# ====================================================================
# =============== TRAIN RANDOM QUANTUM KERNELS =======================
# ====================================================================

def train_random(dataset, path, name, seed):

    np.random.seed(seed)
    jax.random.PRNGKey(seed)
    kerneldata = {}

    n, d = np.shape(dataset['train_x'])
    if not os.path.isdir(path): os.mkdir(path)
    file = path + name + '.npy'

    # load label if they exists, otherwise generate them
    if Path(file).exists():
        print('Kernel ' + name + ' already exists!')
    else:
        data = np.random.uniform(low=-1.0, high=1.0, size=(n, d))
        kerneldata['K'] = pennylane_projected_quantum_kernel(lambda x, wires: random_quantum_embedding(x, wires, seed), data)
        kerneldata['dataset'] = dataset
        kerneldata['weights'] = data
        kerneldata['K_test'] = pennylane_projected_quantum_kernel(lambda x, wires: random_quantum_embedding(x, wires, seed), data, np.array(dataset['test_x']))
        np.save(file, kerneldata)
        print('Kernel ' + name + ' has been generated.')



# ====================================================================
# ============= TRAIN TRAINABLE QUANTUM KERNELS ======================
# ====================================================================

def train_trainable(dataset, epochs, metric, path, name, seed):

    np.random.seed(seed)
    jax.random.PRNGKey(seed)
    kerneldata = {}
    kerneldata['dataset'] = dataset

    n, d = np.shape(dataset['train_x'])
    if not os.path.isdir(path): os.mkdir(path)
    file = path + name + '.npy'
    layers = d
    adam_optimizer = optax.adam(learning_rate=0.1)

    if Path(file).exists():
        print('Kernel ' + name + ' already exists!')
    else:
        params = jax.random.normal(jax.random.PRNGKey(seed), shape=(layers, 2 * d))
        kerneldata['starting_params'] = params
        opt_state = adam_optimizer.init(params)
        for epoch in range(epochs):

            if metric == 'acc':
                K = pennylane_projected_quantum_kernel(
                    lambda x, wires: trainable_embedding(x, params, layers, wires=wires), np.array(dataset['train_x']))
                K_v = pennylane_projected_quantum_kernel(
                    lambda x, wires: trainable_embedding(x, params, layers, wires=wires), np.array(dataset['valid_x']))
                cost, grad_circuit = jax.value_and_grad(lambda theta: accuracy_svc(K, K_v, dataset['train_y'], dataset['valid_y']))(params)
            elif metric == 'mse':
                K = pennylane_projected_quantum_kernel(
                    lambda x, wires: trainable_embedding(x, params, layers, wires=wires), np.array(dataset['train_x']))
                K_v = pennylane_projected_quantum_kernel(
                    lambda x, wires: trainable_embedding(x, params, layers, wires=wires), np.array(dataset['valid_x']))
                cost, grad_circuit = jax.value_and_grad(lambda theta: accuracy_svr(K, K_v, dataset['train_y'], dataset['valid_y']))(params)
            else:
                K = pennylane_projected_quantum_kernel(
                    lambda x, wires: trainable_embedding(x, params, layers, wires=wires), np.array(dataset['train_x'] + dataset['valid_x']))
                cost, grad_circuit = jax.value_and_grad(lambda theta: k_target_alignment(K, np.array(dataset['train_y'])))(params)
            updates, opt_state = adam_optimizer.update(grad_circuit, opt_state)
            params = optax.apply_updates(params, updates)
            print(".", end="", flush=True)

        kerneldata['trained_params'] = params.copy()
        kerneldata['K'] = pennylane_projected_quantum_kernel(lambda x, wires: trainable_embedding(x, kerneldata['trained_params'], layers, wires=wires), np.array(dataset['train_x'] + dataset['valid_x']))
        kerneldata['K_test'] = pennylane_projected_quantum_kernel(lambda x, wires: trainable_embedding(x, kerneldata['trained_params'], layers, wires=wires), np.array(dataset['test_x']))
        np.save(file, kerneldata)
        print('\nKernel ' + name + ' has been generated.')



# # ====================================================================
# # ========= TRAIN QUANTUM KERNELS WITH GENETIC ALGORITHMS ============
# # ====================================================================
#
# np.random.seed(32323234)
#
# print("Calculating genetic quantum kernel 1...")
# if Path(res_dir + '/genetic_qk_1.npy').exists():
#     genetic_qk_1 = np.load(res_dir + '/genetic_qk_1.npy')
# else:
#     ge1 = GeneticEmbedding(X, y, X.shape[1], d, num_generations=100, solution_per_population=10)
#     ge1.run()
#     ge1_best_solution, _, _ = ge1.ga.best_solution()
#     print(ge1_best_solution)
#     genetic_qk_1 = pennylane_projected_quantum_kernel(lambda x, wires: lambda x, wires: ge1.transform_solution_to_embedding(x, ge1_best_solution), X)
#     np.save(res_dir + '/genetic_qk_1.npy', genetic_qk_1)
#
# np.random.seed(453535345)
#
# print("Calculating genetic quantum kernel 2...")
# if Path(res_dir + '/genetic_qk_2.npy').exists():
#     genetic_qk_2 = np.load(res_dir + '/genetic_qk_2.npy')
# else:
#     ge2 = GeneticEmbedding(X, y, X.shape[1], d, num_generations=100, solution_per_population=10)
#     ge2.run()
#     ge2_best_solution, _, _ = ge2.ga.best_solution()
#     print(ge2_best_solution)
#     genetic_qk_2 = pennylane_projected_quantum_kernel(lambda x, wires: lambda x, wires: ge2.transform_solution_to_embedding(x, ge2_best_solution), X)
#     np.save(res_dir + '/genetic_qk_2.npy', genetic_qk_2)
#
# np.random.seed(21113231)
#
# print("Calculating genetic quantum kernel 3...")
# if Path(res_dir + '/genetic_qk_3.npy').exists():
#     genetic_qk_3 = np.load(res_dir + '/genetic_qk_3.npy')
# else:
#     ge3 = GeneticEmbedding(X, y, X.shape[1], d, num_generations=100, solution_per_population=10)
#     ge3.run()
#     ge3_best_solution, _, _ = ge3.ga.best_solution()
#     print(ge3_best_solution)
#     genetic_qk_3 = pennylane_projected_quantum_kernel(lambda x, wires: lambda x, wires: ge3.transform_solution_to_embedding(x, ge2_best_solution), X)
#     np.save(res_dir + '/genetic_qk_3.npy', genetic_qk_3)
#
#
# genetic_qk_1_alignment = k_target_alignment(genetic_qk_1, y)
# genetic_qk_2_alignment = k_target_alignment(genetic_qk_2, y)
# genetic_qk_3_alignment = k_target_alignment(genetic_qk_3, y)
# print("KTA GENETIC WITH BATCH 5")
# print(genetic_qk_1_alignment, genetic_qk_2_alignment, genetic_qk_3_alignment)
#
# # ====================================================================
# # ================ CLASSIFY WITH GENETIC ALGORITHMS ==================
# # ===================== FULL BATCH ===================================
#
# np.random.seed(574534)
#
# print("Calculating genetic quantum kernel 1 FULL BATCH...")
# if Path(res_dir + '/genetic_qk_1_fullbatch.npy').exists():
#     genetic_fb_qk_1 = np.load(res_dir + '/genetic_qk_1_fullbatch.npy')
# else:
#     ge1_fb = GeneticEmbedding(X, y, X.shape[1], d, num_generations=50, solution_per_population=10)
#     ge1_fb.run()
#     ge1_fb_best_solution, ge1_fb_best_solution_fitness, idx = ge1_fb.ga.best_solution()
#     the_feature_map = lambda x, wires: ge1_fb.transform_solution_to_embedding(x, ge1_fb_best_solution)
#     the_gram_matrix = pennylane_projected_quantum_kernel(the_feature_map, X)
#     np.save(res_dir + '/genetic_qk_1_fullbatch.npy', the_gram_matrix)
#
#
# #print("Calculating genetic quantum kernel 2 FULL BATCH...")
# #if Path(res_dir + '/genetic_qk_2_fullbatch.npy').exists():
# #    genetic_fb_qk_2 = np.load(res_dir + '/genetic_qk_2_fullbatch.npy')
# #else:
# #    ge2_fb = GeneticEmbedding(X, y, X.shape[1], d, num_generations=50, solution_per_population=10)
# #    ge2_fb.run()
# #    ge2_fb_best_solution, ge2_fb_best_solution_fitness, idx = ge2_fb.ga.best_solution()
# #    the_feature_map = lambda x, wires: ge2_fb.transform_solution_to_embedding(x, ge2_fb_best_solution)
# #    the_gram_matrix = pennylane_projected_quantum_kernel(the_feature_map, X)
# #    np.save(res_dir + '/genetic_qk_2_fullbatch.npy', the_gram_matrix)
# #    genetic_fb_qk_2 = the_gram_matrix
#
# genetic_fb_qk_2 = genetic_fb_qk_1
# np.random.seed(97979797)
#
# print("Calculating genetic quantum kernel 3 FULL BATCH...")
# if Path(res_dir + '/genetic_qk_3_fullbatch.npy').exists():
#     genetic_fb_qk_2 = np.load(res_dir + '/genetic_qk_3_fullbatch.npy')
# else:
#     ge3_fb = GeneticEmbedding(X, y, X.shape[1], d, num_generations=50, solution_per_population=10)
#     ge3_fb.run()
#     ge3_fb_best_solution, ge3_fb_best_solution_fitness, idx = ge3_fb.ga.best_solution()
#     the_feature_map = lambda x, wires: ge3_fb.transform_solution_to_embedding(x, ge3_fb_best_solution)
#     the_gram_matrix = pennylane_projected_quantum_kernel(the_feature_map, X)
#     np.save(res_dir + '/genetic_qk_3_fullbatch.npy', the_gram_matrix)
#     genetic_fb_qk_3 = the_gram_matrix
#
#
# #genetic_fb_qk_1_alignment = k_target_alignment(genetic_fb_qk_1, y)
# #genetic_fb_qk_2_alignment = k_target_alignment(genetic_fb_qk_2, y)
# #genetic_fb_qk_3_alignment = k_target_alignment(genetic_fb_qk_3, y)
# #print("FULL BATCH GENETIC", genetic_fb_qk_2_alignment)
