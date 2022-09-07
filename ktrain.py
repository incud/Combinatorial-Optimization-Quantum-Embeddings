import sys
import time
import datetime
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
import jax
import os
from jax.config import config
config.update("jax_enable_x64", True)
import pennylane as qml
import numpy as np
import jax.numpy as jnp
import optax
from quask.template_pennylane import pennylane_projected_quantum_kernel, hardware_efficient_ansatz, GeneticEmbeddingUnstructured, GeneticEmbedding
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
        train_genetic(splitteddata,
                      k['generations'],
                      k['spp'],
                      k['npm'],
                      k['metric'],
                      k['variance_threshold'],
                      k['threshold_mode'],
                      path + '/kernels/', name,
                      k['seed'],
                      k['structure'],
                      False if 'cnot' not in k.keys() else k['cnot']=='True')
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

    n, d = np.shape(dataset['train_x'] + dataset['valid_x'])
    if not os.path.isdir(path): os.mkdir(path)
    file = path + name + '.npy'

    # load label if they exists, otherwise generate them
    if Path(file).exists():
        print('Kernel ' + name + ' already exists!')
    else:
        data = np.random.uniform(low=-1.0, high=1.0, size=(n, d))
        kerneldata['K'] = pennylane_projected_quantum_kernel(lambda x, wires: random_quantum_embedding(x, wires, seed), data)
        kerneldata['weights'] = data
        kerneldata['K_test'] = pennylane_projected_quantum_kernel(lambda x, wires: random_quantum_embedding(x, wires, seed), np.array(dataset['test_x']), data)
        np.save(file, kerneldata)
        print('Kernel ' + name + ' has been generated.')



# ====================================================================
# ============= TRAIN TRAINABLE QUANTUM KERNELS ======================
# ====================================================================

def train_trainable(dataset, epochs, metric, path, name, seed):

    np.random.seed(seed)
    jax.random.PRNGKey(seed)
    kerneldata = {}

    if not os.path.isdir(path): os.mkdir(path)
    file = path + name + '.npy'
    d = np.shape(dataset['train_x'])[1]
    layers = np.shape(dataset['train_x'])[1]
    adam_optimizer = optax.adam(learning_rate=0.1)

    if Path(file).exists():
        print('Kernel ' + name + ' already exists!')
    else:
        pretrained = find_pretrained(path, name)
        if pretrained == '':
            params = jax.random.normal(jax.random.PRNGKey(seed), shape=(layers, 2 * d))
            first_epoch = 0
        else:
            kerneldata = load_kernel(path + pretrained, 'trainable')
            first_epoch = int(pretrained.split('_')[1])
            params = kerneldata['trained_params']

        kerneldata['starting_params'] = params
        opt_state = adam_optimizer.init(params)
        valid_x = np.array(dataset['train_x'] + dataset['valid_x'])
        train_x = np.array(dataset['train_x'])
        valid_y = np.array(dataset['train_y'] + dataset['valid_y']).ravel()
        train_y = np.array(dataset['train_y']).ravel()
        start = time.process_time()

        sys.stdout.write('\033[K' + 'Training started. --- Estimated time left: H:mm:ss.dddddd' + '\r')
        for epoch in range(first_epoch, epochs):

            if metric == 'mse':
                # train on partial training set with full training set as validation
                K = pennylane_projected_quantum_kernel(
                    lambda x, wires: trainable_embedding(x, params, layers, wires=wires), train_x)
                K_v = pennylane_projected_quantum_kernel(
                    lambda x, wires: trainable_embedding(x, params, layers, wires=wires),
                    valid_x, train_x)
                cost, grad_circuit = jax.value_and_grad(lambda theta: accuracy_svr(K, K_v, train_y, valid_y))(params)

            else:
                # train on full training set without validation
                K = pennylane_projected_quantum_kernel(
                    lambda x, wires: trainable_embedding(x, params, layers, wires=wires), valid_x)
                cost, grad_circuit = jax.value_and_grad(lambda theta: k_target_alignment(K, valid_y))(params)

            updates, opt_state = adam_optimizer.update(grad_circuit, opt_state)
            params = optax.apply_updates(params, updates)
            end = time.process_time()
            sys.stdout.write('\033[K' + 'Epoch: ' + str(epoch+1) + ' completed. --- Estimated time left: ' + str(datetime.timedelta(seconds=(epochs - epoch) * (end - start)/(epoch - first_epoch + 1))) + '\r')

        kerneldata['trained_params'] = params.copy()
        kerneldata['K'] = pennylane_projected_quantum_kernel(lambda x, wires: trainable_embedding(x, kerneldata['trained_params'], layers, wires=wires), valid_x)
        kerneldata['K_test'] = pennylane_projected_quantum_kernel(lambda x, wires: trainable_embedding(x, kerneldata['trained_params'], layers, wires=wires), np.array(dataset['test_x']), valid_x)
        np.save(file, kerneldata)
        sys.stdout.write('\033[K' + 'Kernel ' + name + ' has been generated.\n' + '\r')



# ====================================================================
# ========= TRAIN QUANTUM KERNELS WITH GENETIC ALGORITHMS ============
# ====================================================================

def train_genetic(dataset, gens, spp, npm, metric, v_thr, thr_mode, path, name, seed, structure, cnot):

    np.random.seed(seed)
    jax.random.PRNGKey(seed)
    kerneldata = {}

    if not os.path.isdir(path): os.mkdir(path)
    file = path + name + '.npy'

    if Path(file).exists():
        print('Kernel ' + name + ' already exists!')
    else:
        d = np.shape(dataset['train_x'])[1]
        layers = np.shape(dataset['train_x'])[1]

        pretrained = find_pretrained(path, name)
        if pretrained == '':
            init_pop = None
            old_gen = 0
            kerneldata['low_variance_list'] = []
        else:
            old_kerneldata = load_kernel(path + pretrained, 'genetic')
            init_pop = old_kerneldata['population']
            kerneldata['low_variance_list'] = old_kerneldata['low_variance_list']
            kerneldata['low_variance_list'].pop()
            assert init_pop is not None
            old_gen = int(pretrained.split('_')[1])
            v_thr = old_kerneldata['v_thr']

        valid_x = np.array(dataset['train_x'] + dataset['valid_x'])
        valid_y = np.array(dataset['train_y'] + dataset['valid_y']).ravel()
        if structure == 'unstructured':
            geclass =  GeneticEmbeddingUnstructured
        else:
            geclass = GeneticEmbedding

        if metric == 'mse':

            ge = geclass(np.array(dataset['train_x']), np.array(dataset['train_y']).ravel(), d, layers, v_thr,
                                  num_parents_mating=int(spp * npm),
                                  num_generations=gens - old_gen,
                                  solution_per_population=spp,
                                  initial_population=init_pop,
                                  fitness_mode='mse',
                                  validation_X=np.array(dataset['valid_x']),
                                  validation_y=np.array(dataset['valid_y']).ravel(),
                                  threshold_mode=thr_mode,
                                  verbose='minimal',
                                  cnot=cnot)
        elif metric == 'kta':
            ge = geclass(valid_x, valid_y, d, layers, v_thr,
                                              num_parents_mating=int(spp * npm),
                                              num_generations=gens - old_gen,
                                              solution_per_population=spp,
                                              initial_population=init_pop,
                                              fitness_mode='kta',
                                              threshold_mode=thr_mode,
                                              verbose='minimal',
                                              cnot=cnot)

        ge.run()
        kerneldata['best_solution'], ge_best_solution_fitness, idx = ge.ga.best_solution()
        kerneldata['population'] = ge.ga.population
        kerneldata['low_variance_list'] = kerneldata['low_variance_list'] + ge.low_variance_list
        kerneldata['v_thr'] = ge.kernel_concentration_threshold
        feature_map = lambda x, wires: ge.transform_solution_to_embedding(x, kerneldata['best_solution'])
        kerneldata['K'] = pennylane_projected_quantum_kernel(feature_map, valid_x)
        kerneldata['K_test'] = pennylane_projected_quantum_kernel(feature_map, np.array(dataset['test_x']), valid_x)
        np.save(file, kerneldata)
        sys.stdout.write('\033[K' + 'Kernel ' + name + ' has been generated.\n' + '\r')

