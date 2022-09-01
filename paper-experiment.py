import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config
import pennylane as qml
import optax
from quask.template_pennylane import pennylane_projected_quantum_kernel, hardware_efficient_ansatz, GeneticEmbedding
from pathlib import Path
import datetime

# reproducibility results
np.random.seed(123456)


# ====================================================================
# ==================== GENERATE SYNTHETIC DATASET ====================
# ====================================================================


d = 10  # number of features
n = 30  # elements of the dataset

# load data if exists
if Path('old results/genetic-algorithm-results/X.npy').exists():
    X = np.load('old results/genetic-algorithm-results/X.npy')
else:
    X = np.random.uniform(low=-1.0, high=1.0, size=(n, d))
    np.save('old results/genetic-algorithm-results/X.npy', X)

# load model if exists
weights_shape = qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=d)

if Path('old results/genetic-algorithm-results/weights.npy').exists():
    weights = np.load('old results/genetic-algorithm-results/weights.npy')
else:
    weights = np.random.uniform(-np.pi, np.pi, size=weights_shape)
    np.save('old results/genetic-algorithm-results/weights.npy', weights)

# create quantum system that generates the labels
@jax.jit
def generate_label(x):

    N = len(x)
    device = qml.device("default.qubit.jax", wires=N)

    @qml.qnode(device, interface='jax')
    def quantum_system():
        qml.AngleEmbedding(x, rotation='X', wires=range(N))
        qml.BasicEntanglerLayers(weights=weights, wires=range(N))
        return qml.expval(qml.PauliZ(1))

    return quantum_system()


# load label if they exists, otherwise generate them
if Path('old results/genetic-algorithm-results/y.npy').exists():
    y = np.load('old results/genetic-algorithm-results/y.npy')
else:
    y = np.array([generate_label(x) for x in X])
    np.save('old results/genetic-algorithm-results/y.npy', y)


# ====================================================================
# =============== CLASSIFY WITH RANDOM QUANTUM KERNELS ===============
# ====================================================================

def random_quantum_embedding(x, wires, seed):
    N = len(x)
    shape = qml.RandomLayers.shape(n_layers=1, n_rotations=1 * N)
    assert x.shape == shape
    qml.RandomLayers(weights=x, seed=seed, wires=wires)


print("Calculating random quantum kernel 1...")
# load label if they exists, otherwise generate them
if Path('old results/genetic-algorithm-results/random_qk_1.npy').exists():
    random_qk_1 = np.load('old results/genetic-algorithm-results/random_qk_1.npy')
else:
    random_qk_1 = pennylane_projected_quantum_kernel(lambda x, wires: random_quantum_embedding(x, wires, 4343), X)
    np.save('old results/genetic-algorithm-results/random_qk_1.npy', random_qk_1)

print("Calculating random quantum kernel 2...")
if Path('old results/genetic-algorithm-results/random_qk_2.npy').exists():
    random_qk_2 = np.load('old results/genetic-algorithm-results/random_qk_2.npy')
else:
    random_qk_2 = pennylane_projected_quantum_kernel(lambda x, wires: random_quantum_embedding(x, wires, 4344), X)
    np.save('old results/genetic-algorithm-results/random_qk_2.npy', random_qk_2)


print("Calculating random quantum kernel 3...")
if Path('old results/genetic-algorithm-results/random_qk_3.npy').exists():
    random_qk_3 = np.load('old results/genetic-algorithm-results/random_qk_3.npy')
else:
    random_qk_3 = pennylane_projected_quantum_kernel(lambda x, wires: random_quantum_embedding(x, wires, 4345), X)
    np.save('old results/genetic-algorithm-results/random_qk_3.npy', random_qk_3)


random_qk_1_alignment = k_target_alignment(random_qk_1, y)
random_qk_2_alignment = k_target_alignment(random_qk_2, y)
random_qk_3_alignment = k_target_alignment(random_qk_3, y)
print("Random quantum kernel alignment:")
print(random_qk_1_alignment, random_qk_2_alignment, random_qk_3_alignment)


# ====================================================================
# ================ CLASSIFY WITH GENETIC ALGORITHMS ==================
# ====================================================================

np.random.seed(32323234)

print("Calculating genetic quantum kernel 1...")
if Path('old results/genetic-algorithm-results/genetic_qk_1.npy').exists():
    genetic_qk_1 = np.load('old results/genetic-algorithm-results/genetic_qk_1.npy')
else:
    ge1 = GeneticEmbedding(X, y, X.shape[1], d, num_generations=100, solution_per_population=10)
    ge1.run()
    ge1_best_solution, _, _ = ge1.ga.best_solution()
    print(ge1_best_solution)
    genetic_qk_1 = pennylane_projected_quantum_kernel(lambda x, wires: lambda x, wires: ge1.transform_solution_to_embedding(x, ge1_best_solution), X)
    np.save('old results/genetic-algorithm-results/genetic_qk_1.npy', genetic_qk_1)

np.random.seed(453535345)

print("Calculating genetic quantum kernel 2...")
if Path('old results/genetic-algorithm-results/genetic_qk_2.npy').exists():
    genetic_qk_2 = np.load('old results/genetic-algorithm-results/genetic_qk_2.npy')
else:
    ge2 = GeneticEmbedding(X, y, X.shape[1], d, num_generations=100, solution_per_population=10)
    ge2.run()
    ge2_best_solution, _, _ = ge2.ga.best_solution()
    print(ge2_best_solution)
    genetic_qk_2 = pennylane_projected_quantum_kernel(lambda x, wires: lambda x, wires: ge2.transform_solution_to_embedding(x, ge2_best_solution), X)
    np.save('old results/genetic-algorithm-results/genetic_qk_2.npy', genetic_qk_2)

np.random.seed(21113231)

print("Calculating genetic quantum kernel 3...")
if Path('old results/genetic-algorithm-results/genetic_qk_3.npy').exists():
    genetic_qk_3 = np.load('old results/genetic-algorithm-results/genetic_qk_3.npy')
else:
    ge3 = GeneticEmbedding(X, y, X.shape[1], d, num_generations=100, solution_per_population=10)
    ge3.run()
    ge3_best_solution, _, _ = ge3.ga.best_solution()
    print(ge3_best_solution)
    genetic_qk_3 = pennylane_projected_quantum_kernel(lambda x, wires: lambda x, wires: ge3.transform_solution_to_embedding(x, ge2_best_solution), X)
    np.save('old results/genetic-algorithm-results/genetic_qk_3.npy', genetic_qk_3)


genetic_qk_1_alignment = k_target_alignment(genetic_qk_1, y)
genetic_qk_2_alignment = k_target_alignment(genetic_qk_2, y)
genetic_qk_3_alignment = k_target_alignment(genetic_qk_3, y)
print("KTA GENETIC WITH BATCH 5")
print(genetic_qk_1_alignment, genetic_qk_2_alignment, genetic_qk_3_alignment)

# ====================================================================
# ============= CLASSIFY WITH TRAINABLE QUANTUM KERNELS ==============
# ====================================================================


# create trainable embedding
def trainable_embedding(x, theta, wires):
    qml.AngleEmbedding(x, rotation='Y', wires=wires)
    for i in range(layers):
        hardware_efficient_ansatz(theta=theta[i], wires=wires)


layers = d
adam_optimizer = optax.adam(learning_rate=0.1)
epochs = 100

if Path('old results/genetic-algorithm-results/trainable_params_init.npy').exists():
    params_tentatives = np.load('old results/genetic-algorithm-results/trainable_params_init.npy')
else:
    params_1 = jax.random.normal(jax.random.PRNGKey(12345), shape=(layers, 2 * d))
    params_2 = jax.random.normal(jax.random.PRNGKey(12346), shape=(layers, 2 * d))
    params_3 = jax.random.normal(jax.random.PRNGKey(12347), shape=(layers, 2 * d))
    params_tentatives = np.array([params_1, params_2, params_3])
    np.save('old results/genetic-algorithm-results/trainable_params_init.npy', params_tentatives)


if Path('old results/genetic-algorithm-results/trainable_params_end.npy').exists():
    params_trained = np.load('old results/genetic-algorithm-results/trainable_params_end.npy')
else:
    params_trained = []

    for params in params_tentatives:
        print("Starting with params ", params)

        opt_state = adam_optimizer.init(params)
        for epoch in range(epochs):
            BATCH_SIZE = 5
            index = np.random.choice(X.shape[0], BATCH_SIZE, replace=False)
            X_batch = X[index]
            y_batch = y[index]

            K = pennylane_projected_quantum_kernel(lambda x, wires: trainable_embedding(x, params, wires=wires),
                                                   X_batch)
            cost, grad_circuit = jax.value_and_grad(lambda theta: k_target_alignment(K, y_batch))(params)
            updates, opt_state = adam_optimizer.update(grad_circuit, opt_state)
            params = optax.apply_updates(params, updates)
            print(".", end="", flush=True)

        params_trained.append(params.copy())
    np.save('old results/genetic-algorithm-results/trainable_params_end.npy', params_trained)


print("Calculating genetic quantum kernel 1...")
if Path('old results/genetic-algorithm-results/trainable_qk_1.npy').exists():
    trainable_qk_1 = np.load('old results/genetic-algorithm-results/trainable_qk_1.npy')
else:
    trainable_qk_1 = pennylane_projected_quantum_kernel(lambda x, wires: trainable_embedding(x, params_trained[0], wires=wires), X)
    np.save('old results/genetic-algorithm-results/trainable_qk_1.npy', trainable_qk_1)

print("Calculating genetic quantum kernel 2...")
if Path('old results/genetic-algorithm-results/trainable_qk_2.npy').exists():
    trainable_qk_2 = np.load('old results/genetic-algorithm-results/trainable_qk_2.npy')
else:
    trainable_qk_2 = pennylane_projected_quantum_kernel(lambda x, wires: trainable_embedding(x, params_trained[1], wires=wires), X)
    np.save('old results/genetic-algorithm-results/trainable_qk_2.npy', trainable_qk_2)

print("Calculating genetic quantum kernel 3...")
if Path('old results/genetic-algorithm-results/trainable_qk_3.npy').exists():
    trainable_qk_3 = np.load('old results/genetic-algorithm-results/trainable_qk_3.npy')
else:
    trainable_qk_3 = pennylane_projected_quantum_kernel(lambda x, wires: trainable_embedding(x, params_trained[2], wires=wires), X)
    np.save('old results/genetic-algorithm-results/trainable_qk_3.npy', trainable_qk_3)

trainable_qk_1_alignment = k_target_alignment(trainable_qk_1, y)
trainable_qk_2_alignment = k_target_alignment(trainable_qk_2, y)
trainable_qk_3_alignment = k_target_alignment(trainable_qk_3, y)
print(trainable_qk_1_alignment, trainable_qk_2_alignment, trainable_qk_3_alignment)


# ====================================================================
# ============= CLASSIFY WITH TRAINABLE QUANTUM KERNELS ==============
# ===================== FULL BATCH ===================================


# create trainable embedding
def trainable_embedding(x, theta, wires):
    qml.AngleEmbedding(x, rotation='Y', wires=wires)
    for i in range(layers):
        hardware_efficient_ansatz(theta=theta[i], wires=wires)


layers = d
adam_optimizer = optax.adam(learning_rate=0.1)
epochs = 100

print("TRAINING PARAMETERS FULLBATCH")
if Path('old results/genetic-algorithm-results/trainable_params_init_fullbatch.npy').exists():
    params_tentatives = np.load('old results/genetic-algorithm-results/trainable_params_init_fullbatch.npy')
else:
    params_1 = jax.random.normal(jax.random.PRNGKey(12345), shape=(layers, 2 * d))
    params_2 = jax.random.normal(jax.random.PRNGKey(12346), shape=(layers, 2 * d))
    params_3 = jax.random.normal(jax.random.PRNGKey(12347), shape=(layers, 2 * d))
    params_tentatives = np.array([params_1, params_2, params_3])
    np.save('old results/genetic-algorithm-results/trainable_params_init_fullbatch.npy', params_tentatives)


if Path('old results/genetic-algorithm-results/trainable_params_end_fullbatch.npy').exists():
    params_trained = np.load('old results/genetic-algorithm-results/trainable_params_end_fullbatch.npy')
else:
    params_trained = []

    for params in params_tentatives:
        print("Starting with params ", params)

        opt_state = adam_optimizer.init(params)
        for epoch in range(epochs):
            # BATCH_SIZE = 5
            # index = np.random.choice(X.shape[0], BATCH_SIZE, replace=False)
            X_batch = X  # [index]
            y_batch = y  # [index]

            K = pennylane_projected_quantum_kernel(lambda x, wires: trainable_embedding(x, params, wires=wires), X_batch)
            cost, grad_circuit = jax.value_and_grad(lambda theta: k_target_alignment(K, y_batch))(params)
            updates, opt_state = adam_optimizer.update(grad_circuit, opt_state)
            params = optax.apply_updates(params, updates)
            print(".", end="", flush=True)

        params_trained.append(params.copy())
    np.save('old results/genetic-algorithm-results/trainable_params_end_fullbatch.npy', params_trained)


print("Calculating TRAINABLE FULL BATCH quantum kernel 1...")
if Path('old results/genetic-algorithm-results/trainable_qk_fb_1.npy').exists():
    trainable_qk_1 = np.load('old results/genetic-algorithm-results/trainable_qk_fb_1.npy')
else:
    trainable_qk_1 = pennylane_projected_quantum_kernel(lambda x, wires: trainable_embedding(x, params_trained[0], wires=wires), X)
    np.save('old results/genetic-algorithm-results/trainable_qk_fb_1.npy', trainable_qk_1)

print("Calculating TRAINABLE FULL BATCH quantum kernel 2...")
if Path('old results/genetic-algorithm-results/trainable_qk_fb_2.npy').exists():
    trainable_qk_2 = np.load('old results/genetic-algorithm-results/trainable_qk_fb_2.npy')
else:
    trainable_qk_2 = pennylane_projected_quantum_kernel(lambda x, wires: trainable_embedding(x, params_trained[1], wires=wires), X)
    np.save('old results/genetic-algorithm-results/trainable_qk_fb_2.npy', trainable_qk_2)

print("Calculating TRAINABLE FULL BATCH quantum kernel 3...")
if Path('old results/genetic-algorithm-results/trainable_qk_fb_3.npy').exists():
    trainable_qk_3 = np.load('old results/genetic-algorithm-results/trainable_qk_fb_3.npy')
else:
    trainable_qk_3 = pennylane_projected_quantum_kernel(lambda x, wires: trainable_embedding(x, params_trained[2], wires=wires), X)
    np.save('old results/genetic-algorithm-results/trainable_qk_fb_3.npy', trainable_qk_3)

trainable_qk_1_alignment = k_target_alignment(trainable_qk_1, y)
trainable_qk_2_alignment = k_target_alignment(trainable_qk_2, y)
trainable_qk_3_alignment = k_target_alignment(trainable_qk_3, y)
print("Trainable full batch:")
print(trainable_qk_1_alignment, trainable_qk_2_alignment, trainable_qk_3_alignment)




# ====================================================================
# ================ CLASSIFY WITH GENETIC ALGORITHMS ==================
# ===================== FULL BATCH ===================================

np.random.seed(574534)

print("Calculating genetic quantum kernel 1 FULL BATCH...")
if Path('old results/genetic-algorithm-results/genetic_qk_1_fullbatch.npy').exists():
    genetic_fb_qk_1 = np.load('old results/genetic-algorithm-results/genetic_qk_1_fullbatch.npy')
else:
    ge1_fb = GeneticEmbedding(X, y, X.shape[1], d, num_generations=50, solution_per_population=10)
    ge1_fb.run()
    ge1_fb_best_solution, ge1_fb_best_solution_fitness, idx = ge1_fb.ga.best_solution()
    the_feature_map = lambda x, wires: ge1_fb.transform_solution_to_embedding(x, ge1_fb_best_solution)
    the_gram_matrix = pennylane_projected_quantum_kernel(the_feature_map, X)
    np.save('old results/genetic-algorithm-results/genetic_qk_1_fullbatch.npy', the_gram_matrix)


#print("Calculating genetic quantum kernel 2 FULL BATCH...")
#if Path('genetic-algorithm-results/genetic_qk_2_fullbatch.npy').exists():
#    genetic_fb_qk_2 = np.load('genetic-algorithm-results/genetic_qk_2_fullbatch.npy')
#else:
#    ge2_fb = GeneticEmbedding(X, y, X.shape[1], d, num_generations=50, solution_per_population=10)
#    ge2_fb.run()
#    ge2_fb_best_solution, ge2_fb_best_solution_fitness, idx = ge2_fb.ga.best_solution()
#    the_feature_map = lambda x, wires: ge2_fb.transform_solution_to_embedding(x, ge2_fb_best_solution)
#    the_gram_matrix = pennylane_projected_quantum_kernel(the_feature_map, X)
#    np.save('genetic-algorithm-results/genetic_qk_2_fullbatch.npy', the_gram_matrix)
#    genetic_fb_qk_2 = the_gram_matrix

genetic_fb_qk_2 = genetic_fb_qk_1
np.random.seed(97979797)

print("Calculating genetic quantum kernel 3 FULL BATCH...")
if Path('genetic-algorithm-results/genetic_qk_3_fullbatch.npy').exists():
    genetic_fb_qk_3 = np.load('genetic-algorithm-results/genetic_qk_3_fullbatch.npy')
else:
    ge3_fb = GeneticEmbedding(X, y, X.shape[1], d, num_generations=50, solution_per_population=10)
    ge3_fb.run()
    ge3_fb_best_solution, ge3_fb_best_solution_fitness, idx = ge3_fb.ga.best_solution()
    the_feature_map = lambda x, wires: ge3_fb.transform_solution_to_embedding(x, ge3_fb_best_solution)
    the_gram_matrix = pennylane_projected_quantum_kernel(the_feature_map, X)
    np.save('old results/genetic-algorithm-results/genetic_qk_3_fullbatch.npy', the_gram_matrix)
    genetic_fb_qk_3 = the_gram_matrix


#genetic_fb_qk_1_alignment = k_target_alignment(genetic_fb_qk_1, y)
#genetic_fb_qk_2_alignment = k_target_alignment(genetic_fb_qk_2, y)
#genetic_fb_qk_3_alignment = k_target_alignment(genetic_fb_qk_3, y)
#print("FULL BATCH GENETIC", genetic_fb_qk_2_alignment)


# ====================================================================
# ================================ PLOTS =============================
# ====================================================================

# import matplotlib.pyplot as plt
# from datetime import datetime
#
# NN = len(ge1.ga.best_solutions_fitness)
# plt.plot(range(NN), ge1.ga.best_solutions_fitness, label="1st run GA (stochastic fitness)")
# plt.plot(range(NN), ge2.ga.best_solutions_fitness, label="2st run GA (stochastic fitness)")
# plt.plot(range(NN), ge3.ga.best_solutions_fitness, label="3st run GA (stochastic fitness)")
# plt.scatter([NN] * 3, [genetic_qk_1_alignment, genetic_qk_2_alignment, genetic_qk_3_alignment], label="GA KTA", s=50, c='red')
# plt.scatter([NN] * 3, [trainable_qk_1_alignment, trainable_qk_2_alignment, trainable_qk_3_alignment], label="Trainable kernel KTA", s=30, c='blue')
# plt.scatter([NN] * 3, [random_qk_1_alignment, random_qk_2_alignment, random_qk_3_alignment], label="Random kernel KTA", s=10, c='green')
# plt.legend()
# plt.ylabel('Kernel-Target alignment')
# plt.xlabel('Iteration of GA algorithm')
# plt.savefig(f'comparison_fullbatch_{datetime.now().strftime("%y%m%d_%H%M%S_%f")}.png')
# plt.clf()

import matplotlib.pyplot as plt
from datetime import datetime

ga1_fb_fitnesses = np.load('old results/genetic-algorithm-results/ga_fitness_fullbatch_1.npy')
ga1_fb_fitnesses_per_iter = [ga1_fb_fitnesses[i*10:i*10+10] for i in range(51)]
ga2_fb_fitnesses = np.load('old results/genetic-algorithm-results/ga_fitness_fullbatch_2.npy')
ga2_fb_fitnesses_per_iter = [ga2_fb_fitnesses[i*10:i*10+10] for i in range(51)]
ga3_fb_fitnesses = np.load('old results/genetic-algorithm-results/ga_fitness_fullbatch_3.npy')
ga3_fb_fitnesses_per_iter = [ga3_fb_fitnesses[i*10:i*10+10] for i in range(51)]
genetic_fb_qk_1_alignment = k_target_alignment(genetic_fb_qk_1, y)
genetic_fb_qk_2_alignment = k_target_alignment(genetic_fb_qk_2, y)
genetic_fb_qk_3_alignment = k_target_alignment(genetic_fb_qk_3, y)
the_ga_x = len(np.max(ga1_fb_fitnesses_per_iter, axis=1))
plt.plot(range(the_ga_x), np.max(ga1_fb_fitnesses_per_iter, axis=1), label="1st run GA (stochastic fitness)", c='#ff0011')
plt.plot(range(the_ga_x), np.max(ga2_fb_fitnesses_per_iter, axis=1), label="2st run GA (stochastic fitness)", c='#ff0022')
plt.plot(range(the_ga_x), np.max(ga3_fb_fitnesses_per_iter, axis=1), label="3st run GA (stochastic fitness)", c='#ff0033')
plt.scatter([the_ga_x] * 3, [genetic_fb_qk_1_alignment, genetic_fb_qk_2_alignment, genetic_fb_qk_3_alignment], label="GA KTA", s=50, c='red')
plt.scatter([the_ga_x] * 3, [trainable_qk_1_alignment, trainable_qk_2_alignment, trainable_qk_3_alignment], label="Trainable kernel KTA", s=50, c='blue')
plt.scatter([the_ga_x] * 3, [random_qk_1_alignment, random_qk_2_alignment, random_qk_3_alignment], label="Random kernel KTA", s=50, c='green')
plt.legend()
plt.ylabel('Kernel-Target alignment')
plt.xlabel('Iteration of GA algorithm')
plt.savefig(f'comparison_fullbatch_{datetime.now().strftime("%y%m%d_%H%M%S_%f")}.png')
plt.clf()