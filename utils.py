import jax
from jax.config import config
config.update("jax_enable_x64", True)
import pennylane as qml
import jax.numpy as jnp
from quask.template_pennylane import pennylane_projected_quantum_kernel, hardware_efficient_ansatz, GeneticEmbedding
import numpy as np
# import optax
# from pathlib import Path
# import quask



# compute kernel-target alignment
def k_target_alignment(K, y):
    B = jnp.outer(y, y)
    norm = jnp.sqrt(jnp.sum(K * K) * jnp.sum(B * B))
    return jnp.sum(K * B) / norm



# compute task-model alignment
def task_model_alignment(K, y):
    raise NotImplementedError("TMA Not implemented!")



# create random kernel
def random_quantum_embedding(x, wires, seed):
    N = len(x)
    shape = qml.RandomLayers.shape(n_layers=1, n_rotations=1 * N)
    assert x.shape == shape
    qml.RandomLayers(weights=x, seed=seed, wires=wires)



# create trainable embedding
def trainable_embedding(x, theta, layers, wires):
    qml.AngleEmbedding(x, rotation='Y', wires=wires)
    for i in range(layers):
        hardware_efficient_ansatz(theta=theta[i], wires=wires)



# create quantum system that generates the labels
@jax.jit
def generate_label(x, weights):

    N = len(x)
    device = qml.device("default.qubit.jax", wires=N)

    @qml.qnode(device, interface='jax')
    def quantum_system():
        qml.AngleEmbedding(x, rotation='X', wires=range(N))
        qml.BasicEntanglerLayers(weights=weights, wires=range(N))
        return qml.expval(qml.PauliZ(1))

    drawer = qml.draw(quantum_system)
    print(drawer())
    return quantum_system()



def load_synthetic(file):
    dataset = {}
    filedata = np.load(file, allow_pickle=True)
    dataset['X'] = filedata.item().get('X')
    dataset['weights_shape'] = filedata.item().get('weights_shape')
    dataset['weights'] = filedata.item().get('weights')
    dataset['Y'] = filedata.item().get('Y')
    return dataset