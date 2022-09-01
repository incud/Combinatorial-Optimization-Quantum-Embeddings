import jax.numpy as jnp
import jax
import pennylane as qml





def k_target_alignment(K, y):
    B = jnp.outer(y, y)
    norm = jnp.sqrt(jnp.sum(K * K) * jnp.sum(B * B))
    return jnp.sum(K * B) / norm


def generate_weights(d):
    weights_shape = qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=d)
    return weights_shape


@jax.jit
def generate_label(x, weights):
    N = len(x)
    device = qml.device("default.qubit.jax", wires=N)
    @qml.qnode(device, interface='jax')
    def quantum_system():
        qml.AngleEmbedding(x, rotation='X', wires=range(N))
        qml.BasicEntanglerLayers(weights=weights, wires=range(N))
        return qml.expval(qml.PauliZ(1))

    return quantum_system()

def random_quantum_embedding(x, wires, seed):
    N = len(x)
    shape = qml.RandomLayers.shape(n_layers=1, n_rotations=1 * N)
    assert x.shape == shape
    qml.RandomLayers(weights=x, seed=seed, wires=wires)
