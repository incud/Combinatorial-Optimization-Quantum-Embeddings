import jax
import os
from jax.config import config
config.update("jax_enable_x64", True)
import pennylane as qml
import jax.numpy as jnp
from quask.template_pennylane import pennylane_projected_quantum_kernel, hardware_efficient_ansatz, GeneticEmbedding
import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error
from scipy.linalg import sqrtm
import numpy.linalg as la
# import optax
# from pathlib import Path
# import quask



# function to split dataset in training and test
def list_train_test_split(X, y, n_test, seed):
    np.random.seed(seed)
    jax.random.PRNGKey(seed)

    idxs_test = np.random.choice(len(X), n_test, replace=False)

    train_x = [X[i] for i in range(len(X)) if i not in idxs_test]
    train_y = [y[i] for i in range(len(X)) if i not in idxs_test]
    test_x = [X[i] for i in idxs_test]
    test_y = [y[i] for i in idxs_test]

    return train_x, test_x, train_y, test_y



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
    x = x.reshape(shape)
    assert x.shape == shape
    qml.RandomLayers(weights=x, seed=seed, wires=wires)



# create trainable embedding
def trainable_embedding(x, theta, layers, wires):
    qml.AngleEmbedding(x, rotation='Y', wires=wires)
    for i in range(layers):
        hardware_efficient_ansatz(theta=theta[i], wires=wires)



# create quantum system that generates the labels
@jax.jit
def generate_label(x, weights, verbose=False):

    N = len(x)
    device = qml.device("default.qubit.jax", wires=N)

    @qml.qnode(device, interface='jax')
    def quantum_system():
        qml.AngleEmbedding(x, rotation='X', wires=range(N))
        qml.BasicEntanglerLayers(weights=weights, wires=range(N))
        return qml.expval(qml.PauliZ(1))

    drawer = qml.draw(quantum_system)
    if verbose: print(drawer())
    return quantum_system()



# takes a numeric list (vals) and computes the histogram depending on the bins ranges (bins)
def compute_histogram(vals, bins):
    hist = np.zeros(len(bins))
    for i in range(len(vals)):
        for c in range(1,len(bins)):
            if vals[i] < bins[c]:
                hist[c-1] +=1
                break
    return hist



# retrieve the upper triangular section of a matrix excluding the diagonal
def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]



# compute the accuracy of a kernelized regression model
def accuracy_svr(gram, gram_test, y, y_test):
    svm = SVR(kernel='precomputed').fit(gram, y)
    y_predict = svm.predict(gram_test)
    return -mean_squared_error(y_test, y_predict)



# compute the accuracy of a kernelized classification model (labels {-1, 1})
def accuracy_svc(gram, gram_test, y, y_test):
    svm = SVC(kernel='precomputed').fit(gram, y)
    y_predict = svm.predict(gram_test)
    return sum((y_predict*y_test)+1)/(2*len(y_test))



# load pretrained kernel with higher training epochs (or generations) ad same hyper parameters
def find_pretrained(path, name):
    namel = (name + '.npy').split('_')
    namel.pop(1)
    best = ''
    best_eps = 0
    kernels = [f for f in os.listdir(path) if f.endswith('.' + "npy")]
    for i in range(len(kernels)):
        tmp = kernels[i].split('_')
        eps = tmp.pop(1)
        if  namel == tmp and int(eps) > best_eps:
            best = kernels[i]
            best_eps = int(eps)
    return best



# compute key for grouping in dictionaries or labeling series in plots
def compute_key(name, differentiate, type_obj):
    key = ''
    if type_obj == 'dataset':
        if differentiate == 'dataset':
            key = name.split('_')[0]
        elif differentiate == 'all':
            if name.split('_')[0] == 'synt':
                key = name.split('_')[0] + ' d=' + name.split('_')[2] + ' N=' + name.split('_')[1]

    elif type_obj == 'kernel':
        if differentiate == 'kernel':
            key = name.split('_')[0]
        elif differentiate == 'all':
            if name.split('_')[0] == 'genetic':
                key = name.split('_')[0] + ' ' + name.split('_')[5] + ' threshold (' + name.split('_')[6] + ')'
            elif name.split('_')[0] == 'trainable':
                key = name.split('_')[0] + ' (' + name.split('_')[2] + ')'
            elif name.split('_')[0] == 'random':
                key = name.split('_')[0]

    return key


# # ====================================================================
# # ==================== LOAD DATASET ==================================
# # ====================================================================

def load_dataset(file, type):
    dataset = {}
    filedata = np.load(file, allow_pickle=True)
    dataset['train_x'] = filedata.item().get('train_x')
    dataset['train_y'] = filedata.item().get('train_y')
    dataset['test_x'] = filedata.item().get('test_x')
    dataset['test_y'] = filedata.item().get('test_y')
    dataset['valid_x'] = filedata.item().get('valid_x')
    dataset['valid_y'] = filedata.item().get('valid_y')

    if type == 'synt':
        return load_synthetic(filedata, dataset)
    elif type == 'mnist':
        return load_mnist(file)
    else:
        dataset['X'] = filedata.item().get('X')
        dataset['Y'] = filedata.item().get('Y')
        return dataset



# load synthetic dataset
def load_synthetic(filedata, dataset):
    dataset['X'] = filedata.item().get('X')
    dataset['weights_shape'] = filedata.item().get('weights_shape')
    dataset['weights'] = filedata.item().get('weights')
    dataset['Y'] = filedata.item().get('Y')
    return dataset



# load mnist dataset
def load_mnist(filedata, dataset):
    dataset['XC1'] = filedata.item().get('XC1')
    dataset['XC2'] = filedata.item().get('XC2')
    dataset['YC1'] = filedata.item().get('YC1')
    dataset['YC2'] = filedata.item().get('YC2')
    return dataset


# # ====================================================================
# # ==================== LOAD DATASET ==================================
# # ====================================================================

def load_kernel(file, type):
    if type == 'random':
        return load_randomk(file)
    elif type == 'trainable':
        return load_trainablek(file)
    elif type == 'genetic':
        return load_genetick(file)



# load random kernel
def load_randomk(file):
    kernel = {}
    filedata = np.load(file, allow_pickle=True)
    kernel['K'] = filedata.item().get('K')
    kernel['K_test'] = filedata.item().get('K_test')
    kernel['weights'] = filedata.item().get('weights')
    return kernel



# load trainable kernel
def load_trainablek(file):
    kernel = {}
    filedata = np.load(file, allow_pickle=True)
    kernel['K'] = filedata.item().get('K')
    kernel['K_test'] = filedata.item().get('K_test')
    kernel['starting_params'] = filedata.item().get('starting_params')
    kernel['trained_params'] = filedata.item().get('trained_params')
    return kernel



# load genetic kernel
def load_genetick(file):
    kernel = {}
    filedata = np.load(file, allow_pickle=True)
    kernel['K'] = filedata.item().get('K')
    kernel['K_test'] = filedata.item().get('K_test')
    kernel['best_solution'] = filedata.item().get('best_solution')
    kernel['population'] = filedata.item().get('population')
    kernel['v_thr'] = filedata.item().get('v_thr')
    kernel['low_variance_list'] = filedata.item().get('low_variance_list')
    return kernel
