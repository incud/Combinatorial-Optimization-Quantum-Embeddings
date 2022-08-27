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



# # ====================================================================
# # ==================== LOAD DATASET ==================================
# # ====================================================================

def load_dataset_ui(res_dir):
    dataset = {}
    name = ''
    print('\nAVAILABLE DATASETS:')
    datalist = [ f.name for f in os.scandir(res_dir) if f.is_dir() ]
    for i in range(len(datalist)):
        print(' ' + str(i) +'- ' + datalist[i])
    index = input('\nChoose a dataset to load by submitting its index or press any key to stop the program: ')
    if index.isdigit() and int(index) < len(datalist):
        name = datalist[int(index)]
        if 'synthetic' in name:
            dataset = load_synthetic(res_dir + '/' + name + '/' + name +'.npy')
        # elif:
        # load other kind of dataset
    else:
        quit()

    return dataset, name



def load_dataset(file, type):
    if type == 'synt':
        return load_synthetic(file)
    elif type == 'mnist':
        return load_mnist(file)



# load synthetic dataset
def load_synthetic(file):
    dataset = {}
    filedata = np.load(file, allow_pickle=True)
    dataset['X'] = filedata.item().get('X')
    dataset['weights_shape'] = filedata.item().get('weights_shape')
    dataset['weights'] = filedata.item().get('weights')
    dataset['Y'] = filedata.item().get('Y')
    dataset['train_x'] = filedata.item().get('train_x')
    dataset['train_y'] = filedata.item().get('train_y')
    dataset['test_x'] = filedata.item().get('test_x')
    dataset['test_y'] = filedata.item().get('test_y')
    dataset['valid_x'] = filedata.item().get('valid_x')
    dataset['valid_y'] = filedata.item().get('valid_y')
    return dataset



# load mnist dataset
def load_mnist(file):
    dataset = {}
    filedata = np.load(file, allow_pickle=True)
    dataset['XC1'] = filedata.item().get('XC1')
    dataset['XC2'] = filedata.item().get('XC2')
    dataset['YC1'] = filedata.item().get('YC1')
    dataset['YC2'] = filedata.item().get('YC2')
    dataset['train_x'] = filedata.item().get('train_x')
    dataset['train_y'] = filedata.item().get('train_y')
    dataset['test_x'] = filedata.item().get('test_x')
    dataset['test_y'] = filedata.item().get('test_y')
    dataset['valid_x'] = filedata.item().get('valid_x')
    dataset['valid_y'] = filedata.item().get('valid_y')
    return dataset



