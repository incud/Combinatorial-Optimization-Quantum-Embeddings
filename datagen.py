import os
from jax.config import config
from quask.datasets import download_dataset_openml
config.update("jax_enable_x64", True)
import numpy as np
from pathlib import Path
from utils import *
from sklearn import decomposition
from sklearn.preprocessing import Normalizer



# main function
def generate_data(dts, res_dir):

    name = ''
    for key in dts.keys():
        name = name + str(dts[key]) + '_'
    name = name[:-1]

    if dts['type'] == 'synt':
        generate_synthetic_data(res_dir, name, dts['features'], dts['samples'], dts['seed'], dts['testset'], dts['validationset'])
    elif dts['type'] == 'mnist':
        generate_mnist_data(res_dir, name, dts['features'], dts['samples'], dts['seed'], dts['testset'], dts['validationset'])
    else:
        print('Invalid dataset type: no data will be generated for this configuration.')
        return False, ''
    return True, name



# # ====================================================================
# # ==================== GENERATE MNIST DATASET ========================
# # ====================================================================

def generate_mnist_data(res_dir, name, d, n, seed, test, valid):

    np.random.seed(seed)
    jax.random.PRNGKey(seed)
    dataset = {}

    if not os.path.isdir('mnist_2C'): os.mkdir('mnist_2C')

    if Path('mnist_2C/mnist_2C_X_' + str(d) + '.npy').exists() and Path('mnist_2C/mnist_2C_Y.npy').exists():
        x = np.load('mnist_2C/mnist_2C_X_' + str(d) + '.npy')
        valid_y = np.load('mnist_2C/mnist_2C_Y.npy')
    else:
        tmp_x, tmp_y = download_dataset_openml(40996)
        idxs = [i for i, elem in enumerate(tmp_y) if elem == 0 or elem == 1 ]
        valid_y = tmp_y[idxs]
        valid_y = (valid_y*2)-1
        valid_x = tmp_x[idxs]

        pca = decomposition.PCA()
        pca.n_components = d
        tmp_x = pca.fit_transform(valid_x)
        x = Normalizer().fit_transform(tmp_x)
        np.save('mnist_2C/mnist_2C_X_' + str(d) + '.npy', x)
        np.save('mnist_2C/mnist_2C_Y.npy', valid_y)

    if not os.path.isdir(res_dir + '/' + name): os.mkdir(res_dir + '/' + name)
    file = res_dir + '/' + name + '/' + name +'.npy'

    assert len(x)==len(valid_y)
    if Path(file).exists():
        print('Dataset ' + name + ' already exists!')
    else:
        nc1 = int(n/2)
        nc2 = n - nc1

        xc1 = [x[i] for i, elem in enumerate(valid_y) if elem == -1]
        xc2 = [x[i] for i, elem in enumerate(valid_y) if elem == 1]
        yc1 = [valid_y[i] for i, elem in enumerate(valid_y) if elem == -1]
        yc2 = [valid_y[i] for i, elem in enumerate(valid_y) if elem == 1]

        idxs_c1 = np.random.choice(len(xc1), nc1, replace=False)
        idxs_c2 = np.random.choice(len(xc2), nc2, replace=False)

        dataset['XC1'] = [xc1[i] for i in idxs_c1]
        dataset['XC2'] = [xc2[i] for i in idxs_c2]
        dataset['YC1'] = [yc1[i] for i in idxs_c1]
        dataset['YC2'] = [yc2[i] for i in idxs_c2]

        train_n = n - int(n * test)
        valid_n = int(train_n * valid)
        train_nc1 = nc1 - int(nc1 * test)
        valid_nc1 = int(train_nc1 * valid)
        train_nc2 = train_n - train_nc1
        valid_nc2 = valid_n - valid_nc1

        idxs_trainc1 = np.random.choice(len(dataset['XC1']), train_nc1, replace=False)
        idxs_trainc2 = np.random.choice(len(dataset['XC2']), train_nc2, replace=False)

        dataset['test_x'] = [dataset['XC1'][i] for i in range(len(dataset['XC1'])) if i not in idxs_trainc1] + [
            dataset['XC2'][i] for i in range(len(dataset['XC2'])) if i not in idxs_trainc2]
        dataset['test_y'] = [dataset['YC1'][i] for i in range(len(dataset['YC1'])) if i not in idxs_trainc1] + [
            dataset['YC2'][i] for i in range(len(dataset['YC2'])) if i not in idxs_trainc2]

        idxs_validc1 = np.random.choice(idxs_trainc1, valid_nc1, replace=False)
        idxs_validc2 = np.random.choice(idxs_trainc2, valid_nc2, replace=False)

        dataset['valid_x'] = [dataset['XC1'][i] for i in idxs_validc1] + [dataset['XC2'][i] for i in idxs_validc2]
        dataset['valid_y'] = [dataset['YC1'][i] for i in idxs_validc1] + [dataset['YC2'][i] for i in idxs_validc2]
        dataset['train_x'] = [dataset['XC1'][i] for i in range(len(dataset['XC1'])) if i in idxs_trainc1 and i not in idxs_validc1] + [
            dataset['XC2'][i] for i in range(len(dataset['XC2'])) if i in idxs_trainc2 and i not in idxs_validc2]
        dataset['train_y'] = [dataset['YC1'][i] for i in range(len(dataset['YC1'])) if i in idxs_trainc1 and i not in idxs_validc1] + [
            dataset['YC2'][i] for i in range(len(dataset['YC2'])) if i in idxs_trainc2 and i not in idxs_validc2]

        ptrain = np.random.permutation(len(dataset['train_x']))
        dataset['train_x'] = [dataset['train_x'][i] for i in ptrain]
        dataset['train_y'] =[dataset['train_y'][i] for i in ptrain]

        np.save(file, dataset)
        print('Dataset ' + name + ' has been generated.')



# # ====================================================================
# # ==================== GENERATE SYNTHETIC DATASET ====================
# # ====================================================================

def generate_synthetic_data(res_dir, name, d, n, seed, test, valid):

    np.random.seed(seed)
    jax.random.PRNGKey(seed)
    dataset = {}

    if not os.path.isdir(res_dir + '/' + name): os.mkdir(res_dir + '/' + name)
    file = res_dir + '/' + name + '/' + name +'.npy'

    if Path(file).exists():
        print('Dataset ' + name + ' already exists!')
    else:
        train_n = n - int(n * test)
        valid_n = int(train_n * valid)

        dataset['X'] = np.random.uniform(low=-1.0, high=1.0, size=(n, d))
        dataset['weights_shape'] = qml.BasicEntanglerLayers.shape(n_layers=1, n_wires=d)
        dataset['weights'] = np.random.uniform(-np.pi, np.pi, size=dataset['weights_shape'])
        dataset['Y'] = np.array([generate_label(x, dataset['weights']) for x in dataset['X']])

        idxs_train = np.random.choice(len(dataset['X']), train_n, replace=False)

        dataset['test_x'] = [dataset['X'][i] for i in range(len(dataset['X'])) if i not in idxs_train]
        dataset['test_y'] = [dataset['Y'][i] for i in range(len(dataset['X'])) if i not in idxs_train]

        idxs_valid = np.random.choice(idxs_train, valid_n, replace=False)

        dataset['train_x'] = [dataset['X'][i] for i in range(len(dataset['X'])) if i in idxs_train and i not in idxs_valid]
        dataset['train_y'] = [dataset['Y'][i] for i in range(len(dataset['X'])) if i in idxs_train and i not in idxs_valid]
        dataset['valid_x'] = [dataset['X'][i] for i in idxs_valid]
        dataset['valid_y'] = [dataset['Y'][i] for i in idxs_valid]

        np.save(file, dataset)
        print('Dataset ' + name + ' has been generated.')

    return [dataset, name]

