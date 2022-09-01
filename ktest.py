import sys
import json
from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from utils import *


# scatter plot of accuracy and variance
def plot_scatter_accuracy_variance(kernels, dataset, y_train, y_test, type = 'mse', differentiate = 'kernel'):

    res_dict = {}

    if differentiate == 'kernel':
        for k in kernels.keys():
            key = k.split('_')[0]
            if key in res_dict.keys():
                res_dict[key]['kernels'].append(kernels[k])
            else:
                res_dict[key] = {}
                res_dict[key]['kernels'] = [kernels[k]]

    elif differentiate == 'all':
        for k in kernels.keys():
            if k.split('_')[0] == 'genetic':
                key = k.split('_')[0] + ' ' + k.split('_')[5] + ' threshold (' + k.split('_')[6] + ')'
            elif k.split('_')[0] == 'trainable':
                key = k.split('_')[0] + ' (' + k.split('_')[2] + ')'
            elif k.split('_')[0] == 'random':
                key = k.split('_')[0]

            if key in res_dict.keys():
                res_dict[key]['kernels'].append(kernels[k])
            else:
                res_dict[key] = {}
                res_dict[key]['kernels'] = [kernels[k]]

    for k in res_dict.keys():
        res_dict[k]['accuracy'] = []
        res_dict[k]['variance'] = []

        for kernel in res_dict[k]['kernels']:
            res_dict[k]['variance'].append(np.var(upper_tri_indexing(kernel['K'])))

        if type == 'mse':
            acc_name = 'Negative Mean Squared Error'
            for kernel in res_dict[k]['kernels']:
                res_dict[k]['accuracy'].append(accuracy_svr(kernel['K'], kernel['K_test'], y_train, y_test))
        else:
            pass
            # acc_name = 'Kernel-Target Alignment'
            # for kernel in []:
            #     res_dict[k]['accuracy'].append()

    keys = list(res_dict.keys())
    keys.sort()
    for k in keys:
        plt.scatter(res_dict[k]['accuracy'], res_dict[k]['variance'], label = k)

    plt.title('Dataset: ' + dataset , fontweight='bold', fontsize=20)
    plt.xlabel(acc_name, fontsize=15)
    plt.ylabel('Variance', fontsize=15)
    plt.legend(prop={'size': 6})

    path = res_dir + '/' + dataset + '/plots'
    if not os.path.isdir(path): os.mkdir(path)
    plt.savefig(path + '/variance_' + type + '_' + differentiate + '.png')
    plt.clf()



# main function
def conf_process(file):
    global res_dir
    with open(file) as config_file:
        data = json.load(config_file)

    res_dir = data['base_dir']
    datasets = {}
    kernels = {}

    if not isinstance(data['experiments'], str):
        datasets_obj = [i['dataset'] for i in data['experiments']]
        kernels_lists = [i['training'] for i in data['experiments']]

        datanames = []
        for obj in datasets_obj:
            value_list = [str(obj[key]) for key in obj.keys()]
            datanames.append("_".join(value_list))

        for i in range(len(kernels_lists)):
            kernels[datanames[i]] = {}
            for k in kernels_lists[i]:
                value_list = [str(k[key]) for key in k.keys()]
                file = "_".join(value_list) + '.npy'
                kernels[datanames[i]]["_".join(value_list)] = load_kernel(res_dir + '/' + datanames[i] + '/kernels/' + file, value_list[0])

    else:
        datanames = [f.name for f in os.scandir(res_dir) if f.is_dir()]

        for data in datanames:
            kernels[data] = {}
            for f in os.listdir(res_dir + '/' + data + '/kernels'):
                if f.endswith('.npy'):
                    kernels[data][f[:-4]] = load_kernel(res_dir + '/' + data + '/kernels/' + f, f.split('_')[0])

    for name in datanames:
        datasets[name] = load_dataset(res_dir + '/' + name + '/' + name + '.npy', name.split('_')[0])

    print('\n##### DATASETS AND KERNELS LOADED #####')


    print('\n##### GA-HYPERPARAMETER ANALYSIS COMPLETED #####\n')

    print('\n##### SPECTRAL ANALYSIS COMPLETED #####\n')

    print('\n##### GRAM MATRICES ANALYSIS COMPLETED #####\n')

    for data in kernels.keys():
        plot_scatter_accuracy_variance(kernels[data], data, datasets[data]['train_y'] + datasets[data]['valid_y'], datasets[data]['test_y'])
        plot_scatter_accuracy_variance(kernels[data], data, datasets[data]['train_y'] + datasets[data]['valid_y'], datasets[data]['test_y'], differentiate = 'all')

    print('\n##### PERFORMANCE ANALYSIS COMPLETED #####\n')


def main(conf=False, file=None):
    print('\n##### PROCESS STARTED #####\n')
    if conf:
        conf_process(file)
    else:
        print("\n!!!!! CONFIGURATION FILE REQUIRED !!!!!\n")
    print("\n##### PROCESS COMPLETED #####\n")



if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(True, sys.argv[1])
    else:
        main()