import sys
import json
from jax.config import config

from quask.template_pennylane import GeneticEmbeddingUnstructured

config.update("jax_enable_x64", True)
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *


# prepare quantum feature map
def plot_quantum_feature_map(dataset, kernelname, val, params, ge):

    n = len(val)
    if kernelname.split('_')[0] == 'random':
        embedding = lambda x, wires: (random_quantum_embedding(x, wires, params['seed']))
    elif kernelname.split('_')[0] == 'trainable':
        embedding = lambda x, wires: (trainable_embedding(x, params['params'], n, wires=wires))
    elif kernelname.split('_')[0] == 'genetic':
        embedding = lambda x, wires: (ge.transform_solution_to_embedding(x, params['best_solution']))

    device = qml.device("default.qubit.jax", wires=n)

    # define the circuit for the quantum kernel ("overlap test" circuit)
    @qml.qnode(device, interface='jax')
    def proj_feature_map(x):
        embedding(x, wires=range(n))
        return (
            [qml.expval(qml.PauliX(i)) for i in range(n)]
        )

    qml.drawer.use_style('black_white')
    fig, ax = qml.draw_mpl(proj_feature_map, decimals=3,expansion_strategy="device")(val)

    path = res_dir + '/' + dataset + '/plots'
    if not os.path.isdir(path): os.mkdir(path)
    plt.savefig(path + '/circuit_' + kernelname + '.png')
    plt.close()



# scatter plot of accuracy and variance
def plot_kernels_eigenvalues_variance(kernels, dataset, data, differentiate = 'kernel'):
    res_dict = {}

    f = plt.figure()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    for k in kernels.keys():
        key = compute_key(k, differentiate, 'kernel')

        if key in res_dict.keys():
            res_dict[key]['kernels'].append(kernels[k])
        else:
            res_dict[key] = {}
            res_dict[key]['kernels'] = [kernels[k]]

    labels = []
    res = {}
    min_size = np.inf
    max_eig = -np.inf
    for k in res_dict.keys():
        if min_size > len(res_dict[k]['kernels']): min_size = len(res_dict[k]['kernels'])
        res_dict[k]['mean'] = 0
        res_dict[k]['variance'] = 0
        res_dict[k]['standard deviation'] = 0
        res_dict[k]['eigevalues'] = []
        res_dict[k]['acc'] = -np.inf
        for i in res_dict[k]['kernels']:
            y_train = data['train_y'] + data['valid_y']
            y_test = data['test_y']
            tmp_acc = accuracy_svr(i['K'], i['K_test'], np.ravel(y_train), np.ravel(y_test))
            if res_dict[k]['acc'] < tmp_acc:
                res_dict[k]['mean'] = np.mean(upper_tri_indexing(i['K']))
                res_dict[k]['variance'] = np.var(upper_tri_indexing(i['K']))
                res_dict[k]['standard deviation'] = np.std(upper_tri_indexing(i['K']))
                res_dict[k]['eigenvalues'] = np.linalg.eigvals(i['K']).tolist()
                res_dict[k]['acc'] = tmp_acc
                if max_eig < max(res_dict[k]['eigenvalues']): max_eig = max(res_dict[k]['eigenvalues'])

    count = 0
    reskeys = [ k for k in res_dict.keys()]
    reskeys.sort()
    for k in reskeys:
        res[k] = []
        labels.append(k)
        for i in range(min_size):
            if len(res) != 0:
                res[k] = res[k] + np.linalg.eigvals(res_dict[k]['kernels'][i]['K']).tolist()
            else:
                res[k] = np.linalg.eigvals(res_dict[k]['kernels'][i]['K']).tolist()
        res[k] = np.array(res[k]).astype(complex)
        res[k] = res[k].real
        count +=1


    # Density Plot with Rug Plot
    gfg = sns.displot(res, kind="kde", bw_method=0.2, rug=True, height=5, aspect=1.5)
    gfg.set_axis_labels('Eigenvalues', 'Density')
    gfg.tight_layout()
    sns.move_legend(gfg, "upper right", bbox_to_anchor=(.8, .9))
    plt.setp(gfg._legend.get_texts(), fontsize=8)

    path = res_dir + '/' + dataset + '/plots'
    if not os.path.isdir(path): os.mkdir(path)
    plt.savefig(path + '/eigenvalues_density_' + differentiate + '.png')
    plt.close()


    # variance means and standard dev
    labels = ['mean', 'variance', 'standard deviation']
    barw = 0.2
    br_old = []
    # Make the plot
    for l in labels:
        if len(br_old)==0:
            br_old = np.arange(len(res_dict.keys()))
            br = br_old
        else:
            br = [x + barw for x in br_old]
            br_old = br

        tmp = plt.bar(br, [res_dict[k][l] for k in res_dict.keys()], width=barw,
                 edgecolor='white', label=l)

        plt.bar_label(tmp, fmt='%.3f', fontsize=6)

    # Adding Xticks
    plt.xlabel('Quantum Kernels', fontweight='bold', fontsize=15)
    plt.ylabel('Values', fontweight='bold', fontsize=15)
    plt.xticks([r + barw for r in range(len(res_dict.keys()))],
               res_dict.keys(), rotation = 45, fontsize=7)
    plt.tight_layout()
    plt.legend()
    path = res_dir + '/' + dataset + '/plots'
    if not os.path.isdir(path): os.mkdir(path)
    plt.savefig(path + '/entries_metrics_' + dataset + '.png')
    plt.close()


    # histogram of best models
    for k in reskeys:
        gfg = sns.displot({k: res_dict[k]['eigenvalues']}, bins=range(-1, int(max_eig +2), max([1, int((max_eig +2)/10)])), rug=True, height=5, aspect=1.5)
        gfg.set_axis_labels('Eigenvalues', 'Count')
        gfg.tight_layout()
        sns.move_legend(gfg, "upper right", bbox_to_anchor=(.8, .9))
        plt.setp(gfg._legend.get_texts(), fontsize=8)

        path = res_dir + '/' + dataset + '/plots'
        if not os.path.isdir(path): os.mkdir(path)
        plt.savefig(path + '/eigenvalues_hist_' + k + '.png')
        plt.close()

    if differentiate == 'all':
        l = [i for i in range(0,len(res.keys())*3,3)]
        rot= 45
    else:
        l = [i for i in range(0,len(res.keys()))]
        rot=0
    v = [res[k].tolist() for k in res.keys()]
    plt.violinplot(
        v,
        l,
        showmeans=True,
        showextrema=True,
    )
    plt.title('Dataset: ' + compute_key(dataset, 'all', 'dataset'), fontsize=15)
    plt.ylabel("Eigenvalues Distribution")
    plt.xticks(l, res.keys(), rotation=rot, fontsize=6)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(path + '/eigenvalues_violin_' + differentiate + '.png')
    plt.close()




# scatter plot of accuracy and variance
def plot_scatter_accuracy_variance(kernels, dataset, y_train, y_test, type = 'mse', differentiate = 'kernel'):
    res_dict = {}
    acc_name = ''
    plt.figure()
    for k in kernels.keys():
        key = compute_key(k, differentiate, 'kernel')

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
                res_dict[k]['accuracy'].append(accuracy_svr(kernel['K'], kernel['K_test'], np.ravel(y_train), np.ravel(y_test)))
        elif type == 'kta':
            pass
            acc_name = 'Kernel-Target Alignment (Training)'
            for kernel in res_dict[k]['kernels']:
                res_dict[k]['accuracy'].append(k_target_alignment(kernel['K'], np.array(y_train).ravel()))

    keys = list(res_dict.keys())
    keys.sort()
    for k in keys:
        plt.scatter(res_dict[k]['accuracy'], res_dict[k]['variance'], label = k)

    plt.title('Dataset: ' + compute_key(dataset, 'all', 'dataset'), fontsize=15)
    plt.xlabel(acc_name, fontsize=15)
    plt.ylabel('Variance', fontsize=15)
    plt.legend(prop={'size': 6})

    path = res_dir + '/' + dataset + '/plots'
    if not os.path.isdir(path): os.mkdir(path)
    plt.savefig(path + '/variance_' + type + '_' + differentiate + '.png')
    plt.close()



# main function
def plot_hyperparams_analysis(kernels, data):
    max_v = {}
    length = {}

    gens_max = max([int(i.split('_')[1]) for i in kernels.keys() if i.split('_')[0] == 'genetic'])

    for kernel in kernels.keys():
        if kernel.split('_')[0] == 'genetic' and int(kernel.split('_')[1]) == gens_max:
            key = compute_key(kernel, 'all', 'kernel')
            if key not in max_v.keys():
                max_v[key] = []
                length[key] = []
            max_v[key].append([max(i) if len(i)!=0 else 0 for i in kernels[kernel]['low_variance_list']])
            length[key].append([len(i) for i in kernels[kernel]['low_variance_list']])

    fig, ax = plt.subplots()
    for i in length.keys():
        ax.plot(range(gens_max + 1),
            np.mean(length[i], axis=0), label=i)
    plt.title('Dataset: ' + compute_key(data, 'all', 'dataset'), fontsize=15)
    ax.set_xlabel("Generations")
    ax.set_ylabel("Excluded Kernels Count")
    plt.legend()

    path = res_dir + '/' + data + '/plots'
    if not os.path.isdir(path): os.mkdir(path)
    plt.savefig(path + '/hyper_analysis_count_' + str(gens_max) + '.png')
    plt.close()


    fig, ax = plt.subplots()
    for i in max_v.keys():
        ax.plot(range(gens_max + 1),
            np.mean(max_v[i], axis=0),label=i)
    ax.set_xlabel("Generations")
    ax.set_ylabel("Max Excluded Variance")
    plt.title('Dataset: ' + compute_key(data, 'all', 'dataset'), fontsize=15)
    plt.legend()

    path = res_dir + '/' + data + '/plots'
    if not os.path.isdir(path): os.mkdir(path)
    plt.savefig(path + '/hyper_analysis_max_' + str(gens_max) + '.png')
    plt.close()


def plot_variance_analysis(kernels, data):
    values = {}

    for kernel in kernels.keys():
        key = compute_key(kernel, 'all', 'kernel')
        if key not in values.keys():
            values[key] = []
        values[key] = np.concatenate([values[key], upper_tri_indexing(kernels[kernel]['K'])])


    l = [i for i in range(0, len(values.keys()) * 3, 3)]
    rot = 45
    v = [values[k].tolist() for k in values.keys()]
    plt.violinplot(
        v,
        l,
        showmeans=True,
        showextrema=True,
    )
    plt.title('Dataset: ' + compute_key(data, 'all', 'dataset'), fontsize=15)
    plt.ylabel("Gram Matrix Entries Distribution")
    plt.xticks(l, values.keys(), rotation=rot, fontsize=6)
    plt.subplots_adjust(bottom=0.25)
    path = res_dir + '/' + data + '/plots'
    if not os.path.isdir(path): os.mkdir(path)
    plt.savefig(path + '/gram_matrix_violin_' + data + '.png')
    plt.close()


def plot_gram_matrix(k, data, kernel):
    plt.title('Kernel: ' + kernel, fontsize=10)
    ax = sns.heatmap(k)
    path = res_dir + '/' + data + '/plots'
    if not os.path.isdir(path): os.mkdir(path)
    plt.savefig(path + '/gram_matrix_' + kernel + '.png')
    plt.close()


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

    print('\n##### DATASETS AND KERNELS LOADED #####\n')

    for data in kernels.keys():
        plot_hyperparams_analysis(kernels[data], data)

    print('\n##### GA-HYPERPARAMETER ANALYSIS COMPLETED #####\n')


    for data in kernels.keys():
        plot_kernels_eigenvalues_variance(kernels[data], data, datasets[data])
        plot_kernels_eigenvalues_variance(kernels[data], data, datasets[data], differentiate='all')

    print('\n##### SPECTRAL ANALYSIS COMPLETED #####\n')


    for data in kernels.keys():
        plot_variance_analysis(kernels[data], data)
        for kernel in kernels[data].keys():
            plot_gram_matrix(kernels[data][kernel]['K'], data, kernel)

    print('\n##### GRAM MATRICES ANALYSIS COMPLETED #####\n')


    for data in kernels.keys():
        plot_scatter_accuracy_variance(kernels[data], data, datasets[data]['train_y'] + datasets[data]['valid_y'], datasets[data]['test_y'])
        plot_scatter_accuracy_variance(kernels[data], data, datasets[data]['train_y'] + datasets[data]['valid_y'], datasets[data]['test_y'], differentiate = 'all')
        plot_scatter_accuracy_variance(kernels[data], data, datasets[data]['train_y'] + datasets[data]['valid_y'],
                                       datasets[data]['test_y'], type = 'kta')
        plot_scatter_accuracy_variance(kernels[data], data, datasets[data]['train_y'] + datasets[data]['valid_y'],
                                       datasets[data]['test_y'], type = 'kta', differentiate='all')

    print('\n##### PERFORMANCE ANALYSIS COMPLETED #####\n')


    for data in kernels.keys():
        for kernel in kernels[data].keys():
            ge = None
            params = {}
            val = datasets[data]['test_x'][0]

            if kernel.split('_')[0] == 'random':
                params['seed'] = int(kernel.split('_')[1])
                val = kernels[data][kernel]['weights'][0]
            elif kernel.split('_')[0] == 'trainable':
                params['params'] = kernels[data][kernel]['trained_params']
            elif kernel.split('_')[0] == 'genetic':
                params['best_solution'] = kernels[data][kernel]['best_solution']
                if kernel.split('_')[7] == 'unstructured':
                    ge = GeneticEmbeddingUnstructured(datasets[data]['test_x'], datasets[data]['test_y'], len(val), len(val), 0.01, num_parents_mating=1)
                else:
                    ge = GeneticEmbedding(datasets[data]['test_x'], datasets[data]['test_y'], len(val), len(val), 0.01, num_parents_mating=1)
            plot_quantum_feature_map(data, kernel, val, params, ge)

    print('\n##### CIRCUITS REPRESENTATION GENERATED #####\n')



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