import sys
import os
import json
from jax.config import config

from quask.template_pennylane import GeneticEmbeddingUnstructured

config.update("jax_enable_x64", True)
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# retrieve the upper triangular section of a matrix excluding the diagonal
def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]


# scatter plot of accuracy and variance
def plot_kernels_eigenvalues_variance(kernels):
    path = res_dir + 'plots'
    if not os.path.isdir(path): os.mkdir(path)
    if not os.path.isdir(path + '/variance'): os.mkdir(path + '/variance')
    if not os.path.isdir(path + '/spectrum'): os.mkdir(path + '/spectrum')
    res_all = {}
    var_all = {}
    labels_all = []
    f = plt.figure()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    red_square = dict(markerfacecolor='r', markersize=3, markeredgewidth=0.0)

    for d in kernels.keys():
        reskeys = [k for k in kernels[d].keys()]
        reskeys.sort()
        res_dataset = {}
        var_dataset = {}
        labels_dataset = []
        for k in reskeys:
            res_dataset[k] = []
            var_dataset[k] = []
            labels_dataset.append(k)
            if k not in labels_all: labels_all.append(k)
            if k not in res_all.keys(): res_all[k] = []
            if k not in var_all.keys(): var_all[k] = []
            for i in range(len(kernels[d][k])):
                if 'gram_train' in kernels[d][k][i].keys():
                    eig_list = np.linalg.eigvals(kernels[d][k][i]['gram_train']).tolist()
                    entr_list = upper_tri_indexing(kernels[d][k][i]['gram_train'])

                    if len(res_dataset[k]) != 0:
                        res_dataset[k] = np.concatenate((res_dataset[k], eig_list))
                        var_dataset[k] = np.concatenate((var_dataset[k], entr_list))
                    else:
                        res_dataset[k] = eig_list
                        var_dataset[k] = entr_list
                    res_dataset[k] = np.array(res_dataset[k]).astype(complex)
                    res_dataset[k] = res_dataset[k].real

                    if len(res_all) != 0:
                        res_all[k] = np.concatenate((res_all[k], eig_list))
                        var_all[k] = np.concatenate((var_dataset[k], entr_list))
                    else:
                        res_all[k] = eig_list
                        var_all[k] = entr_list
                    res_all[k] = np.array(res_all[k]).astype(complex)
                    res_all[k] = res_all[k].real

        # Density Plot with Rug
        gfg = sns.displot(res_dataset, kind="kde", bw_method=0.2, rug=True, height=5, aspect=1.5)
        gfg.set_axis_labels('Eigenvalues', 'Density')
        gfg.tight_layout()
        sns.move_legend(gfg, "upper right", bbox_to_anchor=(.8, .9))
        plt.setp(gfg._legend.get_texts(), fontsize=8)
        plt.savefig(path + '/spectrum/eigenvalues_density_' + d + '.png')
        plt.close()

        # Violin plot
        l = [i for i in range(len(res_dataset.keys()))]
        rot= 45
        v = [res_dataset[k].tolist() for k in res_dataset.keys()]
        plt.violinplot(
            v,
            l,
            showmeans=True,
            showextrema=True,
            widths= 0.8
        )
        plt.title('Dataset: ' + d, fontsize=15)
        plt.ylabel("Eigenvalues Distribution")
        plt.xticks(l, res_dataset.keys(), rotation=rot, fontsize=6)
        plt.subplots_adjust(bottom=0.25)
        plt.savefig(path + '/spectrum/eigenvalues_violin_' + d + '.png')
        plt.close()

        # Box Plot
        l = [i for i in range(len(res_dataset.keys()))]
        rot= 45
        v = [res_dataset[k].tolist() for k in res_dataset.keys()]
        plt.boxplot(
            v,
            notch=False,
            whis=3,
            flierprops=red_square,
        )
        plt.title('Dataset: ' + d, fontsize=15)
        plt.yscale('symlog', linthresh=10**-3)
        plt.ylabel("Eigenvalues Distribution")
        plt.xticks([i +1 for i in l], res_dataset.keys(), rotation=rot, fontsize=6)
        plt.subplots_adjust(bottom=0.25)
        plt.savefig(path + '/spectrum/eigenvalues_boxplot_' + d + '.png')
        plt.close()

        # Violin Plot entries
        rot = 0
        var_labels = [ l + '\n\nVariance: '+ str(np.var(var_dataset[l]))[:6] for l in labels_dataset]
        v = [var_dataset[k].tolist() for k in var_dataset.keys()]
        plt.violinplot(
            v,
            l,
            showmeans=True,
            showextrema=True,
            widths= 0.8
        )
        plt.title('Dataset: ' + d, fontsize=15)
        plt.xlabel("Kernels (higher variance is better)")
        plt.ylabel("Gram Matrix Entries Distribution")
        plt.xticks(l, var_labels, rotation=rot, fontsize=6)
        plt.subplots_adjust(bottom=0.25)
        plt.savefig(path + '/variance/variance_violin_' + d + '.png')
        plt.close()


    # Density Plot with Rug (all)
    gfg = sns.displot(res_all, kind="kde", bw_method=0.2, rug=True, height=5, aspect=1.5)
    gfg.set_axis_labels('Eigenvalues', 'Density')
    gfg.tight_layout()
    sns.move_legend(gfg, "upper right", bbox_to_anchor=(.8, .9))
    plt.setp(gfg._legend.get_texts(), fontsize=8)
    plt.savefig(path + '/spectrum/eigenvalues_density_all.png')
    plt.close()

    # Violin plot all
    l = [i for i in range(len(res_all.keys()))]
    rot = 45
    v = [res_all[k].tolist() for k in res_all.keys()]
    plt.violinplot(
        v,
        l,
        showmeans=True,
        showextrema=True,
        widths=0.8
    )
    plt.title('Dataset: all datasets', fontsize=15)
    plt.ylabel("Eigenvalues Distribution")
    plt.xticks(l, res_all.keys(), rotation=rot, fontsize=6)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(path + '/spectrum/eigenvalues_violin_all.png')
    plt.close()

    # Violin Plot entries all
    rot = 0
    var_labels = [l + '\n\nVariance: ' + str(np.var(var_all[l]))[:6] for l in labels_all]
    v = [var_all[k].tolist() for k in var_all.keys()]
    plt.violinplot(
        v,
        l,
        showmeans=True,
        showextrema=True,
        widths=0.8
    )
    plt.title('Dataset: ' + d, fontsize=15)
    plt.xlabel("Kernels (higher variance is better)")
    plt.ylabel("Gram Matrix Entries Distribution")
    plt.xticks(l, var_labels, rotation=rot, fontsize=6)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(path + '/variance/variance_violin_all.png')
    plt.close()

    # Box Plot all
    l = [i for i in range(len(res_all.keys()))]
    rot = 45
    v = [res_all[k].tolist() for k in res_all.keys()]
    plt.boxplot(
        v,
        notch=False,
        whis=3,
        flierprops=red_square
    )
    plt.title('Dataset: all', fontsize=15)
    plt.yscale('symlog', linthresh=10 ** -3)
    plt.ylabel("Eigenvalues Distribution")
    plt.xticks([i + 1 for i in l], res_all.keys(), rotation=rot, fontsize=6)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(path + '/spectrum/eigenvalues_boxplot_all.png')
    plt.close()



# main function
def conf_process(file):
    global res_dir

    res_dir = file
    datasets = {}
    kernels = {}

    datanames = [f.name for f in os.scandir(res_dir + 'datasets') if f.is_dir()]
    for name in datanames:
        if name not in datasets.keys(): datasets[name] = {}
        datasets[name]['X_train']         = np.load(f"{res_dir}datasets/{name}/X_train.npy")
        datasets[name]['X_validation']    = np.load(f"{res_dir}datasets/{name}/X_validation.npy")
        datasets[name]['X_test']          = np.load(f"{res_dir}datasets/{name}/X_test.npy")
        datasets[name]['y_train']         = np.load(f"{res_dir}datasets/{name}/y_train.npy")
        datasets[name]['y_validation']    = np.load(f"{res_dir}datasets/{name}/y_validation.npy")
        datasets[name]['y_test']          = np.load(f"{res_dir}datasets/{name}/y_test.npy")

    resdatanames = [f.name for f in os.scandir(res_dir + 'intermediate') if f.is_dir()]
    for data in resdatanames:
        kernelsnames = [f.name for f in os.scandir(res_dir + 'intermediate/' + data) if f.is_dir()]
        kernels[data] = {}
        for k in kernelsnames:
            kernels[data][k] = []
            for f in os.scandir(res_dir + 'intermediate/' + data + '/' + k):
                if f.is_dir():
                    kernel = {}
                    for npy in os.listdir(res_dir + 'intermediate/' + data + '/' + k + '/' + f.name):
                        if npy.endswith('.npy'):
                            kernel[npy[:-4]] = np.load(res_dir + 'intermediate/' + data + '/' + k + '/' + f.name + '/' + npy)
                            kernel['dir'] = f
                    kernels[data][k].append(kernel)

    print('\n##### DATASETS AND KERNELS LOADED #####\n')

    plot_kernels_eigenvalues_variance(kernels)

    print('\n##### SPECTRAL ANALYSIS COMPLETED #####\n')


def main(conf=False, file=None):
    print('\n##### PROCESS STARTED #####\n')
    if conf:
        conf_process(file)
    else:
        print("\n!!!!! BASE DIRECTORY REQUIRED !!!!!\n")
    print("\n##### PROCESS COMPLETED #####\n")



if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1][len(sys.argv[1])-1] != '/': sys.argv[1] += '/'
        main(True, sys.argv[1])
    else:
        main(True, "")
