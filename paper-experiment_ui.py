import os
import sys
import json
from jax.config import config
import jax.numpy as jnp
config.update("jax_enable_x64", True)
# import numpy as np
from pathlib import Path
import optax
# import datetime
# import jax
# import pennylane as qml
# from quask.template_pennylane import pennylane_projected_quantum_kernel, hardware_efficient_ansatz, GeneticEmbedding
# import quask

from utils import *
from datagen import *
from ktrain import *
from ktest import *

res_dir = 'experiments-results'
kernel_dir = '/kernels'

menu = '''
TASKS: 
  1- Generate synthetic dataset.
  2- Generate random kernel.
  3- Train quantum kernel with a VQA. 
  4- Train quantum kernel with a genetic algorithm.

Choose a task by submitting its index or press any key to stop the program:'''



def conf_process(file):
    global res_dir
    with open(file) as config_file:
        data = json.load(config_file)

    res_dir = data['base_dir']
    datasets = [i['dataset'] for i in data['experiments']]
    kernels_lists = [i['training'] for i in data['experiments']]
    datanames = []
    count = 0

    for dts in datasets:
        flag, name = generate_data(dts, res_dir)
        if flag:
            count += 1
            datanames.append(name)
        else:
            kernels_lists.pop(count)

    print('\n##### DATASETS GENERATED #####')

    kernelfiles = []
    count = 0
    for i in range(len(kernels_lists)):
        dataname = datanames[i]
        kernels = kernels_lists[i]
        print('\nTraining with dataset: ' + dataname)
        for k in kernels:
            path = res_dir + '/' + dataname
            flag, name = train_kernel(k, path, dataname)
            if flag:
                count += 1
                kernelfiles.append(path + '/' + name)

    print('\n##### KERNELS GENERATED #####\n')



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