import sys
import json
from jax.config import config
config.update("jax_enable_x64", True)
from utils import *


def conf_process(file):
    global res_dir
    with open(file) as config_file:
        data = json.load(config_file)

    res_dir = data['base_dir']
    datasets = [i['dataset'] for i in data['experiments']]
    kernels_lists = [i['training'] for i in data['experiments']]

    datanames = []

    print('\n##### KERNELS LOADED #####')

    kernelfiles = []
    count = 0

    print('\n##### GA-HYPERPARAMETER ANALYSIS COMPLETED #####\n')

    print('\n##### SPECTRAL ANALYSIS COMPLETED #####\n')

    print('\n##### GRAM MATRICES ANALYSIS COMPLETED #####\n')

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