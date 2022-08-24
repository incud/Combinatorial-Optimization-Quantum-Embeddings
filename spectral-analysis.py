from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
# import jax
# import pennylane as qml
# import jax.numpy as jnp
# import optax
# from quask.template_pennylane import pennylane_projected_quantum_kernel, hardware_efficient_ansatz, GeneticEmbedding
# import quask

from utils import *

res_dir = 'genetic-algorithm-results'

# ====================================================================
# ==================== LOAD DATASETS ====================
# ====================================================================

# load data
X = np.load(res_dir + '/X.npy') if Path(res_dir + '/X.npy').exists() else print('Missing synthetic dataset!')
# load model
weights = np.load(res_dir + '/weights.npy') if Path(res_dir + '/weights.npy').exists() else print('Missing synthetic weights!')
# load label
y = np.load(res_dir + '/y.npy') if Path(res_dir + '/y.npy').exists() else print('Missing synthetic labels!')



# ====================================================================
# =============== LOAD RANDOM QUANTUM KERNELS ===============
# ====================================================================

random_qk_1 = np.load(res_dir + '/random_qk_1.npy') if Path(res_dir + '/random_qk_1.npy').exists() else print('')
random_qk_2 = np.load(res_dir + '/random_qk_2.npy') if Path(res_dir + '/random_qk_2.npy').exists() else print('')
random_qk_3 = np.load(res_dir + '/random_qk_3.npy') if Path(res_dir + '/random_qk_3.npy').exists() else print('')

random_qk_1_spectrum = np.linalg.eigvals(random_qk_1)
random_qk_2_spectrum = np.linalg.eigvals(random_qk_2)
random_qk_3_spectrum = np.linalg.eigvals(random_qk_3)

random_qk_1_alignment = k_target_alignment(random_qk_1, y)
random_qk_2_alignment = k_target_alignment(random_qk_2, y)
random_qk_3_alignment = k_target_alignment(random_qk_3, y)
print("Random quantum kernel alignment:")
print(random_qk_1_alignment, random_qk_2_alignment, random_qk_3_alignment)
print(random_qk_1_spectrum.min(),random_qk_1_spectrum.max())



# ====================================================================
# ================ LOAD GENETIC QUANTUM KERNELS ==================
# ====================================================================

genetic_qk_1 = np.load(res_dir + '/genetic_qk_1.npy') if Path(res_dir + '/genetic_qk_1.npy').exists() else print('')
genetic_qk_2 = np.load(res_dir + '/genetic_qk_2.npy') if Path(res_dir + '/genetic_qk_2.npy').exists() else print('')
genetic_qk_3 = np.load(res_dir + '/genetic_qk_3.npy') if Path(res_dir + '/genetic_qk_3.npy').exists() else print('')

genetic_qk_1_spectrum = np.linalg.eigvals(genetic_qk_1)
genetic_qk_2_spectrum = np.linalg.eigvals(genetic_qk_2)
genetic_qk_3_spectrum = np.linalg.eigvals(genetic_qk_3)

genetic_qk_1_alignment = k_target_alignment(genetic_qk_1, y)
genetic_qk_2_alignment = k_target_alignment(genetic_qk_2, y)
genetic_qk_3_alignment = k_target_alignment(genetic_qk_3, y)
print("KTA GENETIC WITH BATCH 5")
print(genetic_qk_1_alignment, genetic_qk_2_alignment, genetic_qk_3_alignment)
print(genetic_qk_1_spectrum.min(),genetic_qk_1_spectrum.max())


# ====================================================================
# ============= CLASSIFY WITH TRAINABLE QUANTUM KERNELS ==============
# ====================================================================

params_tentatives = np.load(res_dir + '/trainable_params_init.npy') if Path(res_dir + '/trainable_params_init.npy').exists() else print ('')
params_trained = np.load(res_dir + '/trainable_params_end.npy') if Path(res_dir + '/trainable_params_end.npy').exists() else print('')

trainable_qk_1 = np.load(res_dir + '/trainable_qk_1.npy') if Path(res_dir + '/trainable_qk_1.npy').exists() else print('')
trainable_qk_2 = np.load(res_dir + '/trainable_qk_2.npy') if Path(res_dir + '/trainable_qk_2.npy').exists() else print('')
trainable_qk_3 = np.load(res_dir + '/trainable_qk_3.npy') if Path(res_dir + '/trainable_qk_3.npy').exists() else print('')

trainable_qk_1_spectrum = np.linalg.eigvals(trainable_qk_1)
trainable_qk_2_spectrum = np.linalg.eigvals(trainable_qk_2)
trainable_qk_3_spectrum = np.linalg.eigvals(trainable_qk_3)

trainable_qk_1_alignment = k_target_alignment(trainable_qk_1, y)
trainable_qk_2_alignment = k_target_alignment(trainable_qk_2, y)
trainable_qk_3_alignment = k_target_alignment(trainable_qk_3, y)
print(trainable_qk_1_alignment, trainable_qk_2_alignment, trainable_qk_3_alignment)
print(trainable_qk_1_spectrum.min(),trainable_qk_1_spectrum.max())



# ====================================================================
# ============= CLASSIFY WITH TRAINABLE QUANTUM KERNELS ==============
# ===================== FULL BATCH ===================================

params_tentatives = np.load(res_dir + '/trainable_params_init_fullbatch.npy') if Path(res_dir + '/trainable_params_init_fullbatch.npy').exists() else print('')
params_trained = np.load(res_dir + '/trainable_params_end_fullbatch.npy') if Path(res_dir + '/trainable_params_end_fullbatch.npy').exists() else print('')

trainable_qk_fb_1 = np.load(res_dir + '/trainable_qk_fb_1.npy') if Path(res_dir + '/trainable_qk_fb_1.npy').exists() else print('')
trainable_qk_fb_2 = np.load(res_dir + '/trainable_qk_fb_2.npy') if Path(res_dir + '/trainable_qk_fb_2.npy').exists() else print('')
trainable_qk_fb_3 = np.load(res_dir + '/trainable_qk_fb_3.npy') if Path(res_dir + '/trainable_qk_fb_3.npy').exists() else print('')

trainable_fb_qk_1_spectrum = np.linalg.eigvals(trainable_qk_fb_1)
trainable_fb_qk_2_spectrum = np.linalg.eigvals(trainable_qk_fb_2)
trainable_fb_qk_3_spectrum = np.linalg.eigvals(trainable_qk_fb_3)

trainable_qk_fb_1_alignment = k_target_alignment(trainable_qk_1, y)
trainable_qk_fb_2_alignment = k_target_alignment(trainable_qk_2, y)
trainable_qk_fb_3_alignment = k_target_alignment(trainable_qk_3, y)
print("Trainable full batch:")
print(trainable_qk_1_alignment, trainable_qk_2_alignment, trainable_qk_3_alignment)
print(trainable_fb_qk_1_spectrum.min(),trainable_fb_qk_1_spectrum.max())


# ====================================================================
# ================ CLASSIFY WITH GENETIC ALGORITHMS ==================
# ===================== FULL BATCH ===================================

genetic_fb_qk_1 = np.load(res_dir + '/genetic_qk_1_fullbatch.npy') if Path(res_dir + '/genetic_qk_1_fullbatch.npy').exists() else print('')
genetic_fb_qk_2 = np.load(res_dir + '/genetic_qk_2_fullbatch.npy') if Path(res_dir + '/genetic_qk_2_fullbatch.npy').exists() else print('')
genetic_fb_qk_3 = np.load(res_dir + '/genetic_qk_3_fullbatch.npy') if Path(res_dir + '/genetic_qk_3_fullbatch.npy').exists() else print('')

genetic_fb_qk_1_spectrum = np.linalg.eigvals(genetic_fb_qk_1)
genetic_fb_qk_2_spectrum = np.linalg.eigvals(genetic_fb_qk_2)
genetic_fb_qk_3_spectrum = np.linalg.eigvals(genetic_fb_qk_3)

genetic_fb_qk_1_alignment = k_target_alignment(genetic_fb_qk_1, y)
genetic_fb_qk_2_alignment = k_target_alignment(genetic_fb_qk_2, y)
genetic_fb_qk_3_alignment = k_target_alignment(genetic_fb_qk_3, y)
print("FULL BATCH GENETIC", genetic_fb_qk_2_alignment)
print(genetic_fb_qk_1_spectrum.min(),genetic_fb_qk_1_spectrum.max())


# ====================================================================
# ================================ PLOTS =============================
# ====================================================================


plt.clf()
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(10)

colors = ['#620808', '#bb1414', '#f14444']
labels = ['Random (1)', 'Random (2)', 'Random (3)']
values = [random_qk_1_spectrum,random_qk_2_spectrum,random_qk_3_spectrum]
ax = fig.add_subplot(321)
ax.hist(values, bins=range(0,36,2), width=1, color=colors, label=labels)
ax.legend()

colors = ['#173384', '#4062c4', '#7795eb']
labels = ['Trainable (1)', 'Trainable (2)', 'Trainable (3)']
values = [trainable_qk_1_spectrum,trainable_qk_2_spectrum,trainable_qk_3_spectrum]
ax = fig.add_subplot(323)
ax.hist(values, bins=range(0,36,2), width=1, color=colors, label=labels)
ax.legend()

colors = ['#173384', '#4062c4', '#7795eb']
labels = ['Trainable full batch (1)', 'Trainable full batch (2)', 'Trainable full batch (3)']
values = [trainable_fb_qk_1_spectrum,trainable_fb_qk_2_spectrum,trainable_fb_qk_3_spectrum]
ax = fig.add_subplot(324)
ax.hist(values, bins=range(0,36,2), width=1, color=colors, label=labels)
ax.legend()

colors = ['#1f7d16', '#3ca932', '#65e159']
labels = ['Genetic (1)', 'Genetic (2)', 'Genetic (3)']
values = [genetic_qk_1_spectrum,genetic_qk_2_spectrum,genetic_qk_3_spectrum]
ax = fig.add_subplot(325)
ax.hist(values, bins=range(0,36,2), width=1, color=colors, label=labels)
ax.legend()

colors = ['#1f7d16', '#3ca932', '#65e159']
labels = ['Genetic full batch (1)', 'Genetic full batch (2)', 'Genetic full batch (3)']
values = [genetic_fb_qk_1_spectrum,genetic_fb_qk_2_spectrum,genetic_fb_qk_3_spectrum]
ax = fig.add_subplot(326)
ax.hist(values, bins=range(0,36,2), width=1, color=colors, label=labels)
ax.legend()

#plt.savefig(f'comparison_1_eigenvalues_{datetime.now().strftime("%y%m%d_%H%M%S_%f")}.png')
plt.savefig(f'comparison_1_eigenvalues.png')



# plt.clf()
# fig = plt.figure()
# fig.set_figwidth(10)
# fig.set_figheight(10)
# bins = [0] + [10**i for i in range(-19,1,1)]
#
# colors = ['#620808', '#bb1414', '#f14444']
# labels = ['Random (1)', 'Random (2)', 'Random (3)']
# values = [random_qk_1_spectrum,random_qk_2_spectrum,random_qk_3_spectrum]
# ax = fig.add_subplot(321)
# ax.hist(values, bins=bins, width=1, color=colors, label=labels)
# ax.set_xscale('log')
# ax.legend()
#
# colors = ['#173384', '#4062c4', '#7795eb']
# labels = ['Trainable (1)', 'Trainable (2)', 'Trainable (3)']
# values = [trainable_qk_1_spectrum,trainable_qk_2_spectrum,trainable_qk_3_spectrum]
# ax = fig.add_subplot(323)
# ax.hist(values, bins=bins, width=1, color=colors, label=labels)
# ax.set_xscale('log')
# ax.legend()
#
# colors = ['#173384', '#4062c4', '#7795eb']
# labels = ['Trainable full batch (1)', 'Trainable full batch (2)', 'Trainable full batch (3)']
# values = [trainable_fb_qk_1_spectrum,trainable_fb_qk_2_spectrum,trainable_fb_qk_3_spectrum]
# ax = fig.add_subplot(324)
# ax.hist(values, bins=bins, width=1, color=colors, label=labels)
# ax.set_xscale('log')
# ax.legend()
#
# colors = ['#1f7d16', '#3ca932', '#65e159']
# labels = ['Genetic (1)', 'Genetic (2)', 'Genetic (3)']
# values = [genetic_qk_1_spectrum,genetic_qk_2_spectrum,genetic_qk_3_spectrum]
# ax = fig.add_subplot(325)
# ax.hist(values, bins=bins, width=1, color=colors, label=labels)
# ax.set_xscale('log')
# ax.legend()
#
# colors = ['#1f7d16', '#3ca932', '#65e159']
# labels = ['Genetic full batch (1)', 'Genetic full batch (2)', 'Genetic full batch (3)']
# values = [genetic_fb_qk_1_spectrum,genetic_fb_qk_2_spectrum,genetic_fb_qk_3_spectrum]
# ax = fig.add_subplot(326)
# ax.hist(values, bins=bins, width=1, color=colors, label=labels)
# ax.set_xscale('log')
# ax.legend()
#
# #plt.savefig(f'comparison_1_eigenvalues_{datetime.now().strftime("%y%m%d_%H%M%S_%f")}.png')
# plt.savefig(f'comparison_2_first_bin.png')

