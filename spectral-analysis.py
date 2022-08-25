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
from quask.metrics import calculate_generalization_accuracy
# import quask

from utils import *

res_dir = 'genetic-algorithm-results'

# ====================================================================
# ==================== LOAD DATASETS ====================
# ====================================================================

X = np.load(res_dir + '/X.npy') if Path(res_dir + '/X.npy').exists() else print('Missing synthetic dataset!')
weights = np.load(res_dir + '/weights.npy') if Path(res_dir + '/weights.npy').exists() else print('Missing synthetic weights!')
y = np.load(res_dir + '/y.npy') if Path(res_dir + '/y.npy').exists() else print('Missing synthetic labels!')

# np.random.seed(77546)
#
# indices = np.random.permutation(y.shape[0])
# training_idx, test_idx = indices[:int((y.shape[0]/2))], indices[int((y.shape[0]/2)):]
# y_train, y_test = y[training_idx], y[test_idx]


# ====================================================================
# =============== LOAD RANDOM QUANTUM KERNELS ===============
# ====================================================================

random_qk_1 = np.load(res_dir + '/random_qk_1.npy') if Path(res_dir + '/random_qk_1.npy').exists() else print('')
random_qk_2 = np.load(res_dir + '/random_qk_2.npy') if Path(res_dir + '/random_qk_2.npy').exists() else print('')
random_qk_3 = np.load(res_dir + '/random_qk_3.npy') if Path(res_dir + '/random_qk_3.npy').exists() else print('')

random_qk_1_spectrum = np.linalg.eigvals(random_qk_1)
random_qk_2_spectrum = np.linalg.eigvals(random_qk_2)
random_qk_3_spectrum = np.linalg.eigvals(random_qk_3)
np.save('genetic-algorithm-results/random_qk_1_spectrum.npy', random_qk_1_spectrum)
np.save('genetic-algorithm-results/random_qk_2_spectrum.npy', random_qk_2_spectrum)
np.save('genetic-algorithm-results/random_qk_3_spectrum.npy', random_qk_3_spectrum)

random_qk_1_alignment = k_target_alignment(random_qk_1, y)
random_qk_2_alignment = k_target_alignment(random_qk_2, y)
random_qk_3_alignment = k_target_alignment(random_qk_3, y)
# accuracy_random_qk = [calculate_generalization_accuracy(random_qk_1,y_train,random_qk_1,y_test),
#                       calculate_generalization_accuracy(random_qk_2,y_train,random_qk_2,y_test),
#                       calculate_generalization_accuracy(random_qk_3,y_train,random_qk_3,y_test)]


# ====================================================================
# ================ LOAD GENETIC QUANTUM KERNELS ==================
# ====================================================================

genetic_qk_1 = np.load(res_dir + '/genetic_qk_1.npy') if Path(res_dir + '/genetic_qk_1.npy').exists() else print('')
genetic_qk_2 = np.load(res_dir + '/genetic_qk_2.npy') if Path(res_dir + '/genetic_qk_2.npy').exists() else print('')
genetic_qk_3 = np.load(res_dir + '/genetic_qk_3.npy') if Path(res_dir + '/genetic_qk_3.npy').exists() else print('')

genetic_qk_1_spectrum = np.linalg.eigvals(genetic_qk_1)
genetic_qk_2_spectrum = np.linalg.eigvals(genetic_qk_2)
genetic_qk_3_spectrum = np.linalg.eigvals(genetic_qk_3)
np.save('genetic-algorithm-results/genetic_qk_1_spectrum.npy', genetic_qk_1_spectrum)
np.save('genetic-algorithm-results/genetic_qk_2_spectrum.npy', genetic_qk_2_spectrum)
np.save('genetic-algorithm-results/genetic_qk_3_spectrum.npy', genetic_qk_3_spectrum)


genetic_qk_1_alignment = k_target_alignment(genetic_qk_1, y)
genetic_qk_2_alignment = k_target_alignment(genetic_qk_2, y)
genetic_qk_3_alignment = k_target_alignment(genetic_qk_3, y)



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
np.save('genetic-algorithm-results/trainable_qk_1_spectrum.npy', trainable_qk_1_spectrum)
np.save('genetic-algorithm-results/trainable_qk_2_spectrum.npy', trainable_qk_2_spectrum)
np.save('genetic-algorithm-results/trainable_qk_3_spectrum.npy', trainable_qk_3_spectrum)

trainable_qk_1_alignment = k_target_alignment(trainable_qk_1, y)
trainable_qk_2_alignment = k_target_alignment(trainable_qk_2, y)
trainable_qk_3_alignment = k_target_alignment(trainable_qk_3, y)



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
np.save('genetic-algorithm-results/trainable_fb_qk_1_spectrum.npy', trainable_fb_qk_1_spectrum)
np.save('genetic-algorithm-results/trainable_fb_qk_2_spectrum.npy', trainable_fb_qk_2_spectrum)
np.save('genetic-algorithm-results/trainable_fb_qk_3_spectrum.npy', trainable_fb_qk_3_spectrum)

trainable_qk_fb_1_alignment = k_target_alignment(trainable_qk_1, y)
trainable_qk_fb_2_alignment = k_target_alignment(trainable_qk_2, y)
trainable_qk_fb_3_alignment = k_target_alignment(trainable_qk_3, y)


# ====================================================================
# ================ CLASSIFY WITH GENETIC ALGORITHMS ==================
# ===================== FULL BATCH ===================================

genetic_fb_qk_1 = np.load(res_dir + '/genetic_qk_1_fullbatch.npy') if Path(res_dir + '/genetic_qk_1_fullbatch.npy').exists() else print('')
genetic_fb_qk_2 = np.load(res_dir + '/genetic_qk_2_fullbatch.npy') if Path(res_dir + '/genetic_qk_2_fullbatch.npy').exists() else print('')
genetic_fb_qk_3 = np.load(res_dir + '/genetic_qk_3_fullbatch.npy') if Path(res_dir + '/genetic_qk_3_fullbatch.npy').exists() else print('')

genetic_fb_qk_1_spectrum = np.linalg.eigvals(genetic_fb_qk_1)
genetic_fb_qk_2_spectrum = np.linalg.eigvals(genetic_fb_qk_2)
genetic_fb_qk_3_spectrum = np.linalg.eigvals(genetic_fb_qk_3)
np.save('genetic-algorithm-results/genetic_fb_qk_1_spectrum.npy', genetic_fb_qk_1_spectrum)
np.save('genetic-algorithm-results/genetic_fb_qk_2_spectrum.npy', genetic_fb_qk_2_spectrum)
np.save('genetic-algorithm-results/genetic_fb_qk_3_spectrum.npy', genetic_fb_qk_3_spectrum)

genetic_fb_qk_1_alignment = k_target_alignment(genetic_fb_qk_1, y)
genetic_fb_qk_2_alignment = k_target_alignment(genetic_fb_qk_2, y)
genetic_fb_qk_3_alignment = k_target_alignment(genetic_fb_qk_3, y)


# ====================================================================
# ================================ PLOTS =============================
# ====================================================================


# plt.clf()
# fig = plt.figure()
# fig.set_figwidth(10)
# fig.set_figheight(10)
# bins = [i for i in range(-2,36,2)]
#
# colors = ['#620808', '#bb1414', '#f14444']
# labels = ['Random (1)', 'Random (2)', 'Random (3)']
# values = [random_qk_1_spectrum,random_qk_2_spectrum,random_qk_3_spectrum]
# ax = fig.add_subplot(321)
# ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
# ax.legend()
#
# ax = fig.add_subplot(322)
# ax.text(0, 0.9, 'Eigenvalues Distribution' ,size=15)
# ax.text(0, 0.62, 'Each histogram describes the eigenvalues \ndistribution of all the kernels that has been \ngenerated with a specific technique.' ,size=10)
# ax.axis('off')
#
# colors = ['#173384', '#4062c4', '#7795eb']
# labels = ['Trainable (1)', 'Trainable (2)', 'Trainable (3)']
# values = [trainable_qk_1_spectrum,trainable_qk_2_spectrum,trainable_qk_3_spectrum]
# ax = fig.add_subplot(323)
# ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
# ax.legend()
#
# colors = ['#173384', '#4062c4', '#7795eb']
# labels = ['Trainable full batch (1)', 'Trainable full batch (2)', 'Trainable full batch (3)']
# values = [trainable_fb_qk_1_spectrum,trainable_fb_qk_2_spectrum,trainable_fb_qk_3_spectrum]
# ax = fig.add_subplot(324)
# ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
# ax.legend()
#
# colors = ['#1f7d16', '#3ca932', '#65e159']
# labels = ['Genetic (1)', 'Genetic (2)', 'Genetic (3)']
# values = [genetic_qk_1_spectrum,genetic_qk_2_spectrum,genetic_qk_3_spectrum]
# ax = fig.add_subplot(325)
# ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
# ax.legend()
#
# colors = ['#1f7d16', '#3ca932', '#65e159']
# labels = ['Genetic full batch (1)', 'Genetic full batch (2)', 'Genetic full batch (3)']
# values = [genetic_fb_qk_1_spectrum,genetic_fb_qk_2_spectrum,genetic_fb_qk_3_spectrum]
# ax = fig.add_subplot(326)
# ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
# ax.legend()
#
# #plt.savefig(f'comparison_1_eigenvalues_{datetime.now().strftime("%y%m%d_%H%M%S_%f")}.png')
# plt.savefig(res_dir + f'/figures/comparison_1_eigenvalues.png')
#
#
#
# plt.clf()
# fig = plt.figure()
# fig.set_figwidth(10)
# fig.set_figheight(10)
# bins = [0] + [10**i for i in range(-19,2,1)]
#
# colors = ['#620808', '#bb1414', '#f14444']
# labels = ['Random (1)', 'Random (2)', 'Random (3)']
# values = [random_qk_1_spectrum,random_qk_2_spectrum,random_qk_3_spectrum]
# ax = fig.add_subplot(321)
# ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
# ax.set_xscale('log')
# ax.legend()
#
# colors = ['#173384', '#4062c4', '#7795eb']
# labels = ['Trainable (1)', 'Trainable (2)', 'Trainable (3)']
# values = [trainable_qk_1_spectrum,trainable_qk_2_spectrum,trainable_qk_3_spectrum]
# ax = fig.add_subplot(323)
# ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
# ax.set_xscale('log')
# ax.legend()
#
# colors = ['#173384', '#4062c4', '#7795eb']
# labels = ['Trainable full batch (1)', 'Trainable full batch (2)', 'Trainable full batch (3)']
# values = [trainable_fb_qk_1_spectrum,trainable_fb_qk_2_spectrum,trainable_fb_qk_3_spectrum]
# ax = fig.add_subplot(324)
# ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
# ax.set_xscale('log')
# ax.legend()
#
# colors = ['#1f7d16', '#3ca932', '#65e159']
# labels = ['Genetic (1)', 'Genetic (2)', 'Genetic (3)']
# values = [genetic_qk_1_spectrum,genetic_qk_2_spectrum,genetic_qk_3_spectrum]
# ax = fig.add_subplot(325)
# ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
# ax.set_xscale('log')
# ax.legend()
#
# colors = ['#1f7d16', '#3ca932', '#65e159']
# labels = ['Genetic full batch (1)', 'Genetic full batch (2)', 'Genetic full batch (3)']
# values = [genetic_fb_qk_1_spectrum,genetic_fb_qk_2_spectrum,genetic_fb_qk_3_spectrum]
# ax = fig.add_subplot(326)
# ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
# ax.set_xscale('log')
# ax.legend()
#
# plt.savefig(res_dir + f'/figures/comparison_2_first_bin.png')





# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))

# set height of bar
means = [
    np.mean([np.mean(upper_tri_indexing(random_qk_1)),np.mean(upper_tri_indexing(random_qk_2)),np.mean(upper_tri_indexing(random_qk_3))]),
    np.mean([np.mean(upper_tri_indexing(trainable_qk_1)),np.mean(upper_tri_indexing(trainable_qk_2)),np.mean(upper_tri_indexing(trainable_qk_3))]),
    np.mean([np.mean(upper_tri_indexing(trainable_qk_fb_1)),np.mean(upper_tri_indexing(trainable_qk_fb_2)),np.mean(upper_tri_indexing(trainable_qk_fb_3))]),
    np.mean([np.mean(upper_tri_indexing(genetic_qk_1)),np.mean(upper_tri_indexing(genetic_qk_2)),np.mean(upper_tri_indexing(genetic_qk_3))]),
    np.mean([np.mean(upper_tri_indexing(genetic_fb_qk_1)),np.mean(upper_tri_indexing(genetic_fb_qk_2)),np.mean(upper_tri_indexing(genetic_fb_qk_3))])
]
variance = [
    np.mean([np.var(upper_tri_indexing(random_qk_1)),np.var(upper_tri_indexing(random_qk_2)),np.var(upper_tri_indexing(random_qk_3))]),
    np.mean([np.var(upper_tri_indexing(trainable_qk_1)),np.var(upper_tri_indexing(trainable_qk_2)),np.var(upper_tri_indexing(trainable_qk_3))]),
    np.mean([np.var(upper_tri_indexing(trainable_qk_fb_1)),np.var(upper_tri_indexing(trainable_qk_fb_2)),np.var(upper_tri_indexing(trainable_qk_fb_3))]),
    np.mean([np.var(upper_tri_indexing(genetic_qk_1)),np.var(upper_tri_indexing(genetic_qk_2)),np.var(upper_tri_indexing(genetic_qk_3))]),
    np.mean([np.var(upper_tri_indexing(genetic_fb_qk_1)),np.var(upper_tri_indexing(genetic_fb_qk_2)),np.var(upper_tri_indexing(genetic_fb_qk_3))])
]
standarddev = [
    np.mean([np.std(upper_tri_indexing(random_qk_1)),np.std(upper_tri_indexing(random_qk_2)),np.std(upper_tri_indexing(random_qk_3))]),
    np.mean([np.std(upper_tri_indexing(trainable_qk_1)),np.std(upper_tri_indexing(trainable_qk_2)),np.std(upper_tri_indexing(trainable_qk_3))]),
    np.mean([np.std(upper_tri_indexing(trainable_qk_fb_1)),np.std(upper_tri_indexing(trainable_qk_fb_2)),np.std(upper_tri_indexing(trainable_qk_fb_3))]),
    np.mean([np.std(upper_tri_indexing(genetic_qk_1)),np.std(upper_tri_indexing(genetic_qk_2)),np.std(upper_tri_indexing(genetic_qk_3))]),
    np.mean([np.std(upper_tri_indexing(genetic_fb_qk_1)),np.std(upper_tri_indexing(genetic_fb_qk_2)),np.std(upper_tri_indexing(genetic_fb_qk_3))])
]

# Set position of bar on X axis
br1 = np.arange(len(means))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
b1 = plt.bar(br1, means, color='#630a4d', width=barWidth,
        edgecolor='white', label='Means')
b2 = plt.bar(br2, variance, color='#c439a1', width=barWidth,
        edgecolor='white', label='Variance')
b3 = plt.bar(br3, standarddev, color='#ed91d6', width=barWidth,
        edgecolor='white', label='Standard Deviation')

plt.bar_label(b1,fmt='%.3f')
plt.bar_label(b2,fmt='%.3f')
plt.bar_label(b3,fmt='%.3f')


# Adding Xticks
plt.xlabel('Quantum Kernels', fontweight='bold', fontsize=15)
plt.ylabel('Values', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(means))],
           ['Random', 'Trainable', 'Trainable (full batch)', 'Genetic', 'Genetic (full batch)'])

plt.legend()
plt.savefig(res_dir + f'/figures/comparison_3_m_v_ds.png')


plt.clf()
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(10)
bins = [0] + [10**i for i in range(-19,2,1)]

colors = ['#620808', '#bb1414', '#f14444']
labels = ['Random (1)', 'Random (2)', 'Random (3)']
values = [random_qk_1_spectrum,random_qk_2_spectrum,random_qk_3_spectrum]
ax = fig.add_subplot(321)
ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
ax.set_xscale('log')
ax.legend()

colors = ['#173384', '#4062c4', '#7795eb']
labels = ['Trainable (1)', 'Trainable (2)', 'Trainable (3)']
values = [trainable_qk_1_spectrum,trainable_qk_2_spectrum,trainable_qk_3_spectrum]
ax = fig.add_subplot(323)
ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
ax.set_xscale('log')
ax.legend()

colors = ['#173384', '#4062c4', '#7795eb']
labels = ['Trainable full batch (1)', 'Trainable full batch (2)', 'Trainable full batch (3)']
values = [trainable_fb_qk_1_spectrum,trainable_fb_qk_2_spectrum,trainable_fb_qk_3_spectrum]
ax = fig.add_subplot(324)
ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
ax.set_xscale('log')
ax.legend()

colors = ['#1f7d16', '#3ca932', '#65e159']
labels = ['Genetic (1)', 'Genetic (2)', 'Genetic (3)']
values = [genetic_qk_1_spectrum,genetic_qk_2_spectrum,genetic_qk_3_spectrum]
ax = fig.add_subplot(325)
ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
ax.set_xscale('log')
ax.legend()

colors = ['#1f7d16', '#3ca932', '#65e159']
labels = ['Genetic full batch (1)', 'Genetic full batch (2)', 'Genetic full batch (3)']
values = [genetic_fb_qk_1_spectrum,genetic_fb_qk_2_spectrum,genetic_fb_qk_3_spectrum]
ax = fig.add_subplot(326)
ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
ax.set_xscale('log')
ax.legend()

plt.savefig(res_dir + f'/figures/comparison_2_first_bin.png')