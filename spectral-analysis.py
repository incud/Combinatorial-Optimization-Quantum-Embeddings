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

params_tentatives_fb = np.load(res_dir + '/trainable_params_init_fullbatch.npy') if Path(res_dir + '/trainable_params_init_fullbatch.npy').exists() else print('')
params_trained_fb = np.load(res_dir + '/trainable_params_end_fullbatch.npy') if Path(res_dir + '/trainable_params_end_fullbatch.npy').exists() else print('')

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

# ================================ EIGENVALUES =============================
plt.clf()
fig = plt.figure()
fig.set_figwidth(24)
fig.set_figheight(10)
bins = [i for i in range(-2,36,2)]

colors = ['#620808', '#bb1414', '#f14444']
labels = ['Random (1)', 'Random (2)', 'Random (3)']
values = [random_qk_1_spectrum,random_qk_2_spectrum,random_qk_3_spectrum]
ax = fig.add_subplot(321)
ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
ax.legend()

ax = fig.add_subplot(322)
ax.text(0, 0.9, 'Eigenvalues Distribution' ,size=15)
ax.text(0, 0.62, 'Each histogram describes the eigenvalues \ndistribution of all the kernels that has been \ngenerated with a specific technique.' ,size=10)
ax.axis('off')

colors = ['#173384', '#4062c4', '#7795eb']
labels = ['Trainable (1)', 'Trainable (2)', 'Trainable (3)']
values = [trainable_qk_1_spectrum,trainable_qk_2_spectrum,trainable_qk_3_spectrum]
ax = fig.add_subplot(323)
ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
ax.legend()

colors = ['#173384', '#4062c4', '#7795eb']
labels = ['Trainable full batch (1)', 'Trainable full batch (2)', 'Trainable full batch (3)']
values = [trainable_fb_qk_1_spectrum,trainable_fb_qk_2_spectrum,trainable_fb_qk_3_spectrum]
ax = fig.add_subplot(324)
ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
ax.legend()

colors = ['#1f7d16', '#3ca932', '#65e159']
labels = ['Genetic (1)', 'Genetic (2)', 'Genetic (3)']
values = [genetic_qk_1_spectrum,genetic_qk_2_spectrum,genetic_qk_3_spectrum]
ax = fig.add_subplot(325)
ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
ax.legend()

colors = ['#1f7d16', '#3ca932', '#65e159']
labels = ['Genetic full batch (1)', 'Genetic full batch (2)', 'Genetic full batch (3)']
values = [genetic_fb_qk_1_spectrum,genetic_fb_qk_2_spectrum,genetic_fb_qk_3_spectrum]
ax = fig.add_subplot(326)
ax.hist(values, bins=bins, width=0.4, color=colors, label=labels)
ax.legend()

#plt.savefig(f'comparison_1_eigenvalues_{datetime.now().strftime("%y%m%d_%H%M%S_%f")}.png')
plt.savefig(res_dir + f'/figures/comparison_1_eigenvalues.png')




# ================================ EIGENVALUES SMALL BINS =============================

plt.clf()
fig = plt.figure()
fig.set_figwidth(24)
fig.set_figheight(10)
# set width 1e-6
barWidth = 0.25
tmp = [(2**i)*-1 for i in range(-5,-3,1)]
tmp.reverse()
bins = tmp + [0] + [2**i for i in range(-5,4,1)]


ax = fig.add_subplot(321)
# set height of bar
d1 = compute_histogram(random_qk_1_spectrum, bins)
d2 = compute_histogram(random_qk_2_spectrum,bins)
d3 = compute_histogram(random_qk_3_spectrum,bins)

# Set position of bar on X axis
br1 = np.arange(len(d1))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
b1 = ax.plot(bins, d1, color='#620808',  label='Random (1)')
b2 = ax.plot(bins, d2, color='#bb1414',  label='Random (2)')
b3 = ax.plot(bins, d3, color='#f14444',  label='Random (3)')
ax.set_xscale('symlog', base=2, linthresh=2**-5)

ax.legend()

ax = fig.add_subplot(322)
ax.text(0, 0.85, 'Eigenvalues Distribution: small values' ,size=20)
ax.text(0, 0.5, 'Each line represent the histogram of the eigenvalues in the interval [-2, 2].\nEach bin contains the eigenvalues that has label greater than the bin\'s label\nbut smaller than the successive.' ,size=15)
ax.axis('off')

ax = fig.add_subplot(323)
# set height of plot
d1 = compute_histogram(trainable_qk_1_spectrum, bins)
d2 = compute_histogram(trainable_qk_2_spectrum,bins)
d3 = compute_histogram(trainable_qk_3_spectrum,bins)
# Make the plot
b1 = ax.plot(bins, d1, color='#173384',  label='Trainable (1)')
b2 = ax.plot(bins, d2, color='#4062c4',  label='Trainable (2)')
b3 = ax.plot(bins, d3, color='#7795eb',  label='Trainable (3)')
ax.set_xscale('symlog', base=2, linthresh=2**-5)
ax.legend()

ax = fig.add_subplot(324)
# set height of plot
d1 = compute_histogram(trainable_fb_qk_1_spectrum, bins)
d2 = compute_histogram(trainable_fb_qk_2_spectrum,bins)
d3 = compute_histogram(trainable_fb_qk_3_spectrum,bins)
# Make the plot
b1 = ax.plot(bins, d1, color='#173384',  label='Trainable full batch (1)')
b2 = ax.plot(bins, d2, color='#4062c4',  label='Trainable full batch (2)')
b3 = ax.plot(bins, d3, color='#7795eb',  label='Trainable full batch (3)')
ax.set_xscale('symlog', base=2, linthresh=2**-5)
ax.legend()

ax = fig.add_subplot(325)
# set height of plot
d1 = compute_histogram(genetic_qk_1_spectrum, bins)
d2 = compute_histogram(genetic_qk_2_spectrum,bins)
d3 = compute_histogram(genetic_qk_3_spectrum,bins)
# Make the plot
b1 = ax.plot(bins, d1, color='#1f7d16',  label='Genetic (1)')
b2 = ax.plot(bins, d2, color= '#3ca932',  label='Genetic (2)')
b3 = ax.plot(bins, d3, color= '#65e159',  label='Genetic (3)')
ax.set_xscale('symlog', base=2, linthresh=2**-5)
ax.legend()


ax = fig.add_subplot(326)
# set height of plot
d1 = compute_histogram(genetic_fb_qk_1_spectrum, bins)
d2 = compute_histogram(genetic_fb_qk_2_spectrum,bins)
d3 = compute_histogram(genetic_fb_qk_3_spectrum,bins)
# Make the plot
b1 = ax.plot(bins, d1, color='#1f7d16',  label='Genetic full batch (1)')
b2 = ax.plot(bins, d2, color= '#3ca932',  label='Genetic full batch (2)')
b3 = ax.plot(bins, d3, color= '#65e159',  label='Genetic full batch (3)')
ax.set_xscale('symlog', base=2, linthresh=2**-5)
plt.legend()
plt.savefig(res_dir + f'/figures/comparison_2_small_bins')




# set width 1e-6
barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))

# set height of bar
accur = [
    np.mean([np.mean(accuracy(random_qk_1,X,y)),np.mean(accuracy(random_qk_2,X,y)),np.mean(accuracy(random_qk_3,X,y))]),
    np.mean([np.mean(accuracy(trainable_qk_1,X,y)),np.mean(accuracy(trainable_qk_2,X,y)),np.mean(accuracy(trainable_qk_3,X,y))]),
    np.mean([np.mean(accuracy(trainable_qk_fb_1,X,y)),np.mean(accuracy(trainable_qk_fb_2,X,y)),np.mean(accuracy(trainable_qk_fb_3,X,y))]),
    np.mean([np.mean(accuracy(genetic_qk_1,X,y)),np.mean(accuracy(genetic_qk_2,X,y)),np.mean(accuracy(genetic_qk_3,X,y))]),
    np.mean([np.mean(accuracy(genetic_fb_qk_1,X,y)),np.mean(accuracy(genetic_fb_qk_2,X,y)),np.mean(accuracy(genetic_fb_qk_3,X,y))])
]

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

br0 = np.arange(len(means))
br1 = [x + barWidth for x in br0]
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
b0 = plt.bar(br0, accur, color='#23ea4d', width=barWidth,
        edgecolor='white', label='Accuracy')
b1 = plt.bar(br1, means, color='#630a4d', width=barWidth,
        edgecolor='white', label='Means')
b2 = plt.bar(br2, variance, color='#c439a1', width=barWidth,
        edgecolor='white', label='Variance')
b3 = plt.bar(br3, standarddev, color='#ed91d6', width=barWidth,
        edgecolor='white', label='Standard Deviation')

plt.bar_label(b0,fmt='%.3f')
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






print(np.allclose(trainable_qk_1,trainable_qk_fb_1) and np.allclose(trainable_qk_2,trainable_qk_fb_2) and np.allclose(trainable_qk_3,trainable_qk_fb_3) and np.allclose(params_tentatives_fb,params_trained))


print("Random 1:\t" + str(accuracy(random_qk_1,X,y)))
print("Random 2:\t" + str(accuracy(random_qk_2,X,y)))
print("Random 3:\t" + str(accuracy(random_qk_3,X,y)))
print("Trainable 1:\t" + str(accuracy(trainable_qk_1,X,y)))
print("Trainable 2:\t" + str(accuracy(trainable_qk_2,X,y)))
print("Trainable 3:\t" + str(accuracy(trainable_qk_3,X,y)))
print("Genetic full batch 1:\t" + str(accuracy(genetic_fb_qk_1,X,y)))
print("Genetic full batch 2:\t" + str(accuracy(genetic_fb_qk_2,X,y)))
print("Genetic full batch 3:\t" + str(accuracy(genetic_fb_qk_3,X,y)))

svm = SVR().fit(X, y)
y_predict = svm.predict(X)
print(mean_squared_error(y, y_predict))