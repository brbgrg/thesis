from enigmatoolbox.datasets import load_sc, load_fc
from nilearn import plotting
import matplotlib.pyplot as plt


# Load cortico-cortical functional connectivity data
fc_ctx, fc_ctx_labels, _, _ = load_fc()

# Load cortico-cortical structural connectivity data
sc_ctx, sc_ctx_labels, _, _ = load_sc()

# Plot cortico-cortical connectivity matrices
fc_plot = plotting.plot_matrix(fc_ctx, figure=(9, 9), labels=fc_ctx_labels, vmax=0.8, vmin=0, cmap='Reds')
sc_plot = plotting.plot_matrix(sc_ctx, figure=(9, 9), labels=sc_ctx_labels, vmax=10, vmin=0, cmap='Blues')


plt.show()
