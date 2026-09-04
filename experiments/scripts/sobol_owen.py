import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from scipy.stats import qmc
import numpy as np

# Define the parameter dimensions and the truncated budget (N1)
dimensions = 2
n_samples = 90

# 1. Generate Raw Sobol' Sequence (Unscrambled)
sobol_raw = qmc.Sobol(d=dimensions, scramble=False)
points_raw = sobol_raw.random(n=n_samples)

# 2. Generate Owen's Scrambled Sobol' Sequence
sobol_scrambled = qmc.Sobol(d=dimensions, scramble=True)
points_scrambled = sobol_scrambled.random(n=n_samples)

# 3. Generate Pseudo-Random Sequence
# Setting a seed for reproducibility
np.random.seed(42)
points_random = np.random.rand(n_samples, dimensions)

# Plot the comparison side-by-side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot Unscrambled Sobol'
ax1.scatter(points_raw[:, 0], points_raw[:, 1], alpha=0.7, color='#1f77b4')
ax1.set_title(f"Raw Sobol'\n(Truncated at {n_samples})")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect('equal')
ax1.grid(True, linestyle='--', alpha=0.5)

# Plot Scrambled Sobol'
ax2.scatter(points_scrambled[:, 0], points_scrambled[:, 1], alpha=0.7, color='#d62728')
ax2.set_title(f"Owen's Scrambled Sobol'\n(Truncated at {n_samples})")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect('equal')
ax2.grid(True, linestyle='--', alpha=0.5)

# Plot Pseudo-Random
ax3.scatter(points_random[:, 0], points_random[:, 1], alpha=0.7, color='#2ca02c')
ax3.set_title(f"Pseudo-Random\n(Standard Uniform)")
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_aspect('equal')
ax3.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()