#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/10/03

# Whitening transform
# Data taken from Gaussian distribution

import numpy as np
import matplotlib.pyplot as plt

# Data preparation
mean  = np.array([0.0, 0.0])
C = np.array([[6.0, -4.0],
              [-4.0, 6.0]])
n_samples = 1000

A = np.random.multivariate_normal(mean, C, size=n_samples)

# Whitening
U, s, V = np.linalg.svd(C, full_matrices=True)
A_w = np.dot(A, (U / np.sqrt(s)))

# Plotting
plt.ylim([-8, 8])
plt.xlim([-8, 8])
plt.scatter(A[:, 0], A[:, 1], 1.0, color='black')
plt.scatter(A_w[:, 0], A_w[:, 1], 1.0, color='red')
plt.show()
