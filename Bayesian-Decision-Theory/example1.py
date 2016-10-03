#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/10/03

# Computing Bayesian decision boundary for two-dimensional data with arbitrary 
# covariance matrix.

# Inspired by Example 1 Decision Regions for Two-Dimensional Gaussian Data from
# book Pattern classification

import numpy as np
import matplotlib.pyplot as plt

def main():
  apriori1 = 0.5
  cluster1 = np.array([[3, 4],
                       [2, 6],
                       [4, 6],
                       [3, 8]])

  apriori2 = 0.5
  cluster2 = np.array([[3, 0],
                       [1, -2],
                       [3, -4],
                       [5, -2]])

  # Means
  mean1 = np.asmatrix(np.mean(cluster1, axis=0)).T
  mean2 = np.asmatrix(np.mean(cluster2, axis=0)).T

  # Covariance matrices
  cov1 = np.cov(cluster1.T)
  cov2 = np.cov(cluster2.T)

  # Inverse of covariance matrices
  inv_cov1 = np.linalg.inv(cov1)
  inv_cov2 = np.linalg.inv(cov2)

  # Determinants of covariance matrices
  det_cov1 = np.linalg.det(cov1)
  det_cov2 = np.linalg.det(cov2)

  # W_i = -1/2*inv(cov_i)
  W1 = -0.5 * inv_cov1
  W2 = -0.5 * inv_cov2
  A = W1 - W2

  # w_i = inv(cov_i)*mean_i
  w1 = inv_cov1 * mean1
  w2 = inv_cov2 * mean2
  B = (w1 - w2).T

  # w_i0 = -1/2*mean_i.T*inv(cov_i)*mean_i + 1/2*ln(det(cov_i)) + ln(apriori_i)
  w10 = -0.5 * mean1.T * inv_cov1 * mean1 + 1/2 * np.log(det_cov1) + apriori1 
  w11 = -0.5 * mean2.T * inv_cov2 * mean2 + 1/2 * np.log(det_cov2) + apriori2 
  C = w10 - w11

  # Discriminant function
  # g_i(x) = x.T*W_i*x + w_i.T*x + w_i0

  eps = 0.00001 # for numeric stability
  x1 = np.arange(-10.0, 10.0, 0.01)
  x2_left =  (-(2*x1*(A[1,0]+A[0,1]) + B[0,0]) + (np.sqrt((2*x1*(A[1,0] + A[0,1]) + B[0,0])**2 - 4*A[0,0]*(A[1,1]*x1**2 + B[0,1]*x1 + C + eps))))/(2*A[0,0])
  x2_right = (-(2*x1*(A[1,0]+A[0,1]) + B[0,0]) - (np.sqrt((2*x1*(A[1,0] + A[0,1]) + B[0,0])**2 - 4*A[0,0]*(A[1,1]*x1**2 + B[0,1]*x1 + C + eps))))/(2*A[0,0])

  # Solution from book
  #x2 = 3.514 - 1.125*x1 + 0.1875*x1**2

  plt.plot(x2_left.T, x1, c='green')
  plt.plot(x2_right.T, x1, c='green')

  plot_clusters(cluster1, cluster2)
  plt.show()

def plot_clusters(c1, c2):
  plt.scatter(c1[:,0], c1[:,1], c='blue')
  plt.scatter(c2[:,0], c2[:,1], c='red')

if __name__ == "__main__":
  main()
