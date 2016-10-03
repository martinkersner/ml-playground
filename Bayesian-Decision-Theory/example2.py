#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/10/03

# Inspired by Example 2 Error Bounds for Gaussian Distributions

import numpy as np

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

  # Fixes means
  #mean1 = np.matrix([[3], [6]])
  #mean2 = np.matrix([[3], [-2]])

  # Covariance matrices
  cov1 = np.cov(cluster1.T)
  cov2 = np.cov(cluster2.T)

  # Fixed covariance matrices
  #cov1 = np.matrix([[0.5, 0], [0, 2]])
  #cov2 = np.matrix([[2, 0],   [0, 2]])

  # Determinants of covariance matrices
  det_cov1 = np.linalg.det(cov1)
  det_cov2 = np.linalg.det(cov2)


  # Bhattacharyya bound
  # auxiliary variables
  mean_cov = (cov1 + cov2) / 2.0
  mean_diff = mean2 - mean1

  k_12 = 0.125*(mean_diff.T * np.linalg.inv(mean_cov) * mean_diff) + 0.5*np.log(np.linalg.det(mean_cov)/np.sqrt(det_cov1*det_cov2))
  P_error = np.sqrt(apriori1*apriori2) * np.exp(-k_12)

  print "Bhattacharyya bound"
  print "k(1/2)", k_12[0,0]
  print "P(error) <=", P_error

  # Chernoff bound
  # TODO optimal value of beta should be found by minimizing P(w1)**beta * P(w2)**(1-beta) * exp**(-k(beta))
  # auxiliary variables
  beta = 0.5 # in range 0 to 1
  harm_cov = (1-beta)*cov1 + beta*cov2

  # computation of determinant of harm_cov was missing in a book                             ______________________
  k_b = (0.5*beta*(1-beta)) * mean_diff.T * np.linalg.inv(harm_cov) * mean_diff + 0.5*np.log(np.linalg.det(harm_cov)/((det_cov1**(1-beta))*(det_cov2**beta)))
  P_error = np.exp(-k_b) 

  print "Chernoff bound"
  print "P(error) <=", P_error

if __name__ == "__main__":
  main()
