#!/usr/bin/env python

'''
Martin Kersner, m.kersner@gmail.com
2016/08/11

Generate multivariate Gaussian distribution in two dimensions and fit linear model on it.
'''

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

def main():
  mean  = np.array([0.0, 0.0])
  C = np.array([[6.0, -4.0],
                [-4.0, 6.0]])
  n_samples = 1000

  data = np.random.multivariate_normal(mean, C, size=n_samples)
  X = map(lambda x: list([x]), data[:, 0])
  y = data[:, 1]

  clf = linear_model.LinearRegression()
  clf.fit(X, y)

  print "Coefficients: \n", clf.coef_
  print "Intercept: \n", clf.intercept_

  ax = plt.axes()
  plot_data(data)
  plt.axis('equal') # plots graph with equal axis ratio
  plt.show()

def plot_data(data):
  plt.scatter(data[:, 0], data[:, 1], 1.0)

if __name__ == "__main__":
  main()
