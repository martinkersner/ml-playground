#!/usr/binb/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/08/08

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def main():
    X = generate_data()
    plot_data(X)

def generate_data():
    data = np.vstack([  shift(generate_gaussian(), offset=[10, 10]), \
                        stretch(generate_gaussian()), \
                        shift(stretch(generate_gaussian(), C=[[1.5, 0.4],[1.9, 1.1]]), offset=[-10, 10]) \
                    ])

    return data

def shift(data, offset=[0, 0]):
    return data + np.array(offset)

def stretch(data, C=[[0.0, 0.7], [8.5, 1.0]]):
    return np.dot(data, np.array(C))

def generate_gaussian(n_samples=300, n_dims=2):
    return np.random.randn(n_samples, n_dims)

def plot_data(data, margin=10):
    min_x = np.min(data[:, 0])
    min_y = np.min(data[:, 1])
    max_x = np.max(data[:, 0])
    max_y = np.max(data[:, 1])

    x = np.linspace(min_x+margin, max_x+margin)
    y = np.linspace(min_y+margin, max_y+margin)

    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    plt.scatter(data[:, 0], data[:, 1], 0.5)

    plt.axis('tight')
    plt.show()

if __name__ == "__main__":
    main()
