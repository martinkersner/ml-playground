#!/usr/binb/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/08/08

import numpy as np

np.random.seed(0)

def main():
    print generate_data()

def generate_data():
    shifted_data = generate_shifted_gaussian(shift=[20, 20])
    stretched_data = generate_stretched_gaussian()
    data = np.vstack([shifted_data, stretched_data])

    return data

def generate_shifted_gaussian(n_dims=2, n_samples=300, shift=[0, 0]):
    return np.random.randn(n_samples, n_dims) + np.array(shift)

def generate_stretched_gaussian(n_samples=300):
    C = np.array([[0., -0.7], [3.5, .7]])
    return np.dot(np.random.randn(n_samples, 2), C)

if __name__ == "__main__":
    main()
