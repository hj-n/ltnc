import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



def dsphere(n=100, d=2, r=1, noise=None):

    data = np.random.randn(n, d)

    # Normalization
    data = r * data / np.sqrt(np.sum(data ** 2, 1)[:, None])

    if noise:
        data += noise * np.random.randn(*data.shape)

    return data


def create_sphere_dataset(total_samples=3000, d=100, n_spheres=6, r=5, var=10, seed=42, plot=False):
    np.random.seed(seed)

    variance = r / np.sqrt(d-1) * var

    shift_matrix = np.random.normal(0, variance, [n_spheres, d])

    spheres = []
    n_datapoints = 0
    n_samples = total_samples // n_spheres
    rest = total_samples % n_spheres

    if rest != 0:
        print(f"warnings... {total_samples} % {n_spheres} != 0 but {rest}...")

    for i in np.arange(n_spheres):
        sphere = dsphere(n=n_samples, d=d, r=r)
        sphere_shifted = sphere + shift_matrix[i, :]
        spheres.append(sphere_shifted)
        n_datapoints += n_samples

    if plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_spheres+1))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color], s=5.0)
        plt.savefig(f"sample_{r}_{var}_{total_samples}.png")

    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index : label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    return dataset, labels



