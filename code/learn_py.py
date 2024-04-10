# install packages
#pip install scprep graphtools

# load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scprep as scprep
import graphtools as gt
from sklearn import datasets, preprocessing

# set random seed
np.random.seed(0)

# Generate datasets. We choose the size to be big enough to see the scalability
# of the algorithms, but not too big to cause long running times

n_samples = 300

# Circles
noisy_circles = datasets.make_circles(n_samples=n_samples, 
                        # Scale factor between inner and outer circle
                        factor=.5,
                        # Gaussian noise added to each point
                        noise=.05)

# Moons
noisy_moons = datasets.make_moons(n_samples=n_samples, 
                                  noise=.05)

# Blobs
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

# Uniform square
no_structure = np.random.rand(n_samples, 2), None

# "Blobs_skew"
# Anisotropically distributed data (i.e. data with unequal length and width)
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
# Changes how x1, x2 coordinates are shifted
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

generated_datasets = {'circles':noisy_circles,
     'moons':noisy_moons,
     'blobs_variance':varied,
     'blobs_skew':aniso,
     'blobs_regular':blobs,
     'uniform':no_structure}


# Plot the datasets
fig, axes = plt.subplots(1,6,figsize=(12,2))

for i, dataset_name in enumerate(generated_datasets):
    ax = axes[i]
    data, group = generated_datasets[dataset_name]

    # normalize dataset for easier parameter selection
    data = preprocessing.StandardScaler().fit_transform(data)
    scprep.plot.scatter2d(data, c=group, 
                          ticks=None, ax=ax, 
                          xlabel='x0', ylabel='x1',
                          title=dataset_name,
                         legend=False)

fig.tight_layout()