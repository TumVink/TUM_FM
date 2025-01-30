import numpy as np
from sklearn.mixture import GaussianMixture

# Assuming you have the channel-wise statistics in a numpy array
# Each row corresponds to a sample, and each column corresponds to a statistic (mean or std for a channel)
# For example, if you have 1000 samples and each sample has 6 statistics (3 channels, mean and std for each)
data = np.array([...])  # Replace with your actual data

# Number of components for the GMM
n_components = 10

# Fit the GMM
gmm = GaussianMixture(n_components=n_components, covariance_type='full')
gmm.fit(data)

# Sample new statistics from the GMM
n_samples = 1  # Number of new samples you want to generate
sampled_statistics = gmm.sample(n_samples)[0]

print(sampled_statistics)