import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Q0.1 a
np.random.seed(0)
x = np.random.uniform(-2, 5, size=1000)

x_std = np.sqrt(np.mean((x - np.mean(x)) ** 2))
# print(x_std == np.std(x))


# Q0.1 b

def plot_hist_with_density_uniform(x, a, b, num_bins=50):
    plt.hist(x, bins=num_bins, density=True)
    plt.plot(x, np.repeat(1 / (b - a), len(x)))
    plt.show()


# plot_hist_with_density_uniform(x, -2, 5)
# plot_hist_with_density_uniform(np.random.uniform(-2, 5, size=10000), -2, 5, 100)


# Q0.1 c

def plot_hist_with_density_normal(x, mu, std, num_bins=50):
    plt.hist(x, bins=num_bins, density=True)
    plt.plot(np.sort(x), norm.pdf(np.sort(x), mu, std))
    plt.show()


x = np.random.normal(3, 3, 1000)
# plot_hist_with_density_normal(x, 3, 3)
# plot_hist_with_density_normal(np.random.normal(3, 3, 10000), 3, 3)


# Q0.1 d

def plot_normal_scatter(x_mu, x_std, y_mu, y_std, sample_size):
    x, y = np.random.normal(x_mu, x_std, sample_size), np.random.normal(y_mu, y_std, sample_size)
    plt.scatter(x, y, s=1)
    plt.show()


# plot_normal_scatter(3, 2, 5, 1, 1000)
# plot_normal_scatter(3, 2, 5, 1, 10000)

# Q0.1 e

x, y = np.random.normal(4, 2, 10000), np.random.normal(5, 1, 10000)
print(np.mean(x < y))
