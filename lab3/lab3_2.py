import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import FastICA

import imageio

# READ IMAGES

fp1 = imageio.v3.imread('https://raw.githubusercontent.com/lorek/datasets/master/fp1.png', pilmode='L')
fp2 = imageio.v3.imread('https://raw.githubusercontent.com/lorek/datasets/master/fp2.png', pilmode='L')
fp3 = imageio.v3.imread('https://raw.githubusercontent.com/lorek/datasets/master/fp3.png', pilmode='L')

print(fp1.shape)
h, w = fp1.shape  # we assume all images have the same size
n = h * w

# CONSTRUCT S
S = np.array([fp1.ravel(), fp2.ravel(), fp3.ravel()])
print(S.shape)

# CONSTRUCT A1, A2

A1 = np.array([[5, 10, 0],
              [10, 0, 40],
              [0, 18, 50]])


A2 = np.array([[1, 1, 0],
              [0, 1, 1],
              [1, 0, 1]])


# DRAW MIX SIGNAL MATRIX (X = A @ S) IN GRAY SCALE

X1 = A1 @ S
X2 = A2 @ S


def draw_mix_signal_matrix(X, str_name_A="A1"):
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"Mix signal matrix on gray scale - matrix {str_name_A}")

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Finger print 1")
    ax1.imshow(X[0, :].reshape(h, w), cmap=plt.get_cmap('gray'))

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Finger print 2")
    ax2.imshow(X[1, :].reshape(h, w), cmap=plt.get_cmap('gray'))

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.set_title("Finger print 3")
    ax2.imshow(X[2, :].reshape(h, w), cmap=plt.get_cmap('gray'))

    plt.show()


draw_mix_signal_matrix(X1)
draw_mix_signal_matrix(X2, "A2")


def draw_reconstructed_S_ICA(X, str_name_A="A1", normalized=True):
    fig = plt.figure(figsize=(14, 8))

    if normalized:
        ica = FastICA(whiten=False)
        X_norm = normalize(X)
        S_reconstructed = ica.fit_transform(X_norm.T).T  # Reconstruct signals
        print(S_reconstructed.shape)

        fig.suptitle(f"Reconstructed S from ICA {str_name_A} with normalizing")
    else:
        ica = FastICA(n_components=3, whiten='unit-variance')
        S_reconstructed = ica.fit_transform(X.T).T  # Reconstruct signals

        fig.suptitle(f"Reconstructed S from ICA ({str_name_A}) without normalizing")

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Finger print 1")
    ax1.imshow(S_reconstructed[0, :].reshape(h, w), cmap=plt.get_cmap('gray'))

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Finger print 2")
    ax2.imshow(S_reconstructed[1, :].reshape(h, w), cmap=plt.get_cmap('gray'))

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.set_title("Finger print 3")
    ax2.imshow(S_reconstructed[2, :].reshape(h, w), cmap=plt.get_cmap('gray'))

    plt.show()


draw_reconstructed_S_ICA(X1)
draw_reconstructed_S_ICA(X1, normalized=False)

draw_reconstructed_S_ICA(X2, str_name_A="A2")
draw_reconstructed_S_ICA(X2, str_name_A="A2", normalized=False)
