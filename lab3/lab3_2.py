import numpy as np
import matplotlib.pyplot as plt

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

# X1

X1 = A1 @ S

fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(1, 3, 1)
ax1.set_title("Finger print 1 - ICA (A1)")
ax1.imshow(X1[0, :].reshape(h, w), cmap=plt.get_cmap('gray'))

ax2 = fig.add_subplot(1, 3, 2)
ax2.set_title("Finger print 2 - ICA (A1)")
ax2.imshow(X1[1, :].reshape(h, w), cmap=plt.get_cmap('gray'))

ax2 = fig.add_subplot(1, 3, 3)
ax2.set_title("Finger print 3 - ICA (A1)")
ax2.imshow(X1[2, :].reshape(h, w), cmap=plt.get_cmap('gray'))

# plt.show()

# X2

X2 = A2 @ S

fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(1, 3, 1)
ax1.set_title("Finger print 1 - ICA (A2)")
ax1.imshow(X2[0, :].reshape(h, w), cmap=plt.get_cmap('gray'))

ax2 = fig.add_subplot(1, 3, 2)
ax2.set_title("Finger print 2 - ICA (A2)")
ax2.imshow(X2[1, :].reshape(h, w), cmap=plt.get_cmap('gray'))

ax2 = fig.add_subplot(1, 3, 3)
ax2.set_title("Finger print 3 - ICA (A2)")
ax2.imshow(X2[2, :].reshape(h, w), cmap=plt.get_cmap('gray'))

plt.show()

