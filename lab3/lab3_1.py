from sklearn import datasets, decomposition
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from models import NCA
import torch
from sklearn.datasets import make_circles

# A

mnist = datasets.load_digits()

mnist_012classes_pos = np.isin(mnist.target, [0, 1, 2])

mnist_012points = mnist.data[mnist_012classes_pos]
mnist_labels = mnist.target[mnist_012classes_pos]
mnist_classes_names = [0, 1, 2]

PCA_3D = decomposition.PCA(n_components=3)
PCA_3D.fit(mnist_012points)
mnist_pca_reduced_3d = PCA_3D.transform(mnist_012points)

fig_PCA = plt.figure(1)
ax_PCA = fig_PCA.add_subplot(111, projection='3d')
ax_PCA.set_title("PCA")

for label in range(len(mnist_classes_names)):
    points_PCA = mnist_pca_reduced_3d[mnist_labels == label]
    ax_PCA.scatter(points_PCA[:, 0], points_PCA[:, 1], points_PCA[:, 2], label=mnist_classes_names[label])

ax_PCA.legend()
plt.show()

# B
mnist_train, mnist_test, label_train, label_test = train_test_split(mnist_012points, mnist_labels,
                                                                    test_size=0.5, random_state=42)
_, d = mnist_train.shape

# PCA
PCA_3D_2 = decomposition.PCA(n_components=3)
PCA_3D_2.fit(mnist_train)
PCA_3D_2.fit(mnist_train)
mnist_pca_reduced_3d = PCA_3D_2.transform(mnist_test)

# NCA
model = NCA()
model.fit(mnist_train.T, label_train, 3)
mnist_test_transformed3D = model.transform(mnist_test.T)

# NCA random
A = torch.rand((3, d))
mnist_test_transformed3D_random = torch.matmul(A, torch.tensor(mnist_test.T, dtype=torch.float32))


fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(1, 3, 2, projection='3d')
ax1.set_title("NCA with optimized A")

ax2 = fig.add_subplot(1, 3, 3, projection='3d')
ax2.set_title("PCA")

ax3 = fig.add_subplot(1, 3, 1, projection='3d')
ax3.set_title("NCA with random A")


for label in mnist_classes_names:
    points_NCA_opt = mnist_test_transformed3D.t()[label_test == label].detach().numpy()
    ax1.scatter(points_NCA_opt[:, 0], points_NCA_opt[:, 1], points_NCA_opt[:, 2], label=label)

    points_PCA = mnist_pca_reduced_3d[label_test == label]
    ax2.scatter(points_PCA[:, 0], points_PCA[:, 1], points_PCA[:, 2], label=label)

    points_NCA_rand = mnist_test_transformed3D_random.t()[label_test == label].detach().numpy()
    ax3.scatter(points_NCA_rand[:, 0], points_NCA_rand[:, 1], points_NCA_rand[:, 2], label=label)


plt.legend()
plt.show()

# C

X_circles, y_circles = make_circles(n_samples=1000, factor=0.3, noise=0.05, random_state=0)

plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, s=1.5)
plt.title("Circles")
plt.show()

Z = np.exp((X_circles[:, 0] ** 2 + X_circles[:, 1] ** 2) / 1.25)
X_prim = np.array([X_circles[:, 0], X_circles[:, 1], Z]).T

# PCA 1D on X_prim
PCA_prim_1d = decomposition.PCA(n_components=1)
PCA_prim_1d_fitted = PCA_prim_1d.fit_transform(X_prim)

plt.figure(figsize=(6.6, 1))
plt.scatter(PCA_prim_1d_fitted, np.zeros(PCA_prim_1d_fitted.shape[0]), c=y_circles, s=1.5, alpha=0.4)
plt.title("1D PCA - X_prim")
plt.show()

# PCA 2D  on X_prim
PCA_prim_2d = decomposition.PCA(n_components=2)
PCA_prim_2d_fitted = PCA_prim_2d.fit_transform(X_prim)

plt.scatter(PCA_prim_2d_fitted[:, 0], PCA_prim_2d_fitted[:, 1], c=y_circles, s=1.5)
plt.title("2D PCA - X_prim")
plt.show()

# PCA 2D on original X_circles
PCA_original_2d = decomposition.PCA(n_components=2)
PCA_2d_fitted = PCA_original_2d.fit_transform(X_circles)


# kernelPCA 2D on original X_circles
kernelPCA_original_2d = decomposition.KernelPCA(kernel="rbf", n_components=2)  # Radial
kernelPCA_2d_fitted = kernelPCA_original_2d.fit_transform(X_circles)


fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(1, 3, 1)
ax1.set_title("Original Circle")
ax1.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, alpha=0.7, s=5)

ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(PCA_2d_fitted[:, 0], PCA_2d_fitted[:, 1], c=y_circles, alpha=0.7, s=5)
ax2.set_title("2D PCA")

ax3 = fig.add_subplot(1, 3, 3)
ax3.set_title("2D KernelPCA - radian kernel")
ax3.scatter(kernelPCA_2d_fitted[:, 0], kernelPCA_2d_fitted[:, 1], c=y_circles, alpha=0.7, s=5)

plt.show()
