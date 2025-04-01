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

mnist_train, mnist_test, label_train, label_test = train_test_split(mnist_012points, mnist_labels,
                                                                    test_size=0.5, random_state=42)

PCA_3D = decomposition.PCA(n_components=3)
PCA_3D.fit(mnist_train)
mnist_pca_reduced_3d = PCA_3D.transform(mnist_test)

fig_PCA = plt.figure(1)
ax_PCA = fig_PCA.add_subplot(111, projection='3d')
ax_PCA.set_title("PCA")

for label in range(len(mnist_classes_names)):
    points_PCA = mnist_pca_reduced_3d[label_test == label]
    ax_PCA.scatter(points_PCA[:, 0], points_PCA[:, 1], points_PCA[:, 2], label=mnist_classes_names[label])

ax_PCA.legend()
plt.show()

# B

mnist_train, mnist_test = mnist_train.T, mnist_test.T
d, _ = mnist_train.shape

model = NCA()
model.fit(mnist_train, label_train, 3)
mnist_test_transformed3D = model.transform(mnist_test)

A = torch.rand((3, d))
mnist_test_transformed3D_random = torch.matmul(A, torch.tensor(mnist_test, dtype=torch.float32))

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
