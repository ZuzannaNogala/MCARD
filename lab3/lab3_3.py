import numpy as np
from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt

# READ DATA
lfw_dataset = datasets.fetch_lfw_people()
n_all, _, _ = lfw_dataset.images.shape

np.random.seed(0)
idx = np.random.randint(1, n_all, 4)

# CHOOSE 4 RANDOM PEOPLE
lfw_images = lfw_dataset.images[idx]
lfw_data_classes = lfw_dataset.target[idx]  # the person IDs
lfw_classes_names = lfw_dataset.target_names

n, h, w = lfw_images.shape
d = h * w

# CHOSEN FACES

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

b = [3, 2, 1, 0]

for row in ax:
    for col in row:
        i = b.pop()
        col.imshow(lfw_images[i], cmap='gray')
        col.set_title(lfw_classes_names[i])

plt.show()


# A

def r_component_explain_90perc(image):
    pca_model = decomposition.PCA()
    pca_model.fit(image)

    eigenvalues_model = pca_model.explained_variance_
    total_variance = np.sum(eigenvalues_model)
    cumulative_variance = np.cumsum(eigenvalues_model) / total_variance

    print(f"The minimal number of components to explain 90% "
          f"of data set is {np.min(np.where(cumulative_variance >= 0.9))}")


for i in lfw_images:
    r_component_explain_90perc(i)


def draw_reconstructed_image_from_PCA(id_person, row):
    j = np.where(lfw_data_classes == id_person)[0][0]
    idxs = [8, 6, 5, 4, 2, 1]

    for col in row:
        r = idxs.pop()
        pca_model = decomposition.PCA(n_components=r)
        pca_fitted = pca_model.fit_transform(lfw_images[j])
        pca_inv_fit = pca_model.inverse_transform(pca_fitted)  # back to the original (or close to the original)
        # # high-dimensional space

        col.imshow(pca_inv_fit, cmap=plt.get_cmap('gray'))
        col.set_title(f"r = {r}")
        col.axis('off')


fig, ax = plt.subplots(nrows=4, ncols=6, sharey=True)

for k in range(n):
    id_person = lfw_data_classes[k]
    draw_reconstructed_image_from_PCA(id_person, ax[k])

plt.show()


# B

all_idx = np.ones(len(lfw_dataset.images), dtype=bool)
all_idx[idx] = False

lfw_images_all = lfw_dataset.images[all_idx]
lfw_targets_all = lfw_dataset.target[all_idx]

collected_images = np.array([lfw_dataset.images[i].flatten() for i in range(n_all - n)])
print(collected_images.shape)


def draw_reconstructed_image_from_PCA_all(id_person, row):
    j = np.where(lfw_data_classes == id_person)[0][0]
    idxs = [120, 100, 8, 6, 4, 2, 1]

    for col in row:
        r = idxs.pop()
        pca_model = decomposition.PCA(n_components=r)
        pca_model.fit(collected_images)
        img_flat = lfw_images[j].flatten().reshape(1, -1)
        img_pca = pca_model.transform(img_flat)
        pca_fitted = pca_model.inverse_transform(img_pca) # back to the original (or close to the original)
        # high-dimensional space

        col.imshow(pca_fitted.reshape(h, w), cmap=plt.get_cmap('gray'), aspect='equal')
        col.set_title(f"r = {r}")
        col.axis('off')


fig, ax = plt.subplots(nrows=4, ncols=6, sharey=True)

for k in range(n):
    id_person = lfw_data_classes[k]
    draw_reconstructed_image_from_PCA_all(id_person, ax[k])

plt.show()
