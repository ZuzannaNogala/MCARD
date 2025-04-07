import numpy as np
from sklearn import datasets

lfw_dataset = datasets.fetch_lfw_people()
idx = np.random.randint(1, 10270, 4)

lfw_images = lfw_dataset.images[idx]
lfw_data_classes = lfw_dataset.target[idx]  # the person IDs
lfw_classes_names = lfw_dataset.target_names



