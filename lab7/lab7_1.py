from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

from gensim.models import KeyedVectors

import numpy as np
import matplotlib.pyplot as plt

# LOAD TRAINING DATA (20newsgroups)
newsgroups_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(newsgroups_data.data, newsgroups_data.target,
                                                    test_size=0.25, random_state=42)
label_names = newsgroups_data.target_names

# CONVERT TEXTS TO COUNT VECTORS (Bag of Words)
newsgroups_BoW_vect = CountVectorizer(max_features=5000)  # stop_words='english'
X_train_BoW = newsgroups_BoW_vect.fit_transform(X_train)


# LOAD GOOGLE PRETRAIN MODEL
file_path = '/Users/zuza/GoogleNews-vectors-negative300-SLIM.bin'
w2v_google = KeyedVectors.load_word2vec_format(file_path, binary=True)


# CONVERT TEXTS TO COUNT VECTORS (Word2Vec average)
# from one sentence, where each word has assigned the value, we compute
# the average across all words
# Now, to each sentence we have assigned fixed-size numeric vector

def avg_w2v(texts, model, dim=300):
    vecs = []
    for doc in texts:
        words = [w for w in doc.lower().split() if w in model]
        if words:
            vec = np.mean([model[w] for w in words], axis=0)
        else:
            vec = np.zeros(dim)
        vecs.append(vec)
    return np.array(vecs)


X_train_w2v = avg_w2v(X_train, w2v_google)

# PCA 2D
# Bag of Words
pca_2d_BoW = PCA(n_components=2)
X_train_BoW_2d = pca_2d_BoW.fit_transform(X_train_BoW.toarray())

# Word2Vec average
pca_2d_w2v = PCA(n_components=2)
X_train_w2v_2d = pca_2d_w2v.fit_transform(X_train_w2v)

# PCA 3D
# Bag of Words
pca_3d_BoW = PCA(n_components=3)
X_train_BoW_3d = pca_3d_BoW.fit_transform(X_train_BoW.toarray())

# Word2Vec average
pca_3d_w2v = PCA(n_components=3)
X_train_w2v_3d = pca_3d_w2v.fit_transform(X_train_w2v)

# TSNE
# tsne_BoW = TSNE(n_components=2)
# X_train_tsne = tsne_BoW.fit_transform(X_train_BoW.toarray())

# Plotting 2D

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
f.suptitle("PCA (2D)", fontsize=16)

ax1.set_title("BagOfWords")
ax2.set_title("Word2Vec average")

for label in np.unique(y_train):
    idx = np.where(y_train == label)
    ax1.scatter(X_train_BoW_2d[idx, 0], X_train_BoW_2d[idx, 1], label=label_names[label], s=5)
    ax2.scatter(X_train_w2v_2d[idx, 0], X_train_w2v_2d[idx, 1], label=label_names[label], s=5)

ax1.legend(prop={"size": 5})
ax2.legend(prop={"size": 5})
plt.show()

# Plotting 3D

fig = plt.figure(figsize=(10, 8))
fig.suptitle("PCA (3D)", fontsize=16)

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.set_title("BagOfWords")
ax2.set_title("Word2Vec average")

for label in np.unique(y_train):
    idx = np.where(y_train == label)
    ax1.scatter(X_train_BoW_3d[idx, 0], X_train_BoW_3d[idx, 1], X_train_BoW_3d[idx, 2], label=label_names[label])
    ax2.scatter(X_train_w2v_3d[idx, 0], X_train_w2v_3d[idx, 1], X_train_w2v_3d[idx, 2], label=label_names[label])

ax1.legend(prop={"size": 5})
# ax2.legend(prop={"size": 5})
plt.show()
