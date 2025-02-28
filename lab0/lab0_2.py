import numpy as np

# Q0.2 (a)

np.random.seed(0)
d = 100
v = np.random.uniform(-10, 10, size=d)
w = np.random.uniform(-5, 5, size=d)
u = np.random.uniform(0, 10, size=d)

lengths = [vec.shape[0] for vec in [v, w, u]]
weighted_avg = np.average(v, weights= w)
dot_prod_vu = np.dot(v, u)

print(lengths, weighted_avg, dot_prod_vu)

# Q0.2 (b)
n = 200
X = np.random.normal(0, 1, size=(d, n))

col_weighted_avg = np.apply_along_axis(np.average, 0, X, weights=w)
col_lengths = np.apply_along_axis(len, 0, X)

print(col_weighted_avg, col_lengths)