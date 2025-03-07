import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Q0.3 a

points = np.array([
    [3, 4, 7, 8, 10, 11, 12, 16, 18, 20],  # x-coordinates
    [5, 6, 7, 11, 12, 14, 10, 15, 19, 21]  # y-coordinates
])

plt.scatter(points[0], points[1])
plt.plot(points[0], 0.6 * points[0] + 5, 'red', label = '$y = 0.6x + 5$')
plt.show()

# Q0.3 b

dists = 0.6 * points[0] + 5 - points[1]
# print(dists)

# Q0.3 c

reg_stats = stats.linregress(points[0], points[1])
print(reg_stats.slope, reg_stats.intercept)

plt.scatter(points[0], points[1])
plt.plot(points[0], 0.6 * points[0] + 5, 'red', label='$y = 0.6x + 5$')
plt.plot(points[0], reg_stats.slope * points[0] + reg_stats.intercept, 'green', label="best fit")
plt.show()

# or write formulas of b and a from derivative of LSM
