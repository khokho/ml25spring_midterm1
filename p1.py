import numpy as np

from scipy import stats

# data from out graph:
x = np.array([-9, -7, -5, -3, -1, 1, 3, 5, 7, 9])
print(f"x = {x}")

y = np.array([-8.5, -6.7, -4.8, -3.1, -0.9, 0.8, 2.9, 4.7, 6.5, 8.3])
print(f"y = {y}")

# calculate means
print(f"x.mean() = {x.mean()}")
print(f"y.mean() = {y.mean()}")
# x.mean() = 0.0
# y.mean() = -0.08000000000000007
# This already proves that the it's not perfect correlation(1)

# calculate intermediate vectors
x_sub_mean = x - x.mean()
y_sub_mean = y - y.mean()

# calculate covariance
cov_x_y = (x_sub_mean * y_sub_mean).sum()
# calculate std dev of x and y
dx = np.sqrt((x_sub_mean**2).sum())
dy = np.sqrt((y_sub_mean**2).sum())

# calculate pearson coefficient
pearson_r = cov_x_y / (dx * dy)

print(f'pearson coefficient = {pearson_r}')
# verify it with scipy version of the calculation
scipy_r = stats.pearsonr(x, y)
print(scipy_r)
print(f'pearson coefficient with scipy = {scipy_r}')
assert pearson_r == scipy_r.statistic
