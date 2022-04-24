import numpy as np
import matplotlib.pyplot as plt
import lin_reg as lin

X = np.genfromtxt("housing_price_index.csv", dtype = np.float32, delimiter=",")

y = X[:,1]
X = X[:,0]
plt.plot(X,y,"ro")
plt.show()

polys = [np.c_[np.ones((len(X), 1)), X],np.c_[np.ones((len(X), 1)), X**1, X**2], 
          np.c_[np.ones((len(X), 1)), X**1, X**2, X**3], np.c_[np.ones((len(X), 1)), X**1, X**2, X**3, X**4]]
values = []
mses = []

for Xe in polys:
  beta = lin.normal_eq(Xe.astype(np.float32), y)
  values.append(np.dot(Xe.astype(np.float32),beta))
  
for i in range(len(values)):
  plt.subplot(2,2,i+1)
  plt.scatter(X, y, edgecolors="red", c="none")
  plt.plot(X, values[i])
plt.show()

for i in values:
  mses.append(lin.mse(i, y))
# Polynomial 4 chosen due to the lowest MSE
# Values calculated were:
# 2896.09375
# 585.0675048828125
# 454.5433349609375
# 444.0847473144531
# Respectively per polynomial.

# Ex _ 3
beta = lin.normal_eq(polys[3].astype(np.float32), y)
print(np.dot([1, 47, 47**2, 47**3, 47**4],beta))
# He bought at index 568 and would now be worth 797
# print(2.3e+6*(798/568))
# The house would be worth 3.231 million SEK in 2022