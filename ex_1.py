import numpy as np
import matplotlib.pyplot as plt
import lin_reg as lin

X = np.genfromtxt("GPUBenchmark.csv", dtype = np.float32, delimiter=",")
y = X[:,6]
X = X[:,:6]

# E _ 1
Xn = lin.normalize_eq(X)

# E _ 2
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.plot(Xn[:,i], y, "bo")
# plt.show()

# E _ 3
Xe = np.c_[np.ones((len(X[:,0]),1)),X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5]].astype(np.float32)
beta = lin.normal_eq(Xe, y)
print("E - 3", np.dot([1, 2432, 1607, 1683, 8, 8, 256], beta))
# Real result = 114
# Result gotten = 110.804 with Xe as non normalized vectors

# E _ 4
J = lin.cost_lin(Xe, beta, y)
print("E - 4", J)
# Cost = 12.39

# E _ 5
beta = [0,0,0,0,0,0,0]
for n in range(1000):
  beta = lin.gradient_lin(Xe, beta, y, 0.000000015)
  print(lin.cost_lin(Xe, beta, y))
  plt.plot(n, lin.cost_lin(Xe, beta, y), "ro")
plt.show()
print("E - 5 Cost =", lin.cost_lin(Xe, beta, y))
print("E - 5 Benchmark =", np.dot([1, 2432, 1607, 1683, 8, 8, 256], beta))
# Cost approx 227.218, could go down to 28 with alot more calculations
# Benchmark predicted to 92.64