import numpy as np
import matplotlib.pyplot as plt
import lin_reg as lin

X = np.genfromtxt("GPUBenchmark.csv", dtype = np.float32, delimiter=",")
y = X[:,6]
X = X[:,:6]

# E _ 1
Xn = lin.normalize_eq(X)

# E _ 2
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.plot(Xn[:,i], y, "bo")
plt.show()

# E _ 3
Xe = np.c_[np.ones((len(Xn[:,0]),1)),Xn[:,0],Xn[:,1],Xn[:,2],Xn[:,3],Xn[:,4],Xn[:,5]].astype(np.float32)
beta = lin.normal_eq(Xe, y)
# non_normalized = np.array([1, 2432, 1607, 1683, 8, 8, 256]).reshape(-1,1)
# normal_test = lin.normalize_eq(non_normalized)
normal_test = (np.array([1, 2432, 1607, 1683, 8, 8, 256]) - np.mean(Xn)) / np.std(Xn)
print("E - 3", np.dot(normal_test, beta))
# Real result = 114
# Result gotten = 3125804 with normalized Xe

# E _ 4
J = lin.cost_lin(Xe, beta, y)
print("E - 4", J)
# Cost = 41215

# E _ 5
beta = [0,0,0,0,0,0,0]
for n in range(100000):
  beta = lin.gradient_lin(Xe, beta, y, 0.0182)
  # print(lin.cost_lin(Xe, beta, y))
  plt.plot(n, lin.cost_lin(Xe, beta, y), "ro")
plt.show()
print("E - 5 Cost =", lin.cost_lin(Xe, beta, y))
print("E - 5 Benchmark =", np.dot(normal_test, beta))
# Cost approx 27.8983
# Benchmark predicted to 142283