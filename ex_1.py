import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numba import njit

@njit
def normalized(Xi):
  mu = np.mean(Xi)
  omega = np.std(Xi)
  return (Xi - mu) / omega

@njit
def normal_eq(Xe, y):
  return inv(np.dot(Xe.T,Xe)).dot(Xe.T).dot(y)

@njit
def cost_eq(Xe, beta, y):
  return ((np.dot(Xe,beta.astype(Xe.dtype))- y).T.dot(np.dot(Xe,beta.astype(Xe.dtype)) - y))/len(Xe)

def gradient_eq(Xe, beta, y, alpha):
  return beta - np.dot(alpha,Xe.T).dot(np.dot(Xe,beta) - y)

X = np.genfromtxt("GPUBenchmark.csv", dtype = np.float32, delimiter=",")
y = X[:,6]
X = X[:,:6]

# E _ 1
Xn = normalized(X)

# E _ 2
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.plot(Xn[:,i], y, "bo")
# plt.show()

# E _ 3
Xe = np.c_[np.ones((len(X[:,0]),1)),X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5]].astype(np.float32)
beta = normal_eq(Xe, y)
print("E - 3", np.dot([1, 2432, 1607, 1683, 8, 8, 256], beta))
# Real result = 114
# Result gotten = 110.804 with Xe as non normalized vectors

# E _ 4
J = cost_eq(Xe, beta, y)
print("E - 4", J)
# Cost = 12.39

# E _ 5
beta = [0,0,0,0,0,0,0]
for n in range(1000):
  beta = gradient_eq(Xe, beta, y, 0.000000015)
  print(cost_eq(Xe, beta, y))
  plt.plot(n, cost_eq(Xe, beta, y), "ro")
plt.show()
print("E - 5 Cost =",cost_eq(Xe, beta, y))
print("E - 5 Benchmark =", np.dot([1, 2432, 1607, 1683, 8, 8, 256], beta))
# Cost approx 227.218, could go down to 28 with alot more calculations
# Benchmark predicted to 92.64