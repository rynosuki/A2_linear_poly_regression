import numpy as np
import matplotlib.pyplot as plt
import lin_reg as lin

X = np.genfromtxt("GPUBenchmark.csv", dtype = np.float32, delimiter=",")
y = X[:,6]
X = X[:,:6]

# E _ 1
Xn = lin.normalize_eq(X,X)

# E _ 2
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.plot(Xn[:,i], y, "bo")
plt.show()

# E _ 3
Xe = np.c_[np.ones((len(Xn[:,0]),1)),Xn[:,0],Xn[:,1],Xn[:,2],Xn[:,3],Xn[:,4],Xn[:,5]].astype(np.float32)
beta = lin.normal_eq(Xe, y)
# test = np.array([1, 2432, 1607, 1683, 8, 8, 256])
# for i in range(len(test)-1):
#   test[i+1] = (test[i+1] - np.mean(X[:,i])) / np.std(X[:,i])
# print(np.mean(test[:-1]), np.std(test[:-1]))
test = np.array([1, 2432, 1607, 1683, 8, 8, 256]).astype(np.float32)

for i in range(len(test)-1):
  test[i+1] = (test[i+1] - np.mean(X[:,i])) / np.std(X[:,i])

print("E - 3", np.dot(test, beta))
# Real result = 114
# Result gotten = 110.80

# E _ 4
J = lin.cost_lin(Xe, y, beta)
print("E - 4", J)
# Cost = 12.39

# E _ 5
beta = [0,0,0,0,0,0,0]
for n in range(20000):
  beta = lin.gradient_lin(Xe, beta, y, 0.0182)
  # print(lin.cost_lin(Xe, beta, y))
  plt.plot(n, lin.cost_lin(Xe, y, beta), "ro")
plt.show()
print("E - 5 Cost =", lin.cost_lin(Xe, y, beta))
print("E - 5 Benchmark =", np.dot(test, beta))
# Cost approx 12.39
# Benchmark predicted to 110.80