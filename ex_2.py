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

@njit
def mse(pred_y, y):
  return ((pred_y - y)**2).mean()

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
  beta = normal_eq(Xe.astype(np.float32), y)
  values.append(np.dot(Xe.astype(np.float32),beta))
  
for i in range(len(values)):
  plt.subplot(2,2,i+1)
  plt.scatter(X, y, edgecolors="red", c="none")
  plt.plot(X, values[i])
plt.show()

for i in values:
  mses.append(mse(i, y))
# Polynomial 4 chosen due to the lowest MSE
# Values calculated were:
# 2896.09375
# 585.0675048828125
# 454.5433349609375
# 444.0847473144531
# Respectively per polynomial.

# Ex _ 3
beta = normal_eq(polys[3].astype(np.float32), y)
print(np.dot([1, 47, 47**2, 47**3, 47**4],beta))
# He bought at index 568 and would now be worth 797
# print(2.3e+6*(798/568))
# The house would be worth 3.231 million SEK in 2022


#### Calculations with gradient.
values = []
Xn = normalized(X)
print("Normalized mean:", np.mean(Xn), "Non-normalized: ",np.mean(X))
print("Normalized STD:", np.std(Xn), "Non-normalized:", np.std(X))
polys = [np.c_[np.ones((len(Xn), 1)), Xn],np.c_[np.ones((len(Xn), 1)), Xn**1, Xn**2], 
          np.c_[np.ones((len(Xn), 1)), Xn**1, Xn**2, Xn**3], np.c_[np.ones((len(Xn), 1)), Xn**1, Xn**2, Xn**3, Xn**4]]
x3 = np.c_[np.ones((len(X), 1)), X**1, X**2, X**3, X**4]
beta = np.zeros(polys[0].shape[1])
for n in range(50):
  beta = gradient_eq(polys[0], beta, y, 0.003)
  values.append(cost_eq(polys[0], beta, y))
beta = np.zeros(polys[1].shape[1])
for n in range(50):
  beta = gradient_eq(polys[1], beta, y, 0.015)
  values.append(cost_eq(polys[1], beta, y))
beta = np.zeros(polys[2].shape[1])
for n in range(250):
  beta = gradient_eq(polys[2], beta, y, 0.0095)
  values.append(cost_eq(polys[2], beta, y))
beta = np.zeros(polys[3].shape[1])
for n in range(250):
  beta = gradient_eq(polys[3], beta, y, 0.0041)
  values.append(cost_eq(polys[3], beta, y))
  
print("For d:1 alpha =", 0.003, "with n of", 50, "value:", values[0])
print("For d:2 alpha =", 0.0015, "with n of", 50, "value:", values[1])
print("For d:3 alpha =", 0.0095, "with n of", 250, "value:", values[2])
print("For d:4 alpha =", 0.0041, "with n of", 250, "value:", values[3])