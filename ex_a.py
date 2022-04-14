import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def normalized(Xi):
  mu = np.mean(Xi)
  omega = np.std(Xi)
  return (Xi - mu) / omega

def normal_eq(Xe, y):
  return inv(np.dot(Xe.T,Xe)).dot(Xe.T).dot(y)

def cost_eq(Xe, beta, y):
  return ((np.dot(Xe,beta)- y).T.dot(np.dot(Xe,beta) - y))/len(Xe)

def gradient_eq(Xe, beta, y, alpha):
  return beta - np.dot(alpha,Xe.T).dot(np.dot(Xe,beta) - y)

X = np.genfromtxt("girls_height.csv", dtype = np.float64, delimiter="	")

#Ex _ 1
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(X[:,2], X[:,0], "go")
ax2.plot(X[:,1], X[:,0], "go")
plt.show()

#Ex _ 2
Xe = np.c_[np.ones((len(X[:,0]),1)), X[:,1], X[:,2]]
y = X[:,0]
# print(y.shape)

#Ex _ 3
beta = normal_eq(Xe, y)
print(beta)
model = np.dot(Xe, beta)
print(model[2])

#Ex _ 4
x1 = normalized(X[:,1])
x2 = normalized(X[:,2])
Xe = np.c_[np.ones((len(X[:,0]))), x1, x2]

#Ex _ 5
beta = normal_eq(Xe, y)
model = Xe.dot(beta)
print(model[2])
#Ex _ 6
# print(cost_eq(Xe, beta, y))

#Ex _ 7
beta = [18.50, 0.303, 0.388]
for n in range(30):
  beta = gradient_eq(Xe, beta, y, 0.002)
  plt.plot(n, cost_eq(Xe, beta, y), "ro")
# print(cost_eq(Xe, beta, y))
plt.show()
# print("Suitable alpha and N would be alpha = 0.002 and N = 30")
# print(Xe.dot(beta)[2])