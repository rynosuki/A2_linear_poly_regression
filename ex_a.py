import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import lin_reg as lin

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
beta = lin.normal_eq(Xe, y)
print(beta)
model = np.dot(Xe, beta)
print(model[2])

#Ex _ 4
x1 = lin.normalize_eq(X[:,1])
x2 = lin.normalize_eq(X[:,2])
Xe = np.c_[np.ones((len(X[:,0]))), x1, x2]

#Ex _ 5
beta = lin.normal_eq(Xe, y)
model = Xe.dot(beta)
print(model[2])
#Ex _ 6
# print(lin.cost_lin(Xe, beta, y))

#Ex _ 7
beta = [18.50, 0.303, 0.388]
for n in range(30):
  beta = lin.gradient_lin(Xe, beta, y, 0.002)
  plt.plot(n, lin.cost_lin(Xe, beta, y), "ro")
# print(lin.cost_lin(Xe, beta, y))
plt.show()
# print("Suitable alpha and N would be alpha = 0.002 and N = 30")
# print(Xe.dot(beta)[2])