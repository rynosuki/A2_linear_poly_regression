import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import lin_reg as lin

def main():
  data = np.genfromtxt("microchips.csv", delimiter=",")
  X1 = data[:,0]
  X2 = data[:,1]
  y = data[:,2]
  # plt.scatter(X1, X2, c = y)
  # plt.show()
  X = lin.mapFeature(X1, X2, 2)
  beta = np.array([0,0,0,0,0,0])
  
  fig, (ax1,ax2) = plt.subplots(1,2)
  
  # for n in range(10000):
  #   beta = lin.gradient_log(beta, 5, X, y)
  #   ax1.scatter(n, lin.cost_log(X, y, beta), c = "none", edgecolors="red")
  
  # lin.plot_grid(X1, X2, beta, y)
  # print(beta)
  
  # print("Training errors:", lin.training_errors(X, beta, y))
  
  X = lin.mapFeature(X1, X2, 5)
  print(X)
  beta = np.array(np.zeros(X.shape[1]))
  print(beta)
  beta = calc_gradient(X, y, beta, 10000)

def calc_gradient(X, y, beta, n):
  for i in range(n):
    beta = lin.gradient_log(beta, 10, X, y)
    plt.scatter(i, lin.cost_log(X, y, beta), c="red")
    # print(lin.cost_log(X, y, beta))
  plt.show()

main()