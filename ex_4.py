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
  alpha = 5
  beta = calc_gradient(X, y, beta, 10000, alpha, ax1)
  
  lin.plot_grid(X1, X2, beta, y, 2, ax2)
  plt.show()
  print("alpha:", alpha, "N:", 10000)
  print("Training errors:", lin.training_errors(X, beta, y))
  
  X = lin.mapFeature(X1, X2, 5)
  beta = np.array(np.zeros(X.shape[1]))
  
  fig, (ax1,ax2) = plt.subplots(1,2)
  alpha = 15
  beta = calc_gradient(X, y, beta, 100000, alpha, ax1)
  lin.plot_grid(X1, X2, beta, y, 5, ax2)
  plt.show()
  print("alpha:", alpha, "N:", 100000)
  print("Training errors:", lin.training_errors(X, beta, y))

def calc_gradient(X, y, beta, n, alpha, plot):
  values = []
  for i in range(n):
    beta = lin.gradient_log(X, beta, y, alpha)
    values.append(lin.cost_log(X,y,beta))
  plot.plot(range(n), values, "ro")
  return beta

main()