import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from ex_3 import cost_eq, sigmoid_eq, gradient_eq, training_errors

def main():
  data = np.genfromtxt("microchips.csv", delimiter=",")
  plt.scatter(data[:,0], data[:,1], c = data[:,2])
  plt.show()
  print(data[:,0], "\n", data[:,1])
  print(np.dot(np.dot(data[:,0], data[:,1]),[1,2,3,4,5,6]))
  # X = np.c_[np.zeros([len(data[:,0]),1]), data[:,0], data[:,1], data[:,0]**2, np.matmul(data[:,0], data[:,1]), data[:,1]**2]
  # print(X)

main()