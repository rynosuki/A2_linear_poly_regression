from operator import indexOf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import lin_reg as lin
import sklearn.model_selection as sk

def main():
  X = np.genfromtxt("GPUbenchmark.csv", delimiter=",")
  y = X[:,6]
  X = X[:,:6]
  M = np.array(np.ones([len(X),1]))
  # X, nxp - k = range(p)
  for m in range(X.shape[1]):
    lowest_cost = []
    for k in range(X.shape[1]):
      train_set = M[:,:m+1]
      train_set = np.append(train_set, X[:,k].reshape(-1,1), axis = 1)
      beta = np.zeros(train_set.shape[1])
      for n in range(40000):
        beta = lin.gradient_lin(train_set, beta, y, 0.000000001)
        # print(lin.cost_lin(train_set, y, beta))
      lowest_cost.append(lin.cost_lin(train_set, y, beta))
    M = np.append(M, X[:,indexOf(lowest_cost, min(lowest_cost))].reshape(-1,1), axis = 1)
  # print(M)

main()