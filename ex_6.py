from operator import indexOf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import lin_reg as lin
import sklearn.model_selection as sk

def main():
  X = np.genfromtxt("GPUbenchmark.csv", delimiter=",")
  y = X[:,6]
  X = X[:,:6]
  X = lin.normalize_eq(X,X)
  M = np.array(np.ones([len(X),1]))
  vals = []
  # X, nxp - k = range(p)
  for m in range(X.shape[1]):
    lowest_cost = []
    for k in range(X.shape[1]):
      train_set = M[:,:m+1]
      train_set = np.append(train_set, X[:,k].reshape(-1,1), axis = 1)
      beta = np.zeros(train_set.shape[1])
      for n in range(40000):
        beta = lin.gradient_lin(train_set, beta, y, 0.001)
      lowest_cost.append(lin.cost_lin(train_set, y, beta))
    M = np.append(M, X[:,indexOf(lowest_cost, min(lowest_cost))].reshape(-1,1), axis = 1)
    vals.append(M)
    if m == 0:
      print("Feature", indexOf(lowest_cost, min(lowest_cost)), "is the most important.")
    X = np.delete(X, indexOf(lowest_cost, min(lowest_cost)), axis=1)
  
  print("\nCosts")
  for i in vals:
    lg = LinearRegression().fit(i, y)
    ypred = sk.cross_val_predict(lg, i, y, cv=3)
    print(lin.mse(ypred, y))
  
  print("Model with 5 features is the best as it has the lowest cost.")
main()