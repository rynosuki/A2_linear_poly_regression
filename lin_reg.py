from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

@njit
def cost_log(X, y, beta):
  return - (1/len(X)*(y.T.dot(np.log(sigmoid_log(np.dot(X, beta.astype(np.float64))))) + (1-y).T.dot(np.log(1-sigmoid_log(np.dot(X, beta.astype(np.float64)))))))

@njit
def cost_lin(X, beta, y):
  return ((np.dot(X,beta.astype(X.dtype))- y).T.dot(np.dot(X,beta.astype(X.dtype)) - y))/len(X)

@njit
def gradient_log(beta, alpha, X, y):
  return beta - (alpha / len(X))*(X.T).dot(sigmoid_log(np.dot(X, beta.astype(np.float64))) - y)

@njit
def gradient_lin(beta, alpha, X, y):
  return beta.astype(X.dtype) - np.dot(alpha,X.T).dot(np.dot(X,beta.astype(X.dtype)) - y)

@njit
def sigmoid_log(z):
  return 1/(1 + np.exp(-z))

def normalize_eq(X, x = None):
  if x == None:
    x = X
  mu = np.mean(x)
  sigma = np.std(x)
  return (X - mu) / sigma

def mse(pred_y, y):
  return ((pred_y - y)**2).mean()

@njit
def normal_eq(X, y):
  return np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y)

def training_errors(X, beta, y):
  p = np.dot(X, beta).reshape(-1,1)
  p = sigmoid_log(p)
  pp = np.round(p)
  yy = y.reshape(-1,1)
  return np.sum(yy!=pp)

def mapFeature(X1,X2,D): # Pyton
  one = np.ones([len(X1),1])
  Xe = np.c_[one,X1,X2] # Start with [1,X1,X2]
  for i in range(2,D+1):
    for j in range(0,i+1):
      Xnew = X1**(i-j)*X2**j # type (N)
      Xnew = Xnew.reshape(-1,1) # type (N,1) required by append
      Xe = np.append(Xe,Xnew,1) # axis = 1 ==> append column
  return Xe

def plot_grid(X1, X2, beta, y):
  min_x, max_x = min(X1), max(X1)
  min_y, max_y = min(X2), max(X2)
  grid_size = 200
  x_axis = np.linspace(min_x, max_x, grid_size)
  y_axis = np.linspace(min_y, max_y, grid_size)
  
  xx, yy = np.meshgrid(x_axis, y_axis)
  x1, x2 = xx.ravel(), yy.ravel()
  XXe = mapFeature(x1, x2, 2)
  
  p = sigmoid_log(np.dot(XXe, beta))
  classes = p > 0.5
  clz_mesh = classes.reshape(xx.shape)
  
  cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
  cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
  
  plt.figure(1)
  plt.pcolormesh(xx,yy,clz_mesh, cmap=cmap_light)
  plt.scatter(X1, X2,c=y, marker='.', cmap=cmap_bold)
  plt.show()