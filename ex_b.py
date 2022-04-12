from matplotlib.colors import ListedColormap
import numpy as np
from numba import njit
import matplotlib.pyplot as plt


def main():
  data = np.genfromtxt("admission.csv", delimiter=",")
  X = data[:,:2]
  y = data[:,2]
  
  Xn = normalize_eq(X)
  # plt.scatter(Xn[:,0], Xn[:,1], c = data[:,2])
  # plt.show()
  
  test_data = np.array([[0,1],[2,3]])
  
  Xe = np.c_[np.ones([len(Xn[:,0]), 1]), Xn[:,0], Xn[:,1]]
  beta = np.array([0,0,0])
  assy = np.dot(Xe, beta)
  print(cost_eq(y, assy))
    
  for n in range(2000):
    beta = gradient_eq(beta, Xe, 0.5, np.dot(Xe,beta), y)
  print(beta)
  
  S = np.array([45,85])
  Sn = (S-np.mean(S))/np.std(S)
  Sne = np.c_[1,Sn[0],Sn[1]]
  prob = sigmoid_eq(np.dot(Sne, beta))
  print("Adm. prob. for scores %i, %i is %0.2f" % (S[0],S[1],prob[0]))
  
  p = np.dot(Xe, beta).reshape(-1,1)
  p = sigmoid_eq(p)
  pp = np.round(p)
  yy = y.reshape(-1,1)
  print("Training errors:", (np.sum(yy!=pp)))
  
  decision_boundary(Xe, y, beta)

def decision_boundary(X, y, beta):
  x_min, x_max = np.min(X[:,1])-0.1, np.max(X[:,1])+0.1
  y_min, y_max = np.min(X[:,2])-0.1, np.max(X[:,2])+0.1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min, y_max, .01))
  x1,x2 = xx.ravel(), yy.ravel()
  XXe = mapFeature(x1,x2,1)

  p = sigmoid_eq(np.dot(XXe, beta))
  classes = p > 0.5
  clz_mesh = classes.reshape(xx.shape)
  
  cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
  cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
  
  test_case = normalize_eq(np.array([45,85]))
  
  plt.figure(2)
  plt.pcolormesh(xx,yy,clz_mesh, cmap=cmap_light)
  plt.scatter(X[:,1], X[:,2],c=y, marker='.', cmap=cmap_bold)
  plt.scatter(test_case[0],test_case[1], c="green")
  plt.show()
  
def mapFeature(X1,X2,D): # Pyton
  one = np.ones([len(X1),1])
  Xe = np.c_[one,X1,X2] # Start with [1,X1,X2]
  for i in range(2,D+1):
    for j in range(0,i+1):
      Xnew = X1**(i-j)*X2**j # type (N)
      Xnew = Xnew.reshape(-1,1) # type (N,1) required by append
      Xe = np.append(Xe,Xnew,1) # axis = 1 ==> append column
  return Xe

def normalize_eq(X):
  mu = np.mean(X)
  sigma = np.std(X)
  return (X - mu) / sigma

def sigmoid_eq(z):
  return 1/(1 + np.e**(-z))

def gradient_eq(beta, X, alpha, sig, y):
  return beta - (alpha/len(X))*(X.T.dot(sigmoid_eq(sig) - y))

def cost_eq(y, Xn):
  return -(1/len(Xn))*((y.T.dot(np.log(sigmoid_eq(Xn)))) + (1-y).T.dot(np.log(1-sigmoid_eq(Xn))))

if __name__ == "__main__":
  main()