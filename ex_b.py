from matplotlib.colors import ListedColormap
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import lin_reg as lin

def main():
  data = np.genfromtxt("admission.csv", delimiter=",")
  X = data[:,:2]
  y = data[:,2]
  
  Xn = lin.normalize_eq(X)
  # plt.scatter(Xn[:,0], Xn[:,1], c = data[:,2])
  # plt.show()
  
  test_data = np.array([[0,1],[2,3]])
  
  Xe = np.c_[np.ones([len(Xn[:,0]), 1]), Xn[:,0], Xn[:,1]]
  beta = np.array([0,0,0])
  print(lin.cost_eq(Xe, y, beta))
  
  # test_X = np.array([[0,1],[2,3]])
  # print(sigmoid_eq(test_X))

  for n in range(100000):
    beta = lin.gradient_eq(beta, 0.5, Xe, y)
  print(lin.cost_eq(Xe, y, beta))
  print(beta)
  S = np.array([45,85])
  Sn = (S-np.mean(X))/np.std(X)
  Sne = np.c_[1,Sn[0],Sn[1]]
  prob = lin.sigmoid_eq(np.dot(Sne, beta))
  print("Adm. prob. for scores %i, %i is %0.2f" % (S[0],S[1],prob[0]))
  
  p = np.dot(Xe, beta)
  p = lin.sigmoid_eq(p)
  pp = np.round(p)
  yy = y
  print("Training errors:", (np.sum(yy!=pp)))
  
  decision_boundary(Xe, y, beta)

def decision_boundary(X, y, beta):
  x_min, x_max = np.min(X[:,1])-0.1, np.max(X[:,1])+0.1
  y_min, y_max = np.min(X[:,2])-0.1, np.max(X[:,2])+0.1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, .01), np.arange(y_min, y_max, .01))
  x1,x2 = xx.ravel(), yy.ravel()
  XXe = lin.mapFeature(x1,x2,1)

  p = lin.sigmoid_eq(np.dot(XXe, beta))
  classes = p > 0.5
  clz_mesh = classes.reshape(xx.shape)
  
  cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
  cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
  
  test_case = lin.normalize_eq(np.array([45,85]))
  
  plt.figure(2)
  plt.pcolormesh(xx,yy,clz_mesh, cmap=cmap_light)
  plt.scatter(X[:,1], X[:,2],c=y, marker='.', cmap=cmap_bold)
  plt.scatter(test_case[0],test_case[1], c="green")
  plt.show()

if __name__ == "__main__":
  main()