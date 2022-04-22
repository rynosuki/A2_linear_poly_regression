from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score
import numpy as np
import lin_reg as lin
import matplotlib.pyplot as plt

def main():
  data = np.genfromtxt("microchips.csv", delimiter=",")
  X = data[:,:2]
  y = data[:,2]
  
  fig, ((ax1,ax2,ax3), (ax4, ax5, ax6), (ax7,ax8,ax9)) = plt.subplots(3,3)
  fig.suptitle("C=10000")
  axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
  for i in range(9):
    Xe = lin.mapFeature(X[:,0], X[:,1], i+1, Ones=False)
    lg = LogisticRegression(C=10000, max_iter=1000)
    lg.fit(Xe,y)
    ypred = lg.predict(Xe)
    # p = lg.predict_proba(Xe)
    print(np.sum(ypred != y))
    lin.plot_grid_sklearn(Xe[:,0], Xe[:,1], ypred, axes[i], lg, i+1)
  
  big, ((bx1,bx2,bx3), (bx4, bx5, bx6), (bx7,bx8,bx9)) = plt.subplots(3,3)
  big.suptitle("C=1")
  axes = [bx1,bx2,bx3,bx4,bx5,bx6,bx7,bx8,bx9]
  for i in range(9):
    Xe = lin.mapFeature(X[:,0], X[:,1], i+1, Ones=False)
    lg = LogisticRegression(C=1, max_iter=1000)
    lg.fit(Xe,y)
    ypred = lg.predict(Xe)
    # p = lg.predict_proba(Xe)
    print(np.sum(ypred != y))
    lin.plot_grid_sklearn(Xe[:,0], Xe[:,1], ypred, axes[i], lg, i+1)
  plt.show()
  
  print(cross_val_score(lg, X, y, cv=3))
  print(cross_val_predict(lg, X, y, cv=3))
    
main()