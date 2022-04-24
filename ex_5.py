from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import numpy as np
import lin_reg as lin
import matplotlib.pyplot as plt

def main():
  data = np.genfromtxt("microchips.csv", delimiter=",")
  X = data[:,:2]
  y = data[:,2]
  
  # Plots inside of prediction function.
  # 10000 = lower regularization, 1 = higher (1/c)
  errors_unreg = prediction(X, y, 10000)
  errors_reg = prediction(X, y, 1)
  plt.show()
  
  xx = np.linspace(1,9, 9)
  plt.plot(xx,errors_reg, "r")
  plt.plot(xx,errors_unreg, "b")
  plt.legend(["Regularized", "Unregularized"])
  plt.show()
  
def prediction(X, y, c):
  fig, ((ax1,ax2,ax3), (ax4, ax5, ax6), (ax7,ax8,ax9)) = plt.subplots(3,3)
  subtitle = "C=" + str(c)
  fig.suptitle(subtitle)
  axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
  errors_unreg = []
  for i in range(9):
    Xe = lin.mapFeature(X[:,0], X[:,1], i+1, Ones=False)
    lg = LogisticRegression(C=c, max_iter=1000)
    lg.fit(Xe,y)
    ypred = lg.predict(Xe)
    y_predsk = cross_val_predict(lg,Xe,y)
    errors_unreg.append(np.sum(y_predsk != y))
    lin.plot_grid_sklearn(Xe[:,0], Xe[:,1], ypred, axes[i], lg, i+1)
  return errors_unreg
main()