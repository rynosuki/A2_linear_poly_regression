from sklearn.linear_model import LogisticRegression
import numpy as np
import lin_reg as lin
import matplotlib.pyplot as plt

def main():
  data = np.genfromtxt("microchips.csv", delimiter=",")
  X = data[:,:2]
  y = data[:,2]
  
  fig, ((ax1,ax2,ax3), (ax4, ax5, ax6), (ax7,ax8,ax9)) = plt.subplots(3,3)
  
  for i in range(9):
    Xe = lin.mapFeature(X[:,0], X[:,1], i+1)
    plt.figure(i+1)
    lg = LogisticRegression(C=1, max_iter=1000)
    lg.fit(Xe,y)
    ypred = lg.predict(Xe)
    print(np.sum(ypred != y))
    # lin.plot_grid()

  # plt.show()
  
def calc_gradient(X, y, beta, n, alpha):
  values = []
  for i in range(n):
    beta = lin.gradient_log(beta, alpha, X, y)
    values.append(lin.cost_log(X,y,beta))
  plt.plot(range(n), values, "ro")
  return beta

main()