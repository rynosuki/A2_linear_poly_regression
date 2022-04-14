import numpy as np
import matplotlib.pyplot as plt
from numba import njit

def normalize_eq(X):
  mu = np.mean(X)
  sigma = np.std(X)
  return (X - mu) / sigma

@njit
def gradient_eq(beta, alpha, X, y):
  return beta - (alpha / len(X))*(X.T).dot(sigmoid_eq(np.dot(X, beta.astype(np.float64))) - y)

@njit
def sigmoid_eq(z):
  return 1/(1 + np.exp(-z))

@njit
def cost_eq(Xn, y, xb):
  return - (1/len(Xn)*(y.T.dot(np.log(sigmoid_eq(xb))) + (1-y).T.dot(np.log(1-sigmoid_eq(xb)))))

def training_errors(X, beta, y):
  p = np.dot(X, beta).reshape(-1,1)
  p = sigmoid_eq(p)
  pp = np.round(p)
  yy = y.reshape(-1,1)
  return np.sum(yy!=pp)

def main():
  for i in range(10):
    data = np.genfromtxt("breast_cancer.csv", delimiter=",")
    np.random.shuffle(data)
    X = data[:,:9]
    y = data[:,9]
    y = np.where(y == 2, 0, y)
    y = np.where(y == 4, 1, y)
    training_set = X[:500]
    test_set = X[500:]
    training_set_y = y[:500]
    test_set_y = y[500:]
    # I allocated approximately 75% for training and 25% for testing.
    
    training_set = normalize_eq(training_set)
    training_set = np.c_[np.ones([len(training_set),1]), training_set]
    test_set = normalize_eq(test_set)
    test_set = np.c_[np.ones([len(test_set),1]), test_set]
    beta = np.array([0,0,0,0,0,0,0,0,0,0])
    
    # plt.plot(training_set[:,:9], training_set_y, "ro")
    # plt.show()
    
    for n in range(250):
      beta = gradient_eq(beta, 1.5, training_set, training_set_y)
    #   plt.scatter(n, cost_eq(training_set, training_set_y, np.dot(training_set,beta)), edgecolors="red", c = "none")
    # plt.show()
    print("Alpha =", 0.2, "N =", n+1)
    
    print("Round:", i+1)
    print("Training errors:", training_errors(training_set, beta, training_set_y))
    print("Training accuracy =", (len(training_set)-training_errors(training_set, beta, training_set_y))/len(training_set))
    print("Test errors:", training_errors(test_set, beta, test_set_y))
    print("Test accuracy =", (len(test_set)-training_errors(test_set, beta, test_set_y))/len(test_set))
    
    # Repeated runs seem to give a similar qualitatively result. Around 97% accuracy for test. It varies between 96-99% though.

if __name__ == "__main__":
  main()