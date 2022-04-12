import numpy as np
import matplotlib.pyplot as plt

def normalize_eq(X):
  mu = np.mean(X)
  sigma = np.std(X)
  return (X - mu) / sigma

def gradient_eq(beta, alpha, X, fb, y):
  return beta - (alpha/len(X))*(X.T.dot(sigmoid_eq(fb) - y))

def sigmoid_eq(z):
  return 1/(1 + np.e**(-z))

def cost_eq(X, y, fb):
  return -(1/len(X))*y.T.dot(np.log(sigmoid_eq(fb)) + (1-y).T.dot(np.log(1-sigmoid_eq(fb))))

def main():
  data = np.genfromtxt("breast_cancer.csv", delimiter=",")
  np.random.shuffle(data)
  data = np.where(data == 2, 0, data)
  data = np.where(data == 4, 1, data)
  training_set = data[:500]
  test_set = data[500:]
  # I allocated approximately 75% for training and 25% for testing.
  
  training_set = normalize_eq(training_set)
  training_set = np.c_[np.ones([len(training_set),1]), training_set]
  test_set = normalize_eq(test_set)
  test_set = np.c_[np.ones([len(test_set),1]), test_set]
  beta = [0,0,0,0,0,0,0,0,0,0]
  
  plt.plot(training_set[:,:10], training_set[:,10], "ro")
  plt.show()
  
  for n in range(1000):
    beta = gradient_eq(beta, 0.1, training_set[:,:10], sigmoid_eq(np.dot(training_set[:,:10], beta)), training_set[:,10])
    plt.scatter(n, cost_eq(training_set[:,:10], beta, sigmoid_eq(beta)), c="blue")
  plt.show()

main()