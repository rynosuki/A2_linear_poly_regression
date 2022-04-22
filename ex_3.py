import numpy as np
import matplotlib.pyplot as plt
import lin_reg as lin

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
    
    training_set = lin.normalize_eq(training_set)
    training_set = np.c_[np.ones([len(training_set),1]), training_set]
    test_set = lin.normalize_eq(test_set)
    test_set = np.c_[np.ones([len(test_set),1]), test_set]
    beta = np.array([0,0,0,0,0,0,0,0,0,0])
    
    # plt.plot(training_set[:,:9], training_set_y, "ro")
    # plt.show()
    
    for n in range(10000):
      beta = lin.gradient_log(beta, 1.5, training_set, training_set_y)
    #   plt.scatter(n, cost_eq(training_set, training_set_y, np.dot(training_set,beta)), edgecolors="red", c = "none")
    # plt.show()
    print(lin.cost_log(training_set,training_set_y,beta))
    print("Alpha =", 0.2, "N =", n+1)
    
    print("Round:", i+1)
    print("Training errors:", lin.training_errors(training_set, beta, training_set_y))
    print("Training accuracy =", (len(training_set)-lin.training_errors(training_set, beta, training_set_y))/len(training_set))
    print("Test errors:", lin.training_errors(test_set, beta, test_set_y))
    print("Test accuracy =", (len(test_set)-lin.training_errors(test_set, beta, test_set_y))/len(test_set))
    
    # Repeated runs seem to give a similar qualitatively result. Around 97% accuracy for test. It varies between 96-99% though.

if __name__ == "__main__":
  main()