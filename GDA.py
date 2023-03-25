import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import covariance



class GDA:
  def __init__(self):
    ## set mu, phi and sigma to None
    self.phi = None
    self.mu = None
    self.sigma = None
    
  def fit(self,x,y):
    k=len(np.unique(y, return_counts=False)) # Number of class.
    d=x.shape[1]  # input dim
    m= x.shape[0] # Number of examples.
    
    ## Initialize mu, phi and sigma
    self.mu= np.zeros((k, d))#: kxd, i.e., each row contains an individual class mu.
    self.sigma= np.zeros((k,d,d))#: kxdxd, i.e., each row contains an individual class sigma.
    self.phi= np.zeros(d)# d-dimension

    ## START THE LEARNING: estimate mu, phi and sigma.

    for i in range(k):
      self.mu[i] = np.mean(x[y == i], axis = 0)
      self.phi[i] = (1/m)*np.sum(y[y == i])
      self.sigma[i] = covariance(x[y == i], self.mu[i]) 


  def predict_proba(self,x):
    '''
      Inputs: x(shape:m,d)
      Output: matrix of probability(shape:m,k)
    '''
    # input dim
    d= x.shape[1] 
    # Number of classes we have in our case it's k = 2
    k_class= self.mu.shape[0] 
    # we define a matrix that will contain our probabilities
    prob = np.zeros((x.shape[0],k_class))
    # Number of examples.
    m = x.shape[0]
    det = []
    inv_sigma = []
 
    for i in range(k_class):
      # we compute the determinant of each class
      det_ = np.linalg.det(self.sigma[i])
      # we compute the inverse of the covariance matrix for each class
      inv_sigma_ = np.linalg.inv(self.sigma[i])
      det.append(det_)
      inv_sigma.append(inv_sigma_)
      const = 1/np.sqrt((2*np.pi)**d*det[i])
      for j in range(m):
        prob[j,i] = const*np.exp(-0.5*(x[j]-self.mu[i]).T@inv_sigma[i]@(x[j] - self.mu[i]))
    return prob

  def predict(self,x):
    prob = self.predict_proba(x)
    y_pred = np.argmax(prob, axis = 1)
    return y_pred

  
  def accuracy(self, y, ypreds):
    acc = (np.mean(y == ypreds))*100
    return acc