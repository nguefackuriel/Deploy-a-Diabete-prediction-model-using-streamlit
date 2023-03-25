def covariance(x, mu):

  # Easy way: cov= np.cov(x, rowvar=0) but do not use it. One can use it to assess his/her result.
  m, d = x.shape
  #mu = mu.reshape(1,-1)
  cov = (1/(m-1))*((x-mu).T@(x-mu))
  return cov
