import numpy as np
import numpy.linalg as linalg

def log_likelihood_gp(params, t, Yobs):
%  ll = log_likelihood_gp(params, t, Yobs)
%
% computes the log likelihood, one for each row of Yobs, of observing
% the data as per a GP specified by the params
% 
% INPUT:
%   params(3,1): the parameters of the covariance matrix for the GP
%   t(r,1) : the r timepoints at which observations are available
%   Yobs(q,r) : the observations for q genes (q can be 1)
% OUTPUT:
%   ll(q,1): the log-likelihood of seeing the data in the q-th Yobs  row
%

    (q, r) = Yobs.shape
    G = np.zeros((r,r))
    for i in range(r):
        for j in range(r):
            G[i,j] = (params[0] ** 2 * np.exp(-((t[i] - t[j]) ** 2) / (2 * (params[1] ** 2))))
            if i == j:
                G[i, j] = G[i, j] + params[2] ** 2
    detG = linalg.det(G)
    invG = linalg.inv(G)
    ll = zeros(q,1);
    for i in range(r):
        l = 
  l = ...
  ll(i) = l;
end

