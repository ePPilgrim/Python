function ll = log_likelihood_gp(params, t, Yobs)
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

[q, r] = size(Yobs);

%Gram matrix
G = zeros(r,r);
for i=1:r,
  for j=1:r,
    G(i,j) = (params(1)^2 * exp(-((t(i)-t(j))^2)/(2*(params(2)^2)))) ;
    if i==j,
      G(i,j) = G(i,j) + params(3)^2;
    end
  end
end

ll = zeros(q,1);
for i=1:q
  l = ...
  ll(i) = l;
end
