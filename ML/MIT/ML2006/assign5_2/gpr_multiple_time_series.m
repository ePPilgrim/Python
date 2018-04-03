function [out1, out2] = gpr_multiple_time_series(logtheta, covfunc, x, y, xstar);

% gpr - Gaussian process regression, with a named covariance function. Two
% modes are possible: training and prediction: if no test data are given, the
% function returns minus the log likelihood and its partial derivatives with
% respect to the hyperparameters; this mode is used to fit the hyperparameters.
% If test data are given, then (marginal) Gaussian predictions are computed,
% whose mean and variance are returned. Note that in cases where the covariance
% function has noise contributions, the variance returned in S2 is for noisy
% test targets; if you want the variance of the noise-free latent function, you
% must substract the noise variance.
%
% usage: [nlml dnlml] = gpr_multiple_time_series(logtheta, covfunc, x, y)
%    or: [mu S2]  = gpr_multiple_time_series(logtheta, covfunc, x, y, xstar)
%
% where:
%
%   logtheta is a (column) vector of log hyperparameters
%   covfunc  is the covariance function
%   x        is a n x 1 vector of time points with observations
%   y        is a m x n matrix of values for m sets of observations 
%   xstar    is a nn x 1 vector of test training point inputs
%   nlml     is the returned value of the negative log marginal likelihood
%   dnlml    is a (column) vector of partial derivatives of the negative
%                 log marginal likelihood wrt each log hyperparameter
%   mu       is a (column) vector (of size nn) of prediced means
%   S2       is a (column) vector (of size nn) of predicted variances
%
% For more help on covariance functions, see "help covFunctions".
%
% Modified by Rohit Singh, based on code by
% Carl Edward Rasmussen (2006-03-20).

if ischar(covfunc), covfunc = cellstr(covfunc); end % convert to cell if needed
[m, n] = size(y);
%n1 = length(x)
x = reshape(x,n,1);
D = 1;
if nargin==5
  nn = length(xstar);
end

if eval(feval(covfunc{:})) ~= size(logtheta, 1)
  error('Error: Number of parameters do not agree with covariance function')
end

K = feval(covfunc{:}, logtheta, x);    % compute training set covariance matrix
L = chol(K)';                        % cholesky factorization of the covariance

%size(K)
%size(L)
if nargin==4
  out1 = 0;
  out2 = zeros(size(logtheta));
else
  out1 = zeros(m,nn);
  out2 = zeros(m,nn);
end

for j=1:m  
  y_j = reshape(y(j,:),n,1);
  alpha = solve_chol(L',y_j);
  
  if nargin == 4 % if no test cases, compute the negative log marginal likelihood

    out1 = out1 + (0.5*y_j'*alpha + sum(log(diag(L))) + 0.5*n*log(2*pi));
    
    if nargout == 2               % ... and if requested, its partial derivatives
      W = L'\(L\eye(n))-alpha*alpha';                % precompute for convenience
      for i = 1:length(out2)
	out2(i) = out2(i) + (sum(sum(W.*feval(covfunc{:}, logtheta, x, i)))/2);
      end
    end

  else                    % ... otherwise compute (marginal) test predictions ...
    [Kss, Kstar] = feval(covfunc{:}, logtheta, x, xstar);     %  test covariances

    out1a = Kstar' * alpha;                                      % predicted means
    out1(j,:)  = reshape(out1a,1,nn);
    
    if nargout == 2
      v = L\Kstar;
      out2a = Kss - sum(v.*v)';
      out2(j,:) = out2a; 
    end  

  end
end
