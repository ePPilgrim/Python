function [W,V,L] = em_gp(t, Y_obs, k, init_class)
% [W,V,L] = em_gp(t, Y_obs, k, init_class)
% 
% EM algorithm for k Gaussian Processes mixture estimation
%
% Inputs:
%   t (T,1): the set of T timepoints [t(1) ... t(T)] at which observations  were made
%   Y_obs(n, T): the T observations for each of the n genes
%   k : maximum number of Gaussian Process components allowed
%   init_class(T,1) : the initial class assignment of each gene
%       init_class(i) is between 1 and k
%
% Ouputs:
%   W(n,k) - W(i,j) is the probability that i-th gene (row in Y_obs) belongs to class #j
%   V(k,3) -  estimated hyperparameters of the GP covariance matrices
%   L - log likelihood of estimates
%
% Written by
%   Rohit Singh ,   MIT ,  Nov 2006
%
% Based on code from MATLAB Central, written by
%   Patrick P. C. Tsui,
%   PAMI research group
%   Department of Electrical and Computer Engineering
%   University of Waterloo, 
%   March, 2006
%

%%%% Validate inputs %%%%
if nargin < 3,
    error('em_gp must have at least 3 inputs: t, Y_obs, k');
    return
end

[n, T] = size(Y_obs);
if length(t) ~= T | k < 2
  error('Bad inputs');
  return;
end

if nargin == 3
  % randomly assign genes to classes
  init_class = mod(randperm(n), k) + 1;
end

%%%% Initialize W, V,L %%%%
[W,V] = Init_EM(t, Y_obs, k, init_class); 
L = 0;    

% fake log-likelihood values to go through first iteration
L_this = 2;
L_old = 1;
tol = 1e-4;
maxiter = 20;

%%%% EM algorithm %%%%
niter = 0;
while (abs((L_this-L_old)/L_old) > tol & niter<=maxiter)
    niter = niter + 1;
    W = Expectation(t, Y_obs, k, W, V); % E-step    
    V = Maximization(t, Y_obs, k, V, W);  % M-step
    L_old = L_this;
    L_this = Overall_Log_Likelihood(t, Y_obs, k, W, V);
end 
L = L_this;


%%%%%%%%%%%%%%%%%%%%%%
%%%%        End of em_gp              %%%%
%%%%%%%%%%%%%%%%%%%%%%

function W_new = Expectation(t, Y_obs, k, W, V)
P = sum(W);
P = P/sum(P);

[n,T] = size(Y_obs);
W_new = zeros(n,k);

for j=1:k,
  % get likelihood
  ll = ...
  % get posterior assignment probability
  W_new(:,j) = ...
end

%normalize posterior
W_new = ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% End of Expectation %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function V_new = Maximization(t, Y_obs, k, V, W);  % E-step
V_new = zeros(k,3);

% find the most likely class for each gene
[dummy, max_W] = max(W, [], 2);

for i=1:k,  % Compute weights
  % grab observations for that class
  a  = find(max_W==i);
  Y_sub = Y_obs(max_W==i,:);
  
  if length(a)==0
      V_new(i,:) = V(i,:);
      continue;
  end
  %i,  size(Y_sub)
  
  if max(abs(V(i,:))) < 1e-10
    startparams_0 = V(i,:)';
  else
    startparams_0 = exp([-1,-1,-1]');
  end
  covfunc = {'covSum', {'covSEiso','covNoise'}};
  params_0 = zeros(3,1);
  params_0(1) = log(startparams_0(2));
  params_0(2) = log(startparams_0(1));%/2;
  params_0(3) = log(startparams_0(3));%/2;
  params = minimize(log(startparams_0), 'gpr_multiple_time_series', -50, covfunc, t , Y_sub);
  
  V_new(i,1) = exp(params(2)); %*2);
  V_new(i,2) = exp(params(1));
  V_new(i,3) = exp(params(3)); %*2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% End of Maximization %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function L = Overall_Log_Likelihood(t, Y_obs, k, W, V)
L = 0;
[n, T] = size(Y_obs);
S = zeros(n,k);
for i=1:k
  ll_i = log_likelihood_gp(V(i,:), t, Y_obs);
  S(:,i) = ll_i + log(W(:,i));
end

%S(S==0) = 1e-30;
S = exp(S);
s = sum(S,2);
s1 = log(s);
L = sum(s1);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% End of Likelihood %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [W,V] = Init_EM(t, Y_obs, k, init_class)
[n, T] = size(Y_obs);
W =zeros(n,k);
for i=1:n,
  W(i,init_class(i)) = 1;
end
V0 = exp(-1*ones(k,3));
V = Maximization(t, Y_obs, k, V0, W); 

W =(1/k)*ones(n,k);
%%%%%%%%%%%%%%%%%%%%%%%%
%%%% End of Init_EM %%%%
%%%%%%%%%%%%%%%%%%%%%%%%

