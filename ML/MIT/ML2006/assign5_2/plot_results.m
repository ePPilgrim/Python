function plot_results(W, V, t, Yobs, k, nn)
% plot_results(W, V, t, Yobs, k, nn)
%   Plot curves from each cluster, choosing nn curves randomly from each
%

w = ceil(k/2) % plot an L x 2 array of subplots. you can change this to be L x 3 or something

[n, r] = size(Yobs);

t_min = min(t);
t_max = max(t);
t_range = linspace(t_min, t_max, 100);

[dummy, gene_class] = max(W, [], 2);
for j=1:k,
  subplot(2,w,j);
  
  % pick nn rows from Yobs to be plotted
  idx1 = find(gene_class == j);
  if length(idx1) <= nn
    idx = idx1; % less than nn members, choose all
  else
    n1 = length(idx1);  %more than nn members, pick some nn members randomly
    a1 = randperm(nn); 
    a2 = a1(1:nn);
    idx = idx1(a2);
  end

  %predict values at t_range
  x = reshape(t,length(t),1);
  xstar = reshape(t_range, length(t_range), 1);
  covfunc = {'covSum', {'covSEiso', 'covNoise'}};

  params = V(j,:);
  logtheta = [log(params(2)), log(params(1)), log(params(3)) ];
  K = feval(covfunc{:}, logtheta, x);
  L = chol(K)';
  t_vals = zeros(length(idx), length(t_range));
  for i=1:length(idx)
    y_i1 = Yobs(idx(i),:);
    y_i = reshape(y_i1, length(y_i1), 1);
    alpha = solve_chol(L',y_i);
    [Kss, Kstar] = feval(covfunc{:}, logtheta, x, xstar);
    v = L\Kstar;
    t_vals(i,:) = reshape(Kstar' * alpha, 1, length(t_range));
  end  
  t1 = repmat(t_range, length(idx),1);
  plot(t1', t_vals','-');
  hold on;
  t2 = repmat(t, length(idx),1);
  plot(t2', Yobs(idx,:)', '+');
  
  hold off;
end