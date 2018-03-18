function [T,D,lom] = ibm2_train(english,deutsch)
  N = size(english, 1);          % number of sentence pairs
  en_vocab = det(max(max(english)));  % number of english words
  de_vocab = det(max(max(deutsch)));  % number of german words
  lmax = size(english, 2);       % length of longest english sentence
  mmax = size(deutsch, 2);       % length of longest german sentence
% initialize T and D
  fprintf('Initializing...');
  T = sparse(de_vocab, en_vocab);
  D = sparse((mmax+1) * 50 * 50, lmax);
  lom = zeros(mmax,lmax);
  for idx=1:N
    l = length(find(english(idx,:)));
    m = length(find(deutsch(idx,:)));
    % T(deutsch(idx,[1:m]),english(idx,[1:l])) = ??
    % D(indexpack([1:m],l,m),[1:l]) = ??
    lom(m,l) = lom(m,l) + 1;
  end
  [trash,lom] = max(lom,[],2);
  fprintf('done.\n');

  fprintf('Training...');
  for em_iter_idx=1:50
    Tn = sparse(de_vocab, en_vocab);
    Dn = sparse((mmax+1) * 50^2, lmax);
    % loop through all the sentence pairs
    for idx=1:N
      l = length(find(english(idx,:)));
      m = length(find(deutsch(idx,:)));
      for j=1:m
        for i=1:l
%          Tn(deutsch(idx,j),english(idx,i)) = ??
%          Dn(indexpack(j,l,m),i) = ??
        end
      end
    end
    fprintf('%d', em_iter_idx);
    %
    T = sparse(de_vocab, en_vocab);
    D = sparse((mmax+1) * 50 * 50, lmax);
    for enword=1:en_vocab
      for deword=find(Tn(:,enword))
%        T(deword, enword) = ??
      end
    end
%    D = ?? 
    fprintf('n...');
  end
  fprintf('done.\n');
end
