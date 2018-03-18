function [LM,LMc]=ibm2_train_lm(english)
  % we'll store the trigrams in a sparse, 2d matrix
  % the first index corresponds to both of the previous
  % two words, packed using twowordpack.m
  LM = sparse(400000000,20000);
  % this is a store of totals for normalization
  LMc = sparse(20000,20000);
  N = size(english,1);    % number of English sentences
  mmax = size(english,2); % length of longest English sentence

  % We define three additional tokens:
  UNK = 1;  % The uknown word (for words unseen previously)
  S1  = 2;  % S-1
  S0  = 3;  % S0 -- the start symbols, as defined in the pset

  % progress indicator
  fmtstr = 'Processed %5d English sentences'; fmtstrl = 33;
  fprintf(fmtstr, 0);
  for i=1:N
    m1 = S0;
    m2 = S1;
    for j=1:mmax
      m2m1 = twowordpack(m2,m1);
      if english(i,j) == 0
        break
      end
%      LM(m2m1,english(i,j)) = ??
      LMc(m2,m1) = LMc(m2,m1) + 1;
      m2 = m1;
      m1 = english(i,j);
    end
    if mod(i,1000) == 0
      for bcnt=1:fmtstrl
        fprintf('\b')
      end
      fprintf(fmtstr, i);
    end
  end
  for bcnt=1:fmtstrl
    fprintf('\b')
  end
  fprintf(fmtstr, i); fprintf('\n');
  for i=1:20000
    m2m1_indices = find(LM(:,i));
    if length(m2m1_indices) > 0
      for m2m1=full(m2m1_indices)'
        [m2,m1] = twowordunpack(m2m1);
%        LM(m2m1,i) = ??
      end
    end
  end
end
