function [word1,word2]=twowordunpack(idx)
  word2 = mod(idx, 20000);
  word1 = fix(idx / 20000);
