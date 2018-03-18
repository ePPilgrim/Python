function [j,l,m]=indexunpack(idx)
  % this assumes the longest sentence is 50 words long
  % in both languages
  m = fix(idx / (50 * 50));
  idx = rem(idx, 50*50);
  l = fix(idx / 50);
  j = rem(idx, 50);
