function idx=indexpack(j,l,m)
  % this assumes the longest sentence is 50 words long
  % in both languages
  idx = m * 50 * 50 + l * 50 + j;
