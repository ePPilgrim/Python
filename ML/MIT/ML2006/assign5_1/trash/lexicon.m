function lex=lexicon(fname)
  lex = {'UNK', '', ''};
  fde = fopen(fname, 'r');
  a = fscanf(fde,'%s',1);
  while strcmp(a, '') == 0
    lex = [lex, a];
    a = fscanf(fde,'%s',1);
  end
end
