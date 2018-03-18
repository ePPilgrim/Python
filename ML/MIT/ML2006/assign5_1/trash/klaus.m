function klausde=klaus()
  dec_file = fopen('filt-klaus-counts','r');
  de_file  = fopen('filt-klaus','r');
  delens = fscanf(dec_file,'%d');
  N = length(delens);
  klausde = sparse(N,max(delens));
  for i=1:N
    klausde(i,1:delens(i)) = fscanf(de_file,'%d',delens(i));
  end
end
