function [english,deutsch,lmenglish]=europarl()
  enc_file = fopen('filt-en-counts','r');
  en_file  = fopen('filt-en','r');
  lmc_file = fopen('filt-lm-counts','r');
  lm_file  = fopen('filt-lm','r');
  dec_file = fopen('filt-de-counts','r');
  de_file  = fopen('filt-de','r');
  enlens = fscanf(enc_file,'%d');
  lmlens = fscanf(lmc_file,'%d');
  delens = fscanf(dec_file,'%d');
  N = length(enlens);
  english = sparse(N,max(enlens));
  lmenglish = sparse(length(lmlens),max(lmlens));
  deutsch = sparse(N,max(delens));
  fmtstr = 'Loaded %5d sentence pairs'; fmtstrl = 27;
  fprintf(fmtstr, 0);
  for i=1:N
    english(i,1:enlens(i)) = fscanf(en_file,'%d',enlens(i));
    deutsch(i,1:delens(i)) = fscanf(de_file,'%d',delens(i));
    if mod(i,100) == 0
      for bcnt=1:fmtstrl
        fprintf('\b')
      end
      fprintf(fmtstr, i);
    end
  end
  fprintf('\n');
  fmtstr = 'Loaded %5d sentences (for the lm)'; fmtstrl = 35;
  fprintf(fmtstr, 0);
  for i=1:length(lmlens)
    lmenglish(i,1:lmlens(i)) = fscanf(lm_file,'%d',lmlens(i));
    if mod(i,100) == 0
      for bcnt=1:fmtstrl
        fprintf('\b')
      end
      fprintf(fmtstr, i);
    end
  end
  fprintf('\n');
end
