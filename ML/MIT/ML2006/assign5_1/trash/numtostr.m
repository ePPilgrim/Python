function str=numtostr(lex,num)
  fnum = full(num);
  fnum = fnum(find(num));
  str=lex(fnum);
end
