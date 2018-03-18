function num=strtonum(lex,strs)
  num = [];
  for k=1:length(strs)
    num = [num,find(strcmp(lex,strs(k)),1)];
  end
end
