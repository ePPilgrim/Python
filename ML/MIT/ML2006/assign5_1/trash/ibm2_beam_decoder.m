function englisch=ibm2_beam_decoder(T,D,lom,LM,deutsch)
  beamwidth  = 20;
  m          = length(find(deutsch));
  fwords     = log(full(max(T(full(deutsch(:,1:m)), :), [], 2)));
  hypotheses = [2,3];
  covered    = [zeros(1,m)];
  scores     = [0];
  fcosts     = [sum(fwords)];
  l          = lom(m);
  for i=1:l
    nhypotheses = []; ncovered = []; nscores = []; nfcosts = [];
    for hidx=1:size(hypotheses,1)
      for j=find(covered(hidx,:) == 0)
        for ne=find(T(deutsch(j),:))
          nc = covered(hidx,:) ; nc(j) = nc(j) + 1;
          ns = scores(hidx)                       ...
               + log(eps + T(deutsch(j),ne))      ...
               + log(eps + D(indexpack(j,l,m),i)) ...
               + log(eps + LM(twowordpack(hypotheses(hidx,i),hypotheses(hidx,i+1)), ne));
          nf = ns + sum(fwords' .* (1 - nc));
          nhypotheses = [nhypotheses ; hypotheses(hidx,:), ne];
          ncovered    = [ncovered    ; nc];
          nscores     = [nscores     ; ns];
          nfcosts     = [nfcosts     ; nf];
        end
      end
    end
    % cut out a beam
    beam = sortrows([-nfcosts, (1:length(nfcosts))']);
    beam = beam(1:min(beamwidth,size(nfcosts,1)),2);
    hypotheses = nhypotheses(beam,:);
    covered    = ncovered(beam,:);
    scores     = nscores(beam,:);
    fcosts     = nfcosts(beam,:);
    [scores, fcosts]
  end
  englisch = hypotheses(1,:);
end
