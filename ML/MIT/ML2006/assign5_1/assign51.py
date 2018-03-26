
import numpy as np
import scipy.sparse as sp

indexpack = lambda x, y, z: 50 * (50 * z + y ) + x
twowordpack = lambda x, y: y * 20001 + x
numtostr = lambda lex, num: [lex[x - 1] for x in num if x != 0]

def ibm2_train_lm(english):
    LM = sp.csc_matrix((20001 ** 2,20001))
    LMc = sp.csc_matrix((20001,20001))
    
    #We define three additional tokens:
    UNK = 1  # The uknown word (for words unseen previously)
    S1  = 2  # S-1
    S0  = 3  # S0 -- the start symbols, as defined in the pset
    
    for i in range(english.shape[0]):
        m1 = S0
        m2 = S1
        for j in range(english.shape[1]):
            if english[i][j] == 0:
                break
            m2m1 = twowordpack(m1, m2)
            LM[m2m1,english[i][j]] += 1
            LMc[m2,m1] += 1
            m1 = m2
            m2 = english[i][j]
    
    (I, J, V) = sp.find(LM)
    Jr = I // 20001
    Jc = I % 20001
    LM[I, J] = V / LMc[Jr, Jc]
    return (LM, LMc)


def ibm2_train(english,deutsch):
    en_vocab = 1 + int(english.max())
    de_vocab = 1 + int(deutsch.max())
    emax = english.shape[1]
    dmax = deutsch.shape[1]
    N = english.shape[0]
    
    #initialize T and D
    print('Initializing.....')
    T = sp.csc_matrix(de_vocab, en_vocab)
    D = sp.csc_matrix(((dmax+1) * 50 ** 2, emax))
    lom = np.zeros((dmax,emax))
    for i in range(N):
        (sink, sink, l) = sp.find(english[i,:])
        (sink, sink, m) = sp.find(deutsch[i,:])
        T[m,l] = 1 / m.size
        D[indexpack(np.arange(m.size), l.size, m.size), np.arange(l.size)] = 1 / l.size
        lom[m.size, l.size] += 1 
    #[trash,lom] = max(lom,[],2);
    lom = np.argmax(lom,axis = 1)
    print('done.\n');
    for em_iter_idx in range(50):
        Tn = sp.csc_matrix((de_vocab, en_vocab));
        Dn = sp.csc_matrix(((dmax+1) * 50 ** 2, emax));
        tNorm = np.zeros(en_vocab)
        dNorm = np.zeros((dmax+1) * 50 * 50)
        for idx in range(N):
            (sink, sink, l) = sp.find(english[idx,:])
            (sink, sink, m) = sp.find(deutsch[idx,:])
            for j in range(m.size):
                (Jd,Id,Vd) = sp.find(D[indexpack(j,l.size,m.size),:])
                ll = english[idx,Id].toarray()
                den = np.dot(T[j,ll].toarray(), Vd)
                for i in range(l.size):
                    val = T[deutsch[idx,j],english[idx,i]] * D[indexpack(j, l.size, m.size), i] / den
                    Tn[deutsch[idx,j],english[idx,i]] += val
                    Dn[indexpack(j,l.size,m.size),i] += val
                    tNorm[english[idx,i]] += val
                    dNorm[indexpack(j, l.size, m.size)] += val
        T, D = Tn, Dn
        for enword in range(en_vocab):
            T[:,enword] /= tNorm[enword]   
        [I,J,V] = sp.find(D)
        D[I,J] /= V / dNorm[I]
    print('Done with calculation\n')
    return (T, D, lom)


def europarl():
    enc_file = open('filt-en-counts','r')
    en_file  = open('filt-en','r')
    lmc_file = open('filt-lm-counts','r')
    lm_file  = open('filt-lm','r')
    dec_file = open('filt-de-counts','r')
    de_file  = open('filt-de','r')
    enlens = np.array([int(x) for x in enc_file.read().split()])
    delens = np.array([int(x) for x in dec_file.read().split()])
    lmlens = np.array([int(x) for x in lmc_file.read().split()])
    
    N = len(enlens)
    english = sp.csc_matrix((N,max(enlens)))
    lmenglish = sp.csc_matrix((len(lmlens),max(lmlens)))
    deutsch = sp.csc_matrix((N,max(delens)))
    fmtstr = 'Loaded %5d sentence pairs'
    fmtstrl = 27
    print(fmtstr, 0)
    for i in range(N):
        english[i,np.arange(enlens[i])] = np.array([int(x) for x in en_file.read(enlens[i]).split()])
        deutsch[i,np.arange(delens[i])] = np.array([int(x) for x in de_file.read(delens[i]).split()])
        if i % 100 == 0:
            for bcnt in range(fmtstrl):
                print('\b')
        print(fmtstr, i)
    print('\n');
    fmtstr = 'Loaded %5d sentences (for the lm)'
    fmtstrl = 35
    print(fmtstr, 0)
    for i in range(len(lmlens)):
        lmenglish[i, np.arange(lmlens[i])] = np.array([int(x) for x in lm_file.read(lmlens[i]).split()])
        if i % 100 == 0:
            for bcnt in range(fmtstrl):
                print('\b')
        print(fmtstr, i);
    print('\n');
    return (english, deutsch, lmenglish)


def ibm2_beam_decoder(T,D,lom,LM,deutsch):
    beamwidth  = 20
    m          = len(sp.find(deutsch)[0])
    fwords     = np.log(T[deutsch[:,np.arange(m)].toarray(), :].max(1).toarray())
    hypotheses = [[2,3]]
    covered    = [[0] * m]
    scores     = [0]
    fcosts     = [sum(fwords)]
    l          = lom[m]
    eps = np.finfo(np.float32).eps
    for i in range(l):
        nhypotheses, ncovered, nscores, nfcosts = [], [], [], []
        for hidx in range(len(hypotheses)):
            for j in [ x for x in range(len(covered[hidx])) if covered[hidx][x] == 0]:
                for ne in sp.find(T[deutsch[j],:])[1]:
                    if ne != 0:
                        nc = covered[hidx]
                        nc[j] += 1
                        ns = scores[hidx] \
                        + np.log(eps + T[deutsch[j],ne]) \
                        + np.log(eps + D[indexpack(j,l,m),i]) \
                        + np.log(eps + LM[twowordpack(hypotheses[hidx][i],hypotheses[hidx][i+1]), ne])
                        nf = ns + np.dot(fwords, 1 - np.array(nc))
                        nhypotheses.append(hypotheses[hidx].append(ne));
                        ncovered.append(nc)
                        nscores.append(ns)
                        nfcosts.append(nf)
    # cut out a beam
        beam = np.argsort([-x for x in nfcosts])
        beam = beam[ : min(beamwidth,len(nfcosts))]
        hypotheses = [nhypotheses[i] for i in beam]
        covered    = [ncovered[i] for i in beam]
        scores     = [nscores[i] for i in beam]
        fcosts     = [nfcosts[i] for i in beam]
        scores + fcosts
    return hypotheses[1]


def klaus():
    dec_file = open('filt-klaus-counts','r');
    de_file  = open('filt-klaus','r');
    delens = np.array([int(x) for x in dec_file.read().split()])
    N = len(delens)
    klausde = sp.csc_matrix((N,max(delens)))
    for i in range(N):
        klausde[i,np.arange(delens[i])] = np.array([int(x) for x in de_file.read(delens[i]).split()])
    return klausde       


def lexicon(fname):
    lex = ['UNK', '', '']
    fde = open(fname, 'r');
    lex = lex + fde.read().split()
    return lex


delex = lexicon('data-de')
# almost a minute
enlex = lexicon('data-en')
# may take several minutes (but has status indicator)
(english,deutsch,lmenglish)=europarl()
# may also take several minutes (with status indicator)
(N,mmax) = english.shape 
mmax = lmenglish.shape[1] - mmax
(LM , LMc) = ibm2_train_lm(sp.vstack([lmenglish, sp.hstack([english, sp.csc_matrix((N, mmax))])]))
# should be very quick
(T,D,lom)=ibm2_train(english , deutsch);
# also very quick
germans = klaus();
for i in range(germans.shape[0]):
    german = germans[i,:]
    englisch = ibm2_beam_decoder(T,D,lom,LM,german)
    numtostr(delex,german)
    numtostr(enlex,englisch)
