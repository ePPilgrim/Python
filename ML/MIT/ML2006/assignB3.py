import numpy as np
import numpy.random as npr
import numpy.linalg as lin
import matplotlib.pyplot as plt
from scipy import ndimage as im

def strimage(a):
 xy = [(int(j[0]), int(j[1])) for j in [i.split(':') for i in a[4:].split()]]
 grid=np.zeros(784)
 for pic in xy:
  grid[pic[0]]=pic[1]*100.0/255.0
 img = grid.reshape(28,28)
 plt.imshow(img,cmap=plt.cm.gray)
 plt.show() 

#strimg = '-1 127:51 128:159 129:253 130:159 131:50 154:48 155:238 156:252 157:252 158:252 159:237 181:54 182:227 183:253 184:252 185:239 186:233 187:252 188:57 189:6 207:10 208:60 209:224 210:252 211:253 212:252 213:202 214:84 215:252 216:253 217:122 235:163 236:252 237:252 238:252 239:253 240:252 241:252 242:96 243:189 244:253 245:167 262:51 263:238 264:253 265:253 266:190 267:114 268:253 269:228 270:47 271:79 272:255 273:168 289:48 290:238 291:252 292:252 293:179 294:12 295:75 296:121 297:21 300:253 301:243 302:50 316:38 317:165 318:253 319:233 320:208 321:84 328:253 329:252 330:165 343:7 344:178 345:252 346:240 347:71 348:19 349:28 356:253 357:252 358:195 371:57 372:252 373:252 374:63 384:253 385:252 386:195 399:198 400:253 401:190 412:255 413:253 414:196 426:76 427:246 428:252 429:112 440:253 441:252 442:148 454:85 455:252 456:230 457:25 466:7 467:135 468:253 469:186 470:12 482:85 483:252 484:223 493:7 494:131 495:252 496:225 497:71 510:85 511:252 512:145 520:48 521:165 522:252 523:173 538:86 539:253 540:225 547:114 548:238 549:253 550:162 566:85 567:252 568:249 569:146 570:48 571:29 572:85 573:178 574:225 575:253 576:223 577:167 578:56 594:85 595:252 596:252 597:252 598:229 599:215 600:252 601:252 602:252 603:196 604:130 622:28 623:199 624:252 625:252 626:253 627:252 628:252 629:233 630:145 651:25 652:128 653:252 654:253 655:252 656:141 657:37'
#strimage(strimg)

f1 = './data/train-01-images.svm'
f2 = './data/train-01-images-W.svm'
f3 = './data/test-01-images.svm'

def show_pic_list(filename, lst):
 f = open(filename,'r')
 str = f.read().split('\n')
 f.close()
 for ind in lst:
  strimage(str[ind])

