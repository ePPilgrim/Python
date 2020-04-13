import os
import tensorflow as tf
import pandas as pd

class Generator(object):
    '''
    Create generator for binary classification where class '0' is first subset of categories and '1' includs other categories.
    '''
    def __init__(self, csvFile, rootDir, imgSize):
        self.baseDf = None
        self.curDf = None
        self.rootDir = None
        self.imgSize = None
        if os.path.isfile(csvFile):
            self.baseDf = pd.read_csv(csvFile)
        if os.path.exists(rootDir):
            self.rootDir = rootDir
        if imgSize is not None:
            self.imgSize = imgSize

    def GetGenerator(self, bs = 32):
        if self.imgSize is None or not self.rootDir or self.curDf is None:
            return None
        for i in range(256):
            self.curDf= self.curDf.sample(frac = 1).reset_index(drop=True)
        gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
        return gen.flow_from_dataframe(self.curDf,target_size = self.imgSize,directory = self.rootDir,
                                            class_mode = 'binary', shuffle = True,batch_size = bs)
    def SetDataFrameForGeneration(self, cv, nv):
        self.curDf = None
        if not cv or not nv:
            return
        if len(cv) != 2 or len(nv) != 2:
            return 
        fv1 = self._get_classes_elements(cv[0],nv[0])
        fv2 = self._get_classes_elements(cv[1],nv[1])
        self.curDf = pd.DataFrame({'filename' : fv1 + fv2, 'class' : ['0']*len(fv1) + ['1']*len(fv2)})

    def _get_classes_elements(self,catv,nv):
        res = []
        for i in range(len(catv)):
            res += self._get_single_class(catv[i],nv[i])
        return res

    def _get_single_class(self, cat, n):
        lx = (self.baseDf['class'] == cat) & (self.baseDf['type'] == 0)
        res = list(self.baseDf[lx]['filename'].to_numpy())
        if lx.sum() > n:
            return res[:n]
        m = n - lx.sum()
        lx = (self.baseDf['class'] == cat) & (self.baseDf['type'] != 0)
        lst = list(self.baseDf[lx].sort_values(by='type')['filename'].to_numpy())
        res += lst[:m]
        return res


 