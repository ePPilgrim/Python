import pandas as pd

class Generator(object):
    def __init__(self, csvFile, rootDir,imgSize):
        self.df = None
        self.rootDir = None
        self.imgSize = None
        if os.path.isfile(csvFile):
            self.df = pd.read_csv(csvFile)
        if os.path.exists(rootDir):
            self.rootDir = rootDir
        if not imgSize:
            self.imgSize = imgSize

    def __call__(self, bs = 32):
        if not self.imgSize or not self.rootDir or not self.df:
            return None
        for i in range(100):
            self.df = self.df.sample(frac = 1).reset_index(drop=True)
        gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
        return gen.flow_from_dataframe(self.df,target_size = (IMG_SIZE, IMG_SIZE),directory = self.rootDir,
                                            class_mode = 'binary', shuffle = True,batch_size = bs) 

    def _get_classes_elements(self,catv,nv):
        res = []
        for i in range(len(catv)):
            res += self._get_single_class(catv[i],nv[i])
        return res

    def _get_single_class(self, cat, n):
        lx = (self.df['class'] == cat) & (self.df['type'] == 0)
        res = list(self.df[lx]['filename'].to_numpy())
        if lx.sum() > n:
            return res[:n]
        m = n - lx.sum()
        lx = (self.df['class'] == cat) & (self.df['type'] != 0)
        lst = list(self.df[lx].sort_values(by='type')['filename'].to_numpy())
        res += lst[:m]
        return res


 