import tensorflow as tf
import pandas as pd
import numpy as np
import os


TermMapToMonth = { 'F1' : 0.25, 'F2' : 0.5, 'M01' : 1.0, 'M02' : 2.0, 'M03' : 3.0, 'M04' : 4.0, 'M05' : 5.0, 'M06' : 6.0, 'M07' : 7.0, 'M08' : 8.0,
                    'M09' : 9.0, 'M10' : 10.0, 'M11' : 11.0, 
                    'Y01' : 12.0, 'Y02' : 24.0, 'Y03' : 36.0, 'Y04' : 48.0, 'Y05' : 60.0, 'Y06' : 72.0, 'Y07' : 84.0,
                    'Y08' : 96.0, 'Y09' : 108.0, 'Y10' : 120.0, 'Y12' : 144.0, 'Y15' : 180.0, 'Y20' : 240.0,
                    'Y25' : 300.0, 'Y30' : 360.0, 'Y40' : 480.0, 'Y50' : 600.0}

class NelsonSiegelLayer(tf.keras.layers.Layer):
    def __init__(self, thau0 = 64.0, eps = 0.00001):
        super(NelsonSiegelLayer, self).__init__()
        self.eps = eps
        self.thau = self.add_weight(name = "Thau", shape = (), 
                                     initializer = tf.keras.initializers.Constant(thau0),
                                     #constraint = tf.keras.constraints.NonNeg(),
                                     trainable = True)
        self.alpha0 = self.add_weight(name = "Alpha0", shape = (), 
                                     initializer = tf.keras.initializers.RandomUniform(0.0, 1.0),
                                     constraint = tf.keras.constraints.NonNeg(),
                                     trainable = True)
        self.alpha1 = self.add_weight(name = "Alpha1", shape = (), 
                                     initializer = tf.keras.initializers.RandomUniform(0.0, 1.0),
                                     constraint = tf.keras.constraints.NonNeg(),
                                     trainable = True)
        self.alpha2 = self.add_weight(name = "Alpha2", shape = (), 
                                     initializer = "random_normal",                           
                                     trainable = True)
        #self.bias = self.add_weight(name = "Bias", shape = (), trainable = True)
        
    def reassignValues(self, thau0 = 64.0):
        self.thau.assign(thau0)
        self.alpha0.assign(tf.random.uniform(shape=(), minval = 0.0, maxval = 1.0))
        self.alpha1.assign(tf.random.uniform(shape=(), minval = 0.0, maxval = 1.0))
        self.alpha2.assign(tf.random.normal(shape=()))
        
    def assignValues(self, x):
        self.thau.assign(x[0])
        self.alpha0.assign(x[1])
        self.alpha1.assign(x[2])
        self.alpha2.assign(x[3])
        
    def call(self, inputs):
        val1 = tf.divide(inputs, self.thau)
        val2 = tf.math.exp(-val1)
        val3 = tf.divide(1.0 - val2, val1)
        eps = self.eps
        return tf.add(tf.math.multiply(self.alpha0 + eps, 1.0 - val3), 
                                       tf.add(tf.math.multiply(self.alpha1 + eps, val3),
                                        tf.math.multiply(self.alpha2, val3 - val2)))

class NelsonSiegelParameters:
    def __init__(self, ww = 13, thauv = [16, 64, 128], lr = 0.1, epochs = 200):
        self.ww = ww
        self.thauv = thauv
        self.nsld = {thau : NelsonSiegelLayer(thau0 = thau) for thau in self.thauv}
        self.lr = lr
        self.epochs = epochs
        self.modelv = [self.__create_model(nsl) for nsl in self.nsld.values()]
        self.histv = None
        self.resdf = pd.DataFrame(columns = ['Date', 'Alpha0', 'Alpha1', 'Alpha2', 'Thau'])
                                  
    def __create_model(self,nsl):
        inputs = tf.keras.Input(shape = ())
        outputs = nsl(inputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr), loss="mse")
        return model        
    
    def __train_step(self, x, y):
        for thau,nsl in self.nsld.items():
            nsl.reassignValues(thau)
        self.histv = [model.fit(x, y, epochs = self.epochs, verbose = 0) for model in self.modelv]
        
    def find_parameters(self, filename):
        cv = list(pd.read_csv(filename, nrows = 0).columns)
        rec = [tf.float32] * len(cv)
        rec[cv.index('Date')] = tf.string
        dataset = tf.data.experimental.CsvDataset(filename, rec, header = True).window(self.ww,1,1,True)
        resdict = {key : [] for key in self.resdf.columns}
        i = 0
        for window in dataset:
            mdate, x, y = None, None, None
            for column, component in zip(cv, window):
                if column == 'Date':
                    mdate = list(component.as_numpy_iterator())
                    mdate = mdate[len(mdate) // 2]
                    continue
                _y = np.array(list(component.as_numpy_iterator()))
                _y = _y[_y > -255.0]
                if _y.shape[0] == 0:
                    continue
                _x = np.repeat(TermMapToMonth[column], _y.shape[0])
                if x is None:
                    x, y = (_x, _y)
                    continue
                x = np.concatenate([x, _x])
                y = np.concatenate([y,_y])
            self.__train_step(x,y)
            resdict['Date'] += [mdate] * len(self.thauv)
            print("Window - {}".format(i))
            i += 1
            for val in self.nsld.values():
                resdict['Alpha0'].append(val.alpha0.numpy())
                resdict['Alpha1'].append(val.alpha1.numpy())
                resdict['Alpha2'].append(val.alpha2.numpy())
                resdict['Thau'].append(val.thau.numpy())
        self.resdf = pd.DataFrame(resdict)
        self.resdf['Date'] = pd.to_datetime(self.resdf['Date'].str.decode('utf-8'))
        return self.resdf
    
def SaveNSParameters(srcdir, destdir, ww = 13, epochs = 200, thauv = [16, 64, 128]):
    engine, df = None, None
    for filename in os.listdir(srcdir):
        filepath = os.path.join(srcdir,filename)
        if os.path.isfile(filepath):
            engine = NelsonSiegelParameters(ww = ww, epochs = epochs, thauv = thauv)
            df_ext = engine.find_parameters(filepath)
            df_orig = pd.read_csv(filepath, converters = {'Date' : lambda x: pd.to_datetime(x)})
            df = df_orig.merge(df_ext,on = 'Date', sort = True, how = 'outer')
            file_name, file_extension = os.path.splitext(filename)
            newfilename = file_name + '_ns' + file_extension
            df.to_csv(os.path.join(destdir, newfilename))
    return (engine, df)
