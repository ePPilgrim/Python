import math
import time
import os
import numpy as np
from imgprep import Augmentation 
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras import models

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TerminateOnNaN
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow import function



class BaseModel(object):
    def __init__(self,filters = 32, units = 512, input_shape = (224,224,3), learn_rate = 0.0001, conv_layer_cnt = None, L1 = None, L2 = None, dropout = None):
        self.filters = filters
        self.units = units
        self.input_shape = input_shape
        self.learn_rate = learn_rate
        self.l1 = L1
        self.l2 = L2
        self.drop = dropout
        self.conv_layer_cnt = conv_layer_cnt
        self.model = None
        upperlim = int(math.log2(min(input_shape[0:1]))) - 1
        if self.conv_layer_cnt is None or self.conv_layer_cnt > upperlim:
            self.conv_layer_cnt = upperlim
        
    def getDescription(self):
        name = "{}_F{}_U{}_LR{}".format(type(self).__name__, self.filters,self.units,str(self.learn_rate).split('.')[1])
        if self.l1 is not None:
            name += '_L1{}'.format(str(self.l1).split('.')[1])
        if self.l2 is not None:
            name += '_L2{}'.format(str(self.l2).split('.')[1])
        if self.drop is not None:
            name += '_D{}'.format(str(self.drop).split('.')[1])
        if self.conv_layer_cnt is not None:
            name += '_LCnt{}'.format(self.conv_layer_cnt)
        return name
    
    def compliteModelCreation(self,model):
        model.add(GlobalAveragePooling2D())
        if self.drop is not None:
            model.add(Dropout(self.drop))
        if self.l2 is not None:
            model.add(Dense(self.units, activation = 'relu',kernel_regularizer = regularizers.l2(self.l2)))
        elif self.l1 is not None:
            model.add(Dense(self.units, activation = 'relu',kernel_regularizer = regularizers.l1(self.l1)))
        else:
            model.add(Dense(self.units, activation = 'relu')) 
        model.add(Dense(1, activation = 'sigmoid'))
        model.compile(optimizer=Adam(lr=self.learn_rate),loss=BinaryCrossentropy(),metrics=['accuracy'])
        print('Model {} is built'.format(model.name))
        return model
    
    def build_core_part_of_model(self, model): 
        '''
            build_core_part_of_model() - is abstract method that should be implemented in the derived classes. 
        '''
        pass

    def Create(self):
        modelName = self.getDescription()
        model = Sequential(name = modelName)
        model.add(InputLayer(self.input_shape))
        model = self.build_core_part_of_model(model)
        self.model = self.compliteModelCreation(model)
        return self

    def fit(self, train_set, val_set, epochs = 0, shedule = None, log = False, threshold = None, patience = None, save_weights_only = True,  extra_callbacks = None):
        weight_dir = '.\\tmp_weights_{}'.format(self.getDescription())
        os.makedirs(weight_dir,exist_ok = True)
        filepath_loss = os.path.join(weight_dir,'checkpoint_loss_{}_.hdf5'.format(int(time.time()))) 
        filepath_acc = os.path.join(weight_dir,'checkpoint_acc_{}_.hdf5'.format(int(time.time()))) 
        callbacks = [TerminateOnNaN()]
        callbacks.append(ModelCheckpoint(filepath = filepath_loss, monitor='val_loss', save_best_only=True, save_weights_only= save_weights_only))
        callbacks.append(ModelCheckpoint(filepath = filepath_acc, monitor='val_accuracy', save_best_only=True, save_weights_only= save_weights_only))
        if shedule is not None:
            callbacks.append(LearningRateScheduler(schedule = shedule))
        if log:
            callbacks.append(CSVLogger('training_{}_{}.log'.format(self.getDescription(), int(time.time()))))
        if threshold is not None:
            filepath= os.path.join(weight_dir,'checkpoint_acc_{}_.hdf5'.format(int(time.time())))
            def fun(epoch,logs): 
                if logs['val_accuracy'] > threshold: 
                    file_name = os.path.join(weight_dir,'weights_acc_{:.2f}_{:02d}.hdf5'.format(logs['val_accuracy'], epoch ))
                    if save_weights_only:
                        self.model.save_weights(file_name)
                    else: self.model.save(file_name)
            callbacks.append(LambdaCallback(on_epoch_end = fun))
        if patience is not None:
            callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, restore_best_weights=True))
        if extra_callbacks is not None:
            callbacks += extra_callbacks
        return self.model.fit(train_set, epochs = epochs,validation_data = val_set,callbacks=callbacks)

    def load(self, path, rewrite = False):
        loaded_model = models.load_model(path)
        if rewrite:
            self.model = loaded_model
        return loaded_model

    def evaluate(self, eval_set, gen_cnt = 0, alpha = 0.6, sigma_noise = 16, sigma_blur = 1.5, sigma_sharppen = 64.0 ):
        y_true = []
        y_pred = []
        max_n = float('inf')
        cur_index = 0
        if hasattr(eval_set, "samples"):
            max_n = eval_set.samples
        for imgs, labels in eval_set:
            if cur_index >= max_n:
                break
            print(cur_index)
            if len(imgs.shape) < 4:
                imgs = np.expand_dims(imgs, 0)
            if hasattr(labels, "__len__") == False:
                labels = np.expand_dims(labels, axis = 0)
            if len(y_true) == 0:
                y_true = labels
            else: y_true = np.concatenate((y_true, labels))
            temp_y_pred = self.predict(imgs, gen_cnt = gen_cnt, alpha = alpha, sigma_noise = sigma_noise, sigma_blur = sigma_blur, sigma_sharppen = sigma_sharppen)
            temp_y_pred = temp_y_pred.reshape((-1))
            if len(y_pred) == 0:
                y_pred = temp_y_pred
            else: y_pred = np.concatenate((y_pred, temp_y_pred))
            cur_index += 1
        val_loss = BinaryCrossentropy()(y_true,y_pred).numpy()
        acc = Accuracy()
        acc.update_state(y_true, (y_pred >= 0.5).astype('float32'))
        val_acc = acc.result().numpy()
        return (val_loss, val_acc)

    def predict(self,imgs, gen_cnt = 0, alpha = 0.6, sigma_noise = 16, sigma_blur = 1.5, sigma_sharppen = 64.0 ):
        predv = []
        for img in imgs:
            if len(img.shape) < 4:
                img = np.expand_dims(img, axis=0)
            imgAugm = Augmentation(sigma_noise = sigma_noise, sigma_blur = sigma_blur, sigma_sharppen = sigma_sharppen)
            p_orig = self.model.predict(img)
            p_aug = np.copy(p_orig)
            for _ in range(gen_cnt):
                p_aug += self.model.predict(np.expand_dims(imgAugm(img[0] * 255.0),0) / 255.0)
            val = alpha * p_orig + (1.0 - alpha) * p_aug / (gen_cnt + 1.0)
            if len(predv) == 0:
                predv = val
            else: predv = np.vstack((predv,val))
        return predv



