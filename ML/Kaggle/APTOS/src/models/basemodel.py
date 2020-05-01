import math
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from customlayer import ConvBatchNorm
from customlayer import DepthWiseConvBatchNorm
from customlayer import CustomInceptionFactory
from customlayer import InceptionType

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
        return self.compliteModelCreation(model)
    
class ConvModel1(BaseModel):
    def build_core_part_of_model(self,model):
        for i in range(self.conv_layer_cnt):
            model.add(ConvBatchNorm(self.filters * (2**i),(3,3)))
            model.add(MaxPool2D(pool_size=(2, 2), strides = (2,2)))
        return model
    
class DepthWiseConvModel1(BaseModel):
    def build_core_part_of_model(self,model):
        for i in range(self.conv_layer_cnt):
            model.add(DepthWiseConvBatchNorm(self.filters * (2**i),(3,3)))
            model.add(MaxPool2D(pool_size=(2, 2), strides = (2,2)))
        return model

class InterceptionModelV1(BaseModel):
    def build_core_part_of_model(self,model):
        for i in range(self.conv_layer_cnt):
            model.add(CustomInceptionFactory().Create(filters = self.filters * (2**i),
                                                     IncType = InceptionType.Type1))
            model.add(MaxPool2D(pool_size=(2, 2), strides = (2,2)))
        return model
                      
class InterceptionModelV2(BaseModel):
    def build_core_part_of_model(self,model):
        for i in range(self.conv_layer_cnt):
            model.add(CustomInceptionFactory().Create(filters = self.filters * (2**(1 + i)),
                                                             IncType = InceptionType.Type2))
            model.add(MaxPool2D(pool_size=(2, 2), strides = (2,2)))
        return model
    
class StemDecorator(BaseModel):
    def __init__(self, conv_layer_cnt = 1, intercept_module = BaseModel()):
        super(StemDecorator,self).__init__(conv_layer_cnt = conv_layer_cnt)
        self.intercept_module = intercept_module
        self.intercept_module.filters = self.filters * (2 ** self.conv_layer_cnt)
        self.intercept_module.conv_layer_cnt -= self.conv_layer_cnt
        
    def getDescription(self):
        name = "{}_DLayer{}_".format(type(self).__name__, self.conv_layer_cnt)
        return name + self.intercept_module.getDescription()
    
    def build_core_part_of_model(self,model):
        pass

class StemDecoratorConv(StemDecorator):   
    def build_core_part_of_model(self,model):
        for i in range(self.conv_layer_cnt):
            model.add(ConvBatchNorm(self.filters * (2**i),(3,3)))
            model.add(MaxPool2D(pool_size=(2, 2), strides = (2,2)))
        return self.intercept_module.build_core_part_of_model(model)
    
class StemDecoratorDepthWiseConv(StemDecorator):   
    def build_core_part_of_model(self,model):
        for i in range(self.conv_layer_cnt):
            model.add(DepthWiseConvBatchNorm(self.filters * (2**i),(3,3)))
            model.add(MaxPool2D(pool_size=(2, 2), strides = (2,2)))
        return self.intercept_module.build_core_part_of_model(model)