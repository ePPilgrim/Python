from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow import function 


class DepthWiseConvBatchNorm(layers.Layer):
    def __init__(self, filters, kernel_size = (3,3), strides = (1,1), padding='same'):
        super(DepthWiseConvBatchNorm,self).__init__()
        self.Layer1 = DepthwiseConv2D(kernel_size, strides, padding)
        self.Layer2 = BatchNormalization()
        self.Layer3 = ReLU()

        self.Layer11 = Conv2D(filters,(1,1),strides,padding)
        self.Layer12 = BatchNormalization()
        self.Layer13 = ReLU()

    @function   
    def call(self, inputs, training = None):
        x = self.Layer1(inputs)
        x = self.Layer2(x,training = training)
        x = self.Layer3(x)
        
        x = self.Layer11(x)
        x = self.Layer12(x,training = training)
        x = self.Layer13(x)
        return x

class ConvBatchNorm(layers.Layer):
    def __init__(self,filters, kernel_size = (3,3), strides=(1, 1), padding='same') :
        super(ConvBatchNorm,self).__init__()
        self.Layer1 = Conv2D(filters,kernel_size,strides,padding)
        self.Layer2 = BatchNormalization()
        self.Layer3 = ReLU()

    @function    
    def call(self,inputs,training = None):
        x = self.Layer1(inputs)
        x = self.Layer2(x,training = training)
        x = self.Layer3(x)
        return x
    
class InceptionV1(layers.Layer):
    def __init__(self, filters1, filters31, filters33, filters51, filters55, filtersPool):
        super(InceptionV1, self).__init__()
        self.Layer11 = ConvBatchNorm(filters1, (1,1))
        
        self.Layer12 = ConvBatchNorm(filters31, (1,1))
        self.Layer22 = ConvBatchNorm(filters33, (3,3))
        
        self.Layer13 = ConvBatchNorm(filters51, (1,1))
        self.Layer23 = ConvBatchNorm(filters55,(5,5))
        
        self.Layer14 = MaxPool2D(pool_size=(3, 3), strides=(1,1), padding = "same")      
        self.Layer24 = ConvBatchNorm(filtersPool, (1,1))
        
        self.Layer3 = Concatenate()
        self.Layer4 = ReLU()

    @function  
    def call(self, inputs, training = None):
        x1 = self.Layer11(inputs,training)
        x2 = self.Layer12(inputs,training)
        x2 = self.Layer22(x2,training)
        x3 = self.Layer13(inputs,training)
        x3 = self.Layer23(x3, training)
        x4 = self.Layer14(inputs)
        x4 = self.Layer24(x4, training)
        x5 = self.Layer3([x1,x2,x3,x4])
        x5 = self.Layer4(x5)
        return x5

class InceptionV2(layers.Layer):
    def __init__(self, filters1, filters31, filters33, filters51, filters55, filtersPool):
        super(InceptionV2, self).__init__()
        self.Layer11 = ConvBatchNorm(filters1, (1,1))
        
        self.Layer12 = ConvBatchNorm(filters31, (1,1))
        self.Layer22 = ConvBatchNorm(filters33, (3,1))
        self.Layer32 = ConvBatchNorm(filters33, (1,3))
        
        self.Layer13 = ConvBatchNorm(filters51, (1,1))
        self.Layer23 = ConvBatchNorm(filters55,(5,1))
        self.Layer33 = ConvBatchNorm(filters55,(1,5))
        
        self.Layer14 = MaxPool2D(pool_size=(3, 3), strides=(1,1), padding = "same")
        self.Layer24 = ConvBatchNorm(filtersPool, (1,1))
        
        self.Layer4 = Concatenate()
        self.Layer5 = ReLU()

    @function   
    def call(self, inputs, training = None):
        x1 = self.Layer11(inputs,training)
        
        x2 = self.Layer12(inputs,training)
        x2 = self.Layer22(x2,training)
        x2 = self.Layer32(x2,training)
        
        x3 = self.Layer13(inputs,training)
        x3 = self.Layer23(x3, training)
        x3 = self.Layer33(x3, training)
        
        x4 = self.Layer14(inputs)
        x4 = self.Layer24(x4, training)
        
        x5 = self.Layer4([x1,x2,x3,x4])
        x5 = self.Layer5(x5)
        return x5

class InceptionType():
    Type1 = 0
    Type2 = 1

class CustomInceptionFactory(object):
    def __init__(self, r = 0.5):
        self.r = r
            
    def _init_number_of_filters(self, filters):
        self.f1 = int(self.r * filters  * 0.25)
        self.f31 = int(self.r * filters)
        self.f33 = self.f31
        self.f51 = int(self.r * filters * 0.15)
        self.f55 = self.f51
        self.fPool = filters - (self.f1 + self.f33 + self.f55)
          
    def Create(self,filters = 32, IncType = InceptionType.Type1):
        self._init_number_of_filters(filters)
        if IncType == InceptionType.Type1:
            return InceptionV1(self.f1, self.f31, self.f33, self.f51, self.f55, self.fPool)
        if IncType == InceptionType.Type2:
            return InceptionV2(self.f1, self.f31, self.f33, self.f51, self.f55, self.fPool)
        return None




