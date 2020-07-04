from basemodel import BaseModel

from customlayer import ConvBatchNorm
from customlayer import DepthWiseConvBatchNorm
from customlayer import CustomInceptionFactory
from customlayer import InceptionType


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

class PredefinedModel(BaseModel):
    def __init__(self, model, top_layers = 1, intercept_module = BaseModel()):
        super(PredefinedModel,self).__init__()
        if intercept_module is not None:
            for key, val in intercept_module.__dict__.items():
                if hasattr(self, key):
                    setattr(self, key,val)
        self.top_layers = top_layers
        self.pred_model = model

    def getDescription(self):
        id = self.pred_model.name + "_TL{}" + "_" + super(PredefinedModel,self).getDescription()
        return id.format(self.top_layers)

    def build_core_part_of_model(self, model):
        for m in self.pred_model.layers:
            m.trainable = True
        for i in range(len(self.pred_model.layers) - self.top_layers):
            self.pred_model.layers[i].trainable = False
        model.add(self.pred_model)
        return model