
import os
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import sys
import IPython
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard

import kerastuner as kt

absProjectDir = Path(os.getcwd()).resolve().parents[0]
projectDir = os.path.relpath(absProjectDir,os.curdir)
load_dotenv(find_dotenv())

trainDfPath = os.path.join(projectDir, os.environ.get("REF_PROC_TRAIN_DF"))
testDfPath = os.path.join(projectDir, os.environ.get("REF_PROC_TEST_DF"))
testOrigDir = os.path.join(projectDir, os.environ.get("PROC_TEST_ORIG_DIR"))
testAugmDir = os.path.join(projectDir, os.environ.get("PROC_TEST_AUG_DIR"))
trainOrigDir = os.path.join(projectDir, os.environ.get("PROC_TRAIN_ORIG_DIR"))
trainAugmDir = os.path.join(projectDir, os.environ.get("PROC_TRAIN_AUG_DIR"))
testRootDir = os.path.commonpath([testOrigDir, testAugmDir])
trainRootDir = os.path.commonpath([trainOrigDir, trainAugmDir])
modelDir = os.path.join(projectDir, os.environ.get("MODEL_DIR"))

sys.path.append(os.path.join(projectDir,'src/data'))
sys.path.append(os.path.join(projectDir,'src/models'))
sys.path.append(os.path.join(projectDir,'src/visualization'))
import imggen as imgen
from customlayer import DepthWiseConvBatchNorm
from customlayer import ConvBatchNorm
from customlayer import InceptionType
from customlayer import CustomInceptionFactory

from basemodel import BaseModel

from models import DepthWiseConvModel1
from models import ConvModel1
from models import InterceptionModelV1
from models import InterceptionModelV2
from models import StemDecorator
from models import StemDecoratorConv
from models import StemDecoratorDepthWiseConv
from models import PredefinedModel

from visualize import VisualizeIntermedianActivation #(layer_cnt, model, img)
from visualize import GetActivatedRegion #(model, layer_name, img, activator = 0)
from visualize import CAM

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import Xception
