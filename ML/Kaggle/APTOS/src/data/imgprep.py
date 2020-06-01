import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import imgtrans as tr

class ImagePreprocessing(object):
    def __init__(self, transforms = tr.Transformation()):
        self.transforms = transforms
        self.oimg = None
        self.timg = None
        
    def __call__(self,image):
        self.oimg = image
        self.timg = self.transforms(self.oimg)
        if self.timg is not None:
            self.timg = self.timg.astype('uint8')
        return self.timg
    
    def CompareImages(self):
        fig = plt.figure(figsize=(16,16))
        fig.add_subplot(1,2,1)
        plt.imshow(self.oimg)
        fig.add_subplot(1,2,2)
        plt.imshow(self.timg)

    def SaveModifiedImage(self, path):
        cv2.imwrite(path,self.timg)

    def LoadImage(path):
        if os.path.isfile(path):
            return cv2.imread(path)
        return None

def PreprocessImage(cropTol = 7):
    crop = tr.Cropp(cropTol)
    resize = tr.Resize((224,224))
    return ImagePreprocessing(tr.Pipe([crop, resize]))

def Augmentation(sigma = 16):
    effect = tr.Transformation()
    if np.random.randint(2) == 0:
        effect = tr.Sharppen()
    else:
        effect = tf.Blur()
        
    visEffects = tr.RandomVisualEffect()
    gaussNoise = tr.AddGauseNoise(sigma)
    return ImagePreprocessing(tr.Pipe([visEffects,gaussNoise]))




