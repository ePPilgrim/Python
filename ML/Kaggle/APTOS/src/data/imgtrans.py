'''
pixels in the image should be value [0,255]
'''
import tensorflow as tf
import numpy as np
import cv2

class Transformation(object):
    def __init__(self):
        pass
    def __call__(self,image):
        return image

class Resize(Transformation):
    '''
        method - one of the tf.image.ResizeMethod
    '''
    def __init__(self, size = None, method = tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio = False, antialias = False):
        super().__init__()
        self.method = method
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.antialias = antialias
        self.size = size
    def __call__(self,image = None):
        if self.size is None or image is None:
            return super().__call__(image)
        return tf.image.resize(image, self.size, method=self.method, 
                               preserve_aspect_ratio=self.preserve_aspect_ratio, 
                               antialias=self.antialias, name=None).numpy()
class Cropp(Transformation):
    def __init__(self, tolerance = None):
        super().__init__()
        self.tolerance = tolerance
    def __call__(self,image = None):
        if image is None or self.tolerance is None:
            return super().__call__(image)
        return self.__cropImage__(image)
    def __cropImage__(self, image):
        grayImage = tf.image.rgb_to_grayscale(image)
        mask = tf.greater(grayImage, self.tolerance)
        rows = tf.reshape(tf.reduce_any(mask,1), (-1,1))
        r_sz = tf.reduce_sum(tf.cast(rows,tf.int32));
        cols = tf.reshape(tf.reduce_any(mask,0),(1,-1))
        c_sz = tf.reduce_sum(tf.cast(cols,tf.int32))
        if c_sz == 0 or r_sz == 0:
            return image
        mask = tf.logical_and(rows,cols)
        newImage = []
        for i in range(3):
            newImage.append(tf.reshape(tf.boolean_mask(image[:,:,i],mask),(r_sz,c_sz)))
        return tf.stack(newImage,axis = 2).numpy()
    
class Sharppen(Transformation):
    def __init__(self, sigma = None):
        super().__init__()
        self.sigma = sigma
    def __call__(self, image = None):
        if image is None or self.sigma is None:
            return super().__call__(image)
        img = cv2.addWeighted ( image,4, cv2.GaussianBlur( image, (0,0) , self.sigma) ,-4 ,128)
        return img
    
class CircleMask(Transformation):
    def __init__(self):
        super().__init__()
    def __call__(self,image):
        if image is None:
            return super().__call__(image)
        height, width, depth = image.shape    
        x = int(width/2)
        y = int(height/2)
        r = np.amin((x,y))
        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
        return cv2.bitwise_and(image, image, mask=circle_img)

class AddGauseNoise(Transformation):
    def __init__(self,supperSigma = 1):
        super().__init__()
        self.SupperSigma = supperSigma
        
    def __call__(self, image):
        if image is None:
            return super().__call__(image)
        sigmav = np.random.choice(self.SupperSigma, 3)
        noise = np.random.normal(0,sigmav,image.shape)
        return np.clip(noise + image,0.,255.)

class RandomVisualEffect(Transformation):
    def __init__(self):
        super().__init__()
        self.rot = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE,cv2.ROTATE_180]
    
    def __call__(self, image):
        if image is None:
            return super().__call__(image)
        rix = np.random.choice(3,1)[0]
        image = cv2.rotate(image,self.rot[rix])
        image = tf.image.adjust_contrast(image, 1.0 + 2.0 * np.random.random())
        image = tf.image.random_hue(image,0.5)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        return image.numpy()
    
class Pipe(Transformation):
    def __init__(self,trans = None):
        super().__init__()
        self.trans = trans
    def __call__(self,image):
        if self.trans is None or image is None:
            return super().__call__(image)
        for tr in self.trans:
            image = tr(image)
        return image



