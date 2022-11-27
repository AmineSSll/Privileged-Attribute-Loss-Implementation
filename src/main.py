from model import *
import sys
import tensorflow as tf

if __name__ == '__main__':
    
    x = np.ones((224, 244, 3))
    x = tf.constant(x, dtype = np.float32)
    
    
    m = PAL_model(4, 3, backbone = 'vgg16')
    print(m.backbone.layers)