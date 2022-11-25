import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Conv2D, MaxPool2D, Flatten, Dense
from train import *



def create_prior_heatmap(sigma):
    pass
    
    model = keras.Sequential()
    
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=2, activation="softmax"))
    
    return model

    
class PAL_model(keras.model):
    
    def __init__(self, n_PAL_layer, backbone):
        
        super.__init__()
        self.backbone = None # CNN backbone to use
        self.n_PAL_layer = n_PAL_layer
        self.model = keras.Sequential()
    

    def call(self, inputs):
        
        # VGG16 Backbone CNN model if used in argument
        if self.backbone == 'vgg16':
        
            self.backbone = keras.applications.ResNet50(input_tensor = inputs)
        
        # ResNet50 default CNN otherwise
        else:
            
            self.backbone = keras.applications.VGG16(input_tensor = inputs)
            
    
    def train_step(self, data):
        
        x, y = data
        
        
        with tf.GradientTape() as tape:
            
        
    def test_step(self, data):
        
        x, y = data
        
        
        
# Changing grad behavior to input*grad, thus getting attribution of a layer
class AttributionLayer(Conv2D):
    
    
    def __init__(self):
        super(Attribution, self).__init__()
        
    # Calling gradient methods here ???
    def call(self,inputs):
        return grad_input(inputs, output_vector)

class VGG16Attribution(keras.Model):

  def __init__(self, n_PAL_layer):
      
    super(PALVGG16, self).__init__()
    self.n_PAL_layer = n_PAL_layer
    self.model = keras.Model()

    self.conv1 = (Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    self.conv2 = (Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

    self.mp1 = (MaxPool2D(pool_size=(2,2),strides=(2,2)))

    self.conv3 = (Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    self.conv4 = (Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

    self.mp2 = (MaxPool2D(pool_size=(2,2),strides=(2,2)))

    self.conv5=(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    self.conv6=(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    self.conv7=(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

    self.mp3 = (MaxPool2D(pool_size=(2,2),strides=(2,2)))

    self.conv8=(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    self.conv9=(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    self.conv10=(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

    self.mp4 = (MaxPool2D(pool_size=(2,2),strides=(2,2)))

    self.conv11=(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    self.conv12=(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    self.conv13=(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

    self.mp5 = (MaxPool2D(pool_size=(2,2),strides=(2,2)))

    self.flat = (Flatten())
    self.dense1 = (Dense(units=4096,activation="relu"))
    self.dense2 = (Dense(units=4096,activation="relu"))
    self.dense3 = (Dense(units=2, activation="softmax"))


  def call(self,inputs):

    x = self.conv1(inputs)
    x = self.conv2(x)

    x = self.mp1(x)

    x = self.conv3(x)
    x = self.conv4(x)

    x = self.mp2(x)

    x = self.conv5(x)
    x = self.conv6(x)
    x = self.conv7(x)

    x = self.mp4(x)

    x = self.conv8(x)
    x = self.conv9(x)
    x = self.conv10(x)

    x = self.mp5(x)

    x = self.conv11(x)
    x = self.conv12(x)
    x = self.conv13(x)

    x = self.flat(x)
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)
    return x
