import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Conv2D, MaxPool2D, Flatten, Dense
from train import *

    
class PAL_model(keras.Model):
    
    def __init__(self, n_PAL_layer, weights_path, backbone = 'resnet50', ):
        
        super().__init__()
        self.n_PAL_layer = n_PAL_layer
        self.weights_path = weights_path
        # Backbone CNN model choice
        if backbone == 'vgg16':
        
            self.backbone = keras.applications.ResNet50()
        
        # ResNet50 as default CNN backbone otherwise
        else:
            
            self.backbone = keras.applications.VGG16()

        # build model with an input shape of 224,224,3 (image size)
        self.build((None, 224, 224, 3))

    def call(self, inputs):
            
        return self.backbone(inputs)
            
    
    def train_step(self, data):

        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
    
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
            
        
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


