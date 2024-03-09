import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()
#We have 60000 training images and 10000 testing images.
#The x is the image, and the y is the answer i.e a numerical value between 0 and 9 that matches the image
print(x_train.shape)
print(y_train.shape)

#We need to flatten it so we can only have one dimension for each picture, by reshaping
# -1 keeps the dimension the same
# asfloat(32) because the numbers are 64 bits, and we need less computational time
# also the greyscale is given between 0 and 255, we need to normalize it into between 0 and 1
x_train= x_train.reshape(-1,28*28).astype("float32")/255.0
x_test= x_test.reshape(-1,28*28).astype("float32")/255.0

#Tensor flow automatically turns these arrays into tensors auutomatically

#We're going to use sequential API (convienient not flexible)
#One input one output

model = keras.Sequential(
    [
        #We do this so we can get info on the model at the end
        keras.Input(shape=(28*28)),
        #Setting up the layers:
        layers.Dense(512,activation='relu'),
        layers.Dense(256,activation = 'relu'),
        layers.Dense(10), #No activation function because we are going to do sth in the cost function later
    ]
)

#We usually use model.summary as a common debugging tool
#Which is why we sometimes do sequential and add layers line by line
#And we can take a look at the summary in between layers
print(model.summary())

#Now if we want to do a functional API
inputs = keras.Input(shape=(28*28))
x=layers.Dense(512,activation='relu')(inputs)
y=layers.Dense(256,activation = 'relu')(x)
outputs=layers.Dense(10,activation = 'softmax')(y)
#Remember if we were to specifiy softmax then from_logits would have to be set as false
#Which is default
#Now we need to redefine model
model= keras.Model(inputs=inputs, outputs=outputs)
#In this step, I want to remind that layers for a certain model are stored in a list
#So if we want out output to be for example the second to last layer we do sth like this
#outputs=[model.layers[-2].output]
#We can also name layers and use get_layer

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #from_logits sends it through soft max first then categorical cross entropy
    #Softmax outputs a probability distribution i.e this image is 99% a 1 and 0.02% a 7 and so on...
    #So since we don't want to do one hot encoding then we should use sparse categorical cross entropy
    # But basically it's doing the same thing, by setting the y_train of that specific example as the digit.
    #Not too deep of an understanding, but it's the same loss function same rules, just no one-hot encoding
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
    #Adam is an optimizer function, that basically takes the loss functions output and minimizes it
    #There are a gazillion optimizers, and for more complex networks, optimizers are custom made
    #but for most purposes the adam works well.

)
#Up until now we are just making the model, we never actualy give it any data to train or test
#This is what we are doing now
model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)
model.evaluate(x_test,y_test,batch_size=32,verbose=2)

