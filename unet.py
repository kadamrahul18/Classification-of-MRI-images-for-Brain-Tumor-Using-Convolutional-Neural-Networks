import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam


def unet(input_size = (256, 256, 1)):

    initializer = 'he_normal'

    inputs = Input(shape=input_size)                                   #(240,240,1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(inputs)             #(240,240,32)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv1)              #(240,240,32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)                      #(120,120,32)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(pool1)              #(120,120,64)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv2)              #(120,120,64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)                      #(60,60,64)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(pool2)              #(60,60,128)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv3)              #(60,60,128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)                      #(30,30,128)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(pool3)              #(30,30,256)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv4)              #(30,30,256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)                      #(15,15,256)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(pool4)              #(15,15,512)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv5)              #(15,15,512)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same',  #(30,30,256)
                kernel_initializer=initializer)(conv5),conv4], axis=3) #(30,30,512)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(up6)                #(30,30,256)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv6)              #(30,30,256)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',  #(60,60,128)
                kernel_initializer=initializer)(conv6),conv3], axis=3) #(60,60,256)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(up7)                #(60,60,128)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv7)              #(60,60,128)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2),padding='same',    #(120,120,64)
                kernel_initializer=initializer)(conv7),conv2], axis=3) #(120,120,128)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(up8)                #(120,120,64)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',
                    kernel_initializer=initializer)(conv8)             #(120,120,64)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', #(240,240,32)
                kernel_initializer=initializer)(conv8),conv1], axis=3)  #(240,240,64)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(up9)                 #(240,240,32)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv9)               #(240,240,32)

    conv10 = Conv2D(4, (1, 1), activation='relu',
                    kernel_initializer=initializer)(conv9)             #(240,240,4)
    conv10 = Activation('softmax')(conv10)                             #(240,240,4)

    model = tf.keras.Model(inputs = [inputs], outputs = [conv10])

    adam = Adam(learning_rate = 1e-4)
    model.compile(optimizer= adam, loss = tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model
