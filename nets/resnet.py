#-------------------------------------------------------------#
#   ResNet50的网络部分
#-------------------------------------------------------------#
from __future__ import print_function

import numpy as np
import keras.backend as K
from keras import layers

from keras.layers import Input
from keras.layers import Activation,BatchNormalization,Flatten
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D,Dropout,Conv2DTranspose
from keras.models import Model

from keras.preprocessing import image
from keras.regularizers import l2
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(inputs):
    # 512x512x3
    x = ZeroPadding2D((3, 3))(inputs)
    # 256,256,64
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    # 256,256,64 -> 128,128,64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # 128,128,64 -> 128,128,256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 128,128,256 -> 64,64,512
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 64,64,512 -> 32,32,1024
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # 32,32,1024 -> 16,16,2048
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    return x

def centernet_head(x,num_classes):
    x = Dropout(rate=0.5)(x)
    #-------------------------------#
    #   解码器
    #-------------------------------#
    num_filters = 256
    # 16, 16, 2048  ->  32, 32, 256 -> 64, 64, 128 -> 128, 128, 64
    for i in range(3):
        # 进行上采样
        x = Conv2DTranspose(num_filters // pow(2, i), (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    # 最终获得128,128,64的特征层
    # hm header
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg header
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)
    return y1, y2, y3

if __name__ == "__main__":
    inputs = Input(shape=(512, 512, 3))
    model = ResNet50(inputs)
    model.summary()