import keras.backend as K
import numpy as np
from keras.initializers import random_normal
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Input, UpSampling2D, ZeroPadding2D)
from keras.models import Model
from keras.regularizers import l2
from keras.utils import get_file


def conv2d(x, k, out_dim, name, stride=1):
    padding = (k - 1) // 2
    x = ZeroPadding2D(padding=padding, name=name + '.pad')(x)
    x = Conv2D(out_dim, k, strides=stride, kernel_initializer=random_normal(stddev=0.02), use_bias=False, name=name + '.conv')(x)
    x = BatchNormalization(epsilon=1e-5, name=name + '.bn')(x)
    x = Activation('relu', name=name + '.relu')(x)
    return x

def residual(x, out_dim, name, stride=1):
    #-----------------------------------#
    #   残差网络结构
    #   两个形态
    #   1、残差边有卷积，改变维度
    #   2、残差边无卷积，加大深度
    #-----------------------------------#
    shortcut = x
    num_channels = K.int_shape(shortcut)[-1]

    x = ZeroPadding2D(padding=1, name=name + '.pad1')(x)
    x = Conv2D(out_dim, 3, strides=stride, kernel_initializer=random_normal(stddev=0.02), use_bias=False, name=name + '.conv1')(x)
    x = BatchNormalization(epsilon=1e-5, name=name + '.bn1')(x)
    x = Activation('relu', name=name + '.relu1')(x)

    x = Conv2D(out_dim, 3, padding='same', kernel_initializer=random_normal(stddev=0.02), use_bias=False, name=name + '.conv2')(x)
    x = BatchNormalization(epsilon=1e-5, name=name + '.bn2')(x)

    if num_channels != out_dim or stride != 1:
        shortcut = Conv2D(out_dim, 1, strides=stride, kernel_initializer=random_normal(stddev=0.02), use_bias=False, name=name + '.shortcut.0')(
            shortcut)
        shortcut = BatchNormalization(epsilon=1e-5, name=name + '.shortcut.1')(shortcut)

    x = Add(name=name + '.add')([x, shortcut])
    x = Activation('relu', name=name + '.relu')(x)
    return x

def bottleneck_layer(x, num_channels, hgid):
    #-----------------------------------#
    #   中间的深度结构
    #-----------------------------------#
    pow_str = 'center.' * 5
    x = residual(x, num_channels, name='kps.%d.%s0' % (hgid, pow_str))
    x = residual(x, num_channels, name='kps.%d.%s1' % (hgid, pow_str))
    x = residual(x, num_channels, name='kps.%d.%s2' % (hgid, pow_str))
    x = residual(x, num_channels, name='kps.%d.%s3' % (hgid, pow_str))
    return x

def connect_left_right(left, right, num_channels, num_channels_next, name):
    # 图像上半部分的卷积
    left = residual(left, num_channels_next, name=name + 'skip.0')
    left = residual(left, num_channels_next, name=name + 'skip.1')
    # 图像右半部分的上采样
    out = residual(right, num_channels, name=name + 'out.0')
    out = residual(out, num_channels_next, name=name + 'out.1')
    out = UpSampling2D(name=name + 'out.upsampleNN')(out)
    # 利用相加进行全连接
    out = Add(name=name + 'out.add')([left, out])
    return out    

def pre(x, num_channels):
    #-----------------------------------#
    #   图片进入金字塔前的预处理
    #   一般是一次普通卷积
    #   加上残差结构
    #-----------------------------------#
    x = conv2d(x, 7, 128, name='pre.0', stride=2)
    x = residual(x, num_channels, name='pre.1', stride=2)
    return x

def left_features(bottom, hgid, dims):
    #-------------------------------------------------#
    #   进行五次下采样
    #   f1, f2, f4 , f8, f16, f32 : 1, 1/2, 1/4 1/8, 1/16, 1/32 resolution
    #   5 times reduce/increase: (256, 384, 384, 384, 512)
    #-------------------------------------------------#
    features = [bottom]
    for kk, nh in enumerate(dims):
        x = residual(features[-1], nh, name='kps.%d%s.down.0' % (hgid, str(kk)), stride=2)
        x = residual(x, nh, name='kps.%d%s.down.1' % (hgid, str(kk)))
        features.append(x)
    return features

def right_features(leftfeatures, hgid, dims):
    #-------------------------------------------------#
    #   进行五次上采样，并进行连接
    #   f1, f2, f4 , f8, f16, f32 : 1, 1/2, 1/4 1/8, 1/16, 1/32 resolution
    #   5 times reduce/increase: (256, 384, 384, 384, 512)
    #-------------------------------------------------#
    rf = bottleneck_layer(leftfeatures[-1], dims[-1], hgid)
    for kk in reversed(range(len(dims))):
        pow_str = ''
        for _ in range(kk):
            pow_str += 'center.'
        rf = connect_left_right(leftfeatures[kk], rf, dims[kk], dims[max(kk - 1, 0)], name='kps.%d.%s' % (hgid, pow_str))
    return rf


def create_heads(num_classes, rf1, hgid):
    y1 = Conv2D(256, 3, use_bias=True, kernel_initializer=random_normal(stddev=0.02), padding='same', name='hm.%d.0.conv' % hgid)(rf1)
    y1 = Activation('relu', name='hm.%d.0.relu' % hgid)(y1)
    y1 = Conv2D(num_classes, 1, use_bias=True, name='hm.%d.1' % hgid, activation = "sigmoid")(y1)

    y2 = Conv2D(256, 3, use_bias=True, kernel_initializer=random_normal(stddev=0.02), padding='same', name='wh.%d.0.conv' % hgid)(rf1)
    y2 = Activation('relu', name='wh.%d.0.relu' % hgid)(y2)
    y2 = Conv2D(2, 1, use_bias=True, name='wh.%d.1' % hgid)(y2)

    y3 = Conv2D(256, 3, use_bias=True, kernel_initializer=random_normal(stddev=0.02), padding='same', name='reg.%d.0.conv' % hgid)(rf1)
    y3 = Activation('relu', name='reg.%d.0.relu' % hgid)(y3)
    y3 = Conv2D(2, 1, use_bias=True, kernel_initializer=random_normal(stddev=0.02), name='reg.%d.1' % hgid)(y3)

    return [y1,y2,y3]

def hourglass_module(num_classes, bottom, cnv_dim, hgid, dims):
    # 左边下采样的部分
    lfs = left_features(bottom, hgid, dims)

    # 右边上采样与中间的连接部分
    rf1 = right_features(lfs, hgid, dims)
    rf1 = conv2d(rf1, 3, cnv_dim, name='cnvs.%d' % hgid)

    heads = create_heads(num_classes, rf1, hgid)
    return heads, rf1


def HourglassNetwork(inpnuts, num_stacks, num_classes, cnv_dim=256, dims=[256, 384, 384, 384, 512]):
    inter = pre(inpnuts, cnv_dim)
    outputs = []
    for i in range(num_stacks):
        prev_inter = inter
        _heads, inter = hourglass_module(num_classes, inter, cnv_dim, i, dims)
        outputs.append(_heads)
        if i < num_stacks - 1:
            inter_ = Conv2D(cnv_dim, 1, use_bias=False, kernel_initializer=random_normal(stddev=0.02), name='inter_.%d.0' % i)(prev_inter)
            inter_ = BatchNormalization(epsilon=1e-5, name='inter_.%d.1' % i)(inter_)

            cnv_ = Conv2D(cnv_dim, 1, use_bias=False, kernel_initializer=random_normal(stddev=0.02), name='cnv_.%d.0' % i)(inter)
            cnv_ = BatchNormalization(epsilon=1e-5, name='cnv_.%d.1' % i)(cnv_)

            inter = Add(name='inters.%d.inters.add' % i)([inter_, cnv_])
            inter = Activation('relu', name='inters.%d.inters.relu' % i)(inter)
            inter = residual(inter, cnv_dim, 'inters.%d' % i)
    return outputs
    
if __name__ == "__main__":
    image_input = Input(shape=(512, 512, 3))
    outputs = HourglassNetwork(image_input,2,20)
    model = Model(image_input,outputs[-1])
    model.summary()

    