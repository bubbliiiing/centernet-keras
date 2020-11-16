from nets.resnet import ResNet50, centernet_head
from nets.hourglass import HourglassNetwork
from keras.layers import Input, Conv2DTranspose, BatchNormalization, Activation, Conv2D, Lambda, MaxPooling2D, Dropout, Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model
from keras.initializers import normal, constant, zeros
from keras.regularizers import l2
import tensorflow as tf
import keras.backend as K

from nets.center_training import loss

def nms(heat, kernel=3):
    hmax = MaxPooling2D((kernel, kernel), strides=1, padding='SAME')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat

def topk(hm, max_objects=100):
    # hm -> Hot map热力图
    # 进行热力图的非极大抑制，利用3x3的卷积对热力图进行Max筛选，找出值最大的
    hm = nms(hm)
    # (b, h * w * c)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    # 将所有结果平铺，获得(b, h * w * c)
    hm = tf.reshape(hm, (b, -1))
    # (b, k), (b, k)
    scores, indices = tf.math.top_k(hm, k=max_objects, sorted=True)

    # 计算求出网格点，类别
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys

def decode(hm, wh, reg, max_objects=100,num_classes=20):
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    # 获得batch_size
    b = tf.shape(hm)[0]
    
    # (b, h * w, 2)
    reg = tf.reshape(reg, [b, -1, 2])
    # (b, h * w, 2)
    wh = tf.reshape(wh, [b, -1, 2])
    length = tf.shape(wh)[1]

    # 找到其在1维上的索引
    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, max_objects))
    full_indices = tf.reshape(batch_idx, [-1]) * tf.to_int32(length) + tf.reshape(indices, [-1])
                    
    # 取出top_k个框对应的参数
    topk_reg = tf.gather(tf.reshape(reg, [-1,2]), full_indices)
    topk_reg = tf.reshape(topk_reg, [b, -1, 2])
    
    topk_wh = tf.gather(tf.reshape(wh, [-1,2]), full_indices)
    topk_wh = tf.reshape(topk_wh, [b, -1, 2])

    # 计算调整后的中心
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]

    # (b,k,1) (b,k,1)
    topk_x1, topk_y1 = topk_cx - topk_wh[..., 0:1] / 2, topk_cy - topk_wh[..., 1:2] / 2
    # (b,k,1) (b,k,1)
    topk_x2, topk_y2 = topk_cx + topk_wh[..., 0:1] / 2, topk_cy + topk_wh[..., 1:2] / 2
    # (b,k,1)
    scores = tf.expand_dims(scores, axis=-1)
    # (b,k,1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    # (b,k,6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)

    return detections


def centernet(input_shape, num_classes, backbone='resnet50', max_objects=100, mode="train", num_stacks=2):
    assert backbone in ['resnet50', 'hourglass']
    output_size = input_shape[0] // 4
    image_input = Input(shape=input_shape)
    hm_input = Input(shape=(output_size, output_size, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    if backbone=='resnet50':
        #-------------------------------#
        #   编码器
        #-------------------------------#
        C5 = ResNet50(image_input)

        y1, y2, y3 = centernet_head(C5,num_classes)

        if mode=="train":
            loss_ = Lambda(loss, name='centernet_loss')([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
            model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])
            return model
        else:
            detections = Lambda(lambda x: decode(*x, max_objects=max_objects,
                                                num_classes=num_classes))([y1, y2, y3])
            prediction_model = Model(inputs=image_input, outputs=detections)
            return prediction_model

    else:
        outs = HourglassNetwork(image_input,num_stacks,num_classes)

        if mode=="train":
            loss_all = []
            for out in outs:  
                y1, y2, y3 = out
                loss_ = Lambda(loss)([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
                loss_all.append(loss_)
            loss_all = Lambda(tf.reduce_mean,name='centernet_loss')(loss_all)

            model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=loss_all)
            return model
        else:
            y1, y2, y3 = outs[-1]
            detections = Lambda(lambda x: decode(*x, max_objects=max_objects,
                                                num_classes=num_classes))([y1, y2, y3])
            prediction_model = Model(inputs=image_input, outputs=[detections])
            return prediction_model
