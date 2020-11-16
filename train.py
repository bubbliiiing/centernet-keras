from nets.center_training import Generator
from nets.centernet import centernet
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import keras
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__": 
    # 图片的大小
    input_shape = [512,512,3]
    annotation_path = '2007_train.txt'
    #-----------------------------#
    #   训练前注意修改classes_path
    #   对应的txt的内容
    #   修改成自己需要分的类
    #-----------------------------#
    classes_path = 'model_data/voc_classes.txt'
    #-----------------------------#
    #   主干特征提取网络的选择
    #   resnet50
    #   hourglass
    #-----------------------------#
    backbone = "resnet50"

    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    model = centernet(input_shape, num_classes=num_classes, backbone=backbone, mode='train')
    
    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    #------------------------------------------------------#
    model_path = r"model_data/centernet_resnet50_voc.h5"
    model.load_weights(model_path,by_name=True,skip_mismatch=True)
    model.summary()

    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # 训练参数设置
    logging = TensorBoard(log_dir="logs")
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if backbone == "resnet50":
        freeze_layer = 171
    elif backbone == "hourglass":
        freeze_layer = 624
    else:
        raise ValueError('Unsupported backbone - `{}`, Use resnet50, hourglass.'.format(backbone))

    for i in range(freeze_layer):
        model.layers[i].trainable = False

    if True:
        # 每一次训练使用多少个Batch
        Batch_size = 8
        Init_Epoch = 0
        Freeze_Epoch = 50
        Lr = 1e-3

        gen = Generator(Batch_size, lines[:num_train], lines[num_train:], input_shape, num_classes)

        model.compile(
            loss={'centernet_loss': lambda y_true, y_pred: y_pred},
            optimizer=keras.optimizers.Adam(Lr)
        )

        model.fit_generator(gen.generate(True), 
                steps_per_epoch=num_train//Batch_size,
                validation_data=gen.generate(False),
                validation_steps=num_val//Batch_size,
                epochs=Freeze_Epoch, 
                verbose=1,
                initial_epoch=Init_Epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    for i in range(freeze_layer):
        model.layers[i].trainable = True

    if True:
        # 每一次训练使用多少个Batch
        Batch_size = 4
        Freeze_Epoch = 50
        Epoch = 100
        Lr = 1e-4
        
        gen = Generator(Batch_size, lines[:num_train], lines[num_train:], input_shape, num_classes)

        model.compile(
            loss={'centernet_loss': lambda y_true, y_pred: y_pred},
            optimizer=keras.optimizers.Adam(Lr)
        )

        model.fit_generator(gen.generate(True), 
                steps_per_epoch=num_train//Batch_size,
                validation_data=gen.generate(False),
                validation_steps=num_val//Batch_size,
                epochs=Epoch, 
                verbose=1,
                initial_epoch=Freeze_Epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])