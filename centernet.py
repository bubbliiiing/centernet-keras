import colorsys
import os
import time

import numpy as np
from keras import backend as K
from PIL import ImageDraw, ImageFont

from nets.centernet import centernet
from utils.utils import cvtColor, preprocess_input, resize_image, get_classes
from utils.utils_bbox import BBoxUtility

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、classes_path和backbone
#   都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class CenterNet(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'model_data/centernet_resnet50_voc.h5',
        "classes_path"      : 'model_data/voc_classes.txt',
        #--------------------------------------------------------------------------#
        #   输入图片的大小
        #--------------------------------------------------------------------------#
        "input_shape"       : [512, 512],
        #--------------------------------------------------------------------------#
        #   用于选择所使用的模型的主干
        #   resnet50, hourglass
        #--------------------------------------------------------------------------#
        "backbone"          : 'resnet50',
        #--------------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #--------------------------------------------------------------------------#
        "confidence"        : 0.3,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #--------------------------------------------------------------------------#
        #   是否进行非极大抑制，可以根据检测效果自行选择
        #   backbone为resnet50时建议设置为True、backbone为hourglass时建议设置为False
        #--------------------------------------------------------------------------#
        "nms"               : True,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化centernet
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   计算总的类的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility(nms_thresh=self.nms_iou)
        self.generate()

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        #----------------------------------------#
        #   创建centernet模型
        #----------------------------------------#
        self.centernet = centernet([self.input_shape[0], self.input_shape[1], 3], num_classes=self.num_classes, backbone=self.backbone, mode='predict')
        self.centernet.load_weights(self.model_path, by_name=True)
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, crop = False):
        #---------------------------------------------------#
        #   获得输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        #---------------------------------------------------------#
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        outputs = self.centernet.predict(image_data)
        #--------------------------------------------------------------------------------------------#
        #   centernet后处理的过程，包括门限判断和传统非极大抑制。
        #   对于centernet网络来讲，确立中心非常重要。对于大目标而言，会存在许多的局部信息。
        #   此时大目标中心点比较难以确定。使用最大池化的非极大抑制方法无法去除局部框
        #   这里面存在传统的nms处理方法，可以选择关闭和开启。
        #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
        #--------------------------------------------------------------------------------------------#
        results = self.bbox_util.postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, confidence=self.confidence)

        #--------------------------------------#
        #   如果没有检测到物体，则返回原图
        #--------------------------------------#
        if results[0] is None:
            return image

        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]

        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   获得输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        #---------------------------------------------------------#
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        outputs    = self.centernet.predict(image_data)
        #--------------------------------------------------------------------------------------------#
        #   centernet后处理的过程，包括门限判断和传统非极大抑制。
        #   对于centernet网络来讲，确立中心非常重要。对于大目标而言，会存在许多的局部信息。
        #   此时大目标中心点比较难以确定。使用最大池化的非极大抑制方法无法去除局部框
        #   这里面存在传统的nms处理方法，可以选择关闭和开启。
        #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
        #--------------------------------------------------------------------------------------------#
        results = self.bbox_util.postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, confidence=self.confidence)

        t1 = time.time()
        for _ in range(test_interval):
            outputs    = self.centernet.predict(image_data)
            #--------------------------------------------------------------------------------------------#
            #   centernet后处理的过程，包括门限判断和传统非极大抑制。
            #   对于centernet网络来讲，确立中心非常重要。对于大目标而言，会存在许多的局部信息。
            #   此时大目标中心点比较难以确定。使用最大池化的非极大抑制方法无法去除局部框
            #   这里面存在传统的nms处理方法，可以选择关闭和开启。
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            #--------------------------------------------------------------------------------------------#
            results = self.bbox_util.postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, confidence=self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        #---------------------------------------------------#
        #   获得输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        #---------------------------------------------------------#
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        outputs    = self.centernet.predict(image_data)
        #--------------------------------------------------------------------------------------------#
        #   centernet后处理的过程，包括门限判断和传统非极大抑制。
        #   对于centernet网络来讲，确立中心非常重要。对于大目标而言，会存在许多的局部信息。
        #   此时大目标中心点比较难以确定。使用最大池化的非极大抑制方法无法去除局部框
        #   这里面存在传统的nms处理方法，可以选择关闭和开启。
        #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
        #--------------------------------------------------------------------------------------------#
        results = self.bbox_util.postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, confidence=self.confidence)

        #--------------------------------------#
        #   如果没有检测到物体，则返回原图
        #--------------------------------------#
        if results[0] is None:
            return 

        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])
            
            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
