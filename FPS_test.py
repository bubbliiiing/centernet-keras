import time

import numpy as np
from PIL import Image

from centernet import CenterNet
from nets.centernet import centernet
from utils.utils import centernet_correct_boxes, letterbox_image, nms

'''
该FPS测试不包括前处理（归一化与resize部分）、绘图。
包括的内容为：网络推理、得分门限筛选、非极大抑制。
使用'img/street.jpg'图片进行测试，该测试方法参考库https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

video.py里面测试的FPS会低于该FPS，因为摄像头的读取频率有限，而且处理过程包含了前处理和绘图部分。
'''

def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std

class FPS_CenterNet(CenterNet):
    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        crop_img = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        #----------------------------------------------------------------------------------#
        #   将RGB转化成BGR，这是因为原始的centernet_hourglass权值是使用BGR通道的图片训练的
        #----------------------------------------------------------------------------------#
        photo = np.array(crop_img,dtype = np.float32)[:,:,::-1]
        #-----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        #-----------------------------------------------------------#
        photo = np.expand_dims(preprocess_image(photo), 0)

        preds = self.centernet.predict(photo)
        
        if self.nms:
            preds = np.array(nms(preds, self.nms_threhold))

        if len(preds[0])>0:
            #-----------------------------------------------------------#
            #   将预测结果转换成小数的形式
            #-----------------------------------------------------------#
            preds[0][:, [0,2]] = preds[0][:, [0,2]] / (self.input_shape[1] / 4)
            preds[0][:, [1,3]] = preds[0][:, [1,3]] / (self.input_shape[0] / 4)
            
            det_label   = preds[0][:, -1]
            det_conf    = preds[0][:, -2]
            det_xmin, det_ymin, det_xmax, det_ymax = preds[0][:, 0], preds[0][:, 1], preds[0][:, 2], preds[0][:, 3]

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
            
            boxes = centernet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.input_shape[0],self.input_shape[1]]),image_shape)

        t1 = time.time()
        for _ in range(test_interval):
            preds = self.centernet.predict(photo)
            
            if self.nms:
                preds = np.array(nms(preds, self.nms_threhold))

            if len(preds[0])>0:
                #-----------------------------------------------------------#
                #   将预测结果转换成小数的形式
                #-----------------------------------------------------------#
                preds[0][:, [0,2]] = preds[0][:, [0,2]] / (self.input_shape[1] / 4)
                preds[0][:, [1,3]] = preds[0][:, [1,3]] / (self.input_shape[0] / 4)
                
                det_label   = preds[0][:, -1]
                det_conf    = preds[0][:, -2]
                det_xmin, det_ymin, det_xmax, det_ymax = preds[0][:, 0], preds[0][:, 1], preds[0][:, 2], preds[0][:, 3]

                top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
                
                boxes = centernet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.input_shape[0],self.input_shape[1]]),image_shape)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
        
centernet = FPS_CenterNet()
test_interval = 100
img = Image.open('img/street.jpg')
tact_time = centernet.get_FPS(img, test_interval)
print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
