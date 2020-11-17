import colorsys
import copy
import math
import os
import pickle

import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from PIL import Image

from centernet import CenterNet
from nets.centernet import centernet
from utils.utils import centernet_correct_boxes, letterbox_image, nms
from tqdm import tqdm


def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std
    
class mAP_CenterNet(CenterNet):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        f = open("./input/detection-results/"+image_id+".txt","w") 
        self.confidence = 0.01
        self.nms_threhold = 0.5

        image_shape = np.array(np.shape(image)[0:2])
        crop_img = letterbox_image(image, [self.input_shape[0],self.input_shape[1]])
        # 将RGB转化成BGR，这是因为原始的centernet_hourglass权值是使用BGR通道的图片训练的
        photo = np.array(crop_img,dtype = np.float32)[:,:,::-1]

        # 图片预处理，归一化
        photo = np.reshape(preprocess_image(photo),[1,self.input_shape[0],self.input_shape[1],self.input_shape[2]])
        preds = self.centernet.predict(photo)
        
        if self.nms:
            preds = np.array(nms(preds,self.nms_threhold))

        if len(preds[0])<=0:
            return image

        preds[0][:,0:4] = preds[0][:,0:4]/(self.input_shape[0]/4)
        
        # 筛选出其中得分高于confidence的框
        det_label = preds[0][:, -1]
        det_conf = preds[0][:, -2]
        det_xmin, det_ymin, det_xmax, det_ymax = preds[0][:, 0], preds[0][:, 1], preds[0][:, 2], preds[0][:, 3]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices],-1),np.expand_dims(det_ymin[top_indices],-1),np.expand_dims(det_xmax[top_indices],-1),np.expand_dims(det_ymax[top_indices],-1)
        
        # 去掉灰条
        boxes = centernet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.input_shape[0],self.input_shape[1]]),image_shape)

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = str(top_conf[i])

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

centernet = mAP_CenterNet()
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

for image_id in tqdm(image_ids):
    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    # image.save("./input/images-optional/"+image_id+".jpg")
    centernet.detect_image(image_id,image)
    
print("Conversion completed!")
