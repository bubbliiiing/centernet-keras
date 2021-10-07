import numpy as np
import keras.backend as K
import tensorflow as tf

class BBoxUtility(object):
    def __init__(self, nms_thresh=0.45, top_k=300):
        self._nms_thresh    = nms_thresh
        self._top_k         = top_k
        self.boxes          = K.placeholder(dtype='float32', shape=(None, 4))
        self.scores         = K.placeholder(dtype='float32', shape=(None,))
        self.nms            = tf.image.non_max_suppression(self.boxes, self.scores, self._top_k, iou_threshold=self._nms_thresh)
        self.sess           = K.get_session()

    def bbox_iou(self, b1, b2):
        b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        
        inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                    np.maximum(inter_rect_y2 - inter_rect_y1, 0)
        
        area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
        
        iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)
        return iou

    def centernet_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def postprocess(self, prediction, nms, image_shape, input_shape, letterbox_image, confidence=0.5):
        results = [None for _ in range(len(prediction))]
    
        #----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        #----------------------------------------------------------------------------------------------------------------#
        for i in range(len(prediction)):
            #----------------------------------------------------------#
            #   将预测结果调整成小数的形式 0-1
            #----------------------------------------------------------#
            detections              = prediction[i]
            detections[:, [0, 2]]   = detections[:, [0, 2]] / (input_shape[1] / 4)
            detections[:, [1, 3]]   = detections[:, [1, 3]] / (input_shape[0] / 4)

            #----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            #----------------------------------------------------------#
            conf_mask   = detections[:, 4] >= confidence
            detections  = detections[conf_mask]
            
            unique_labels   = np.unique(detections[:, -1])
            #-------------------------------------------------------------------#
            #   对种类进行循环，
            #   非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框，
            #   对种类进行循环可以帮助我们对每一个类分别进行非极大抑制。
            #-------------------------------------------------------------------#
            for c in unique_labels:
                #------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                #------------------------------------------#
                detections_class = detections[detections[:, -1] == c]
                if nms:
                    #-----------------------------------------#
                    #   取出得分高于confidence的框
                    #-----------------------------------------#
                    boxes_to_process = detections_class[:, :4]
                    confs_to_process = detections_class[:, 4]
                    #-----------------------------------------#
                    #   进行iou的非极大抑制
                    #-----------------------------------------#
                    idx             = self.sess.run(self.nms, feed_dict={self.boxes: boxes_to_process, self.scores: confs_to_process})

                    max_detections  = detections_class[idx]
                    
                    # #------------------------------------------#
                    # #   非官方的实现部分
                    # #   获得某一类得分筛选后全部的预测结果
                    # #------------------------------------------#
                    # detections_class    = detections[detections[:, -1] == c]
                    # scores              = detections_class[:, 4]
                    # #------------------------------------------#
                    # #   根据得分对该种类进行从大到小排序。
                    # #------------------------------------------#
                    # arg_sort            = np.argsort(scores)[::-1]
                    # detections_class    = detections_class[arg_sort]
                    # max_detections = []
                    # while np.shape(detections_class)[0]>0:
                    #     #-------------------------------------------------------------------------------------#
                    #     #   每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
                    #     #-------------------------------------------------------------------------------------#
                    #     max_detections.append(detections_class[0])
                    #     if len(detections_class) == 1:
                    #         break
                    #     ious             = self.bbox_iou(max_detections[-1], detections_class[1:])
                    #     detections_class = detections_class[1:][ious < self._nms_thresh]
                else:
                    max_detections  = detections_class
                results[i]      = max_detections if results[i] is None else np.concatenate((results[i], max_detections), axis = 0)

            if results[i] is not None:
                box_xy, box_wh      = (results[i][:, 0:2] + results[i][:, 2:4])/2, results[i][:, 2:4] - results[i][:, 0:2]
                results[i][:, :4]   = self.centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return results