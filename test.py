#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
from nets.centernet import centernet

if __name__ == "__main__":
    model = centernet([512,512,3], 20, backbone='resnet50')
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i, layer.name)