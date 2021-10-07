#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.centernet import centernet

if __name__ == "__main__":
    model = centernet([512, 512, 3], 20, backbone='resnet50')
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i, layer.name)