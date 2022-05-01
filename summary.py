#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.centernet import centernet
from utils.utils import net_flops

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 20
    
    model = centernet([input_shape[0], input_shape[1], 3], num_classes, backbone='resnet50')
    #--------------------------------------------#
    #   查看网络结构网络结构
    #--------------------------------------------#
    model.summary()
    #--------------------------------------------#
    #   计算网络的FLOPS
    #--------------------------------------------#
    net_flops(model, table=False)
    
    #--------------------------------------------#
    #   获得网络每个层的名称与序号
    #--------------------------------------------#
    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
