from nets.centernet import centernet

model = centernet([512,512,3], 20, backbone='resnet50')
model.summary()

for i,layer in enumerate(model.layers):
    print(i,layer.name)