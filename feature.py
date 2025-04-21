#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
import torch.nn as nn
from thop import clever_format, profile
from torchsummary import summary
import matplotlib.pyplot as plt
from torchvision import models, transforms
import imageio
import cv2
import numpy as np
import math
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import os
from yolo_test import YOLO
from PIL import Image

# 导入数据
def get_image_info(image_dir):
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    image_info = Image.open(image_dir).convert('RGB')
    # 数据预处理方法
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_info = image_transform(image_info)
    image_info = image_info.unsqueeze(0)
    return image_info
 
# 获取第k层的特征图，这里应该是看各层情况
# def get_k_layer_feature_map(feature_extractor, k, x):
#     for p in feature_extractor.parameters():
#             p.requires_grad = False
#         # Define which layers you are going to extract
#         = nn.Sequential(*list(feature_extractor.children())[:4])
#     with torch.no_grad():
#         for index,layer in enumerate(feature_extractor):
#             x = layer(x)
#             if k == index:
#                 return x
 
class FeatureExtractor(nn.Module):
    def __init__(self, net):
        super(FeatureExtractor, self).__init__()
        # self.net = models.googlenet(pretrained=True)
        # 查看检测头后三层特征
        # self.net = net
        # 查看backbone后三层特征
        self.net = net.backbone
        # If you treat GooLeNet as a fixed feature extractor, disable the gradients and save some memory
        for p in self.net.parameters():
            p.requires_grad = False
        # Define which layers you are going to extract
        self.features = nn.Sequential(*list(self.net.children())[:4])
        # self.features = nn.Sequential(*list(self.net.forward())[:4])
        # self.features = nn.Sequential(*list(self.net.backbone.children())[:4])

    def forward(self, x):
        # 只看检测头就一个输出
        # if torch.cuda.is_available():
        #     device = torch.device("cuda:0")
        #     cudnn.benchmark = True   # cudnn auto-tuner
        # else:
        #     device = torch.device("cpu")
        # x = torch.autograd.Variable(x).to(device)
        # x = self.features(x)
        x = self.net(x)
        # x = self.features.forward(x)
        return x
        # det, fpn = self.features(x)
        # return det, fpn

#  可视化特征图
def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    # transform=T.ToPILImage()
    for index in range(1, feature_map_num+1):
        # plt.subplot(row_num, row_num, index)
        plt.subplot(int(row_num),int(row_num), index)
        plt.imshow(feature_map[index-1], cmap='gray')
        # plt.imshow(feature_map[index-1], cmap='Accent')
        plt.axis('off')
        # feature_map[index-1] = feature_map[index-1]*255
        # feature_map[index-1] = feature_map[index-1].astype(np.uint8)
        # feature_map[index-1] = Image.fromarray(feature_map[index-1])
        # # feature_map[index-1]=transform(feature_map[index-1])
        # # imageio.imsave(str(index)+".png", feature_map[index-1])
        # imageio.imwrite(str(index)+".png", feature_map[index-1])
        feature_map_int = ((feature_map[index-1] - feature_map[index-1].min()) / (feature_map[index-1].max() - feature_map[index-1].min()) * 255).astype(np.uint8)
        feature_map_image = Image.fromarray(feature_map_int)
        image_path = "./features/" + str(index) + ".png"
        imageio.imsave(image_path, feature_map_image)
    plt.show()



if __name__ == "__main__":
    #--------------------------------------------#    
    # 需要使用device来指定网络在GPU还是CPU运行
    # device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # m       = YoloBody(num_classes, phi).to(device)
    # summary(m, (3, input_shape[0], input_shape[1]))
    yolo = YOLO()

    image_dir = r"test333.jpg"

    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    # mode = "dir_predict"        # dir_predict, predict
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    # crop            = False
    # count           = False
    # img = "aaa.jpg"  # 输入图像路径

    # k = 1
    # 导入Pytorch封装的AlexNet网络模型
    # model = models.alexnet(pretrained=True)
    # 是否使用gpu运算
    # use_gpu = torch.cuda.is_available()
    # use_gpu =False
    # 读取图像信息
    image_info = get_image_info(image_dir)
    image_info = image_info.cuda()

    # 加载网络，现在得到的是检测结果
    network = yolo.net
    extractor = FeatureExtractor(network)
    # 这样得到的是检测头三层的特征图
    feature_map = extractor(image_info)
    feature_map1 = feature_map[2]
    show_feature_map(feature_map1)

    # detection_map, fpn_output = extractor(image_info)
    # fpn_output1 = fpn_output[0]

    # show_feature_map(fpn_output1)

    # image = Image.open(img)

    # r_image = yolo.detect_image(image, crop = crop, count=count)
    


