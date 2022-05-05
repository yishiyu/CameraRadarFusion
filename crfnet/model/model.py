import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from .loss import CRFLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pytorch VGG 源码: https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html
# model_urls = {
#     "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
#     "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
#     "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
#     "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
#     "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
#     "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
#     "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
#     "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
# }

# Anchors Parameters
# 参考代码: anchor_parameter.py 46
# AnchorParameters.small
anchor_parameters = {
    'stride': [8, 16, 32, 64, 128],
    'ratios': [0.5, 1, 2.0],
    'sizes': [16, 32, 64, 128, 256],
    # array([1.       , 1.2599211, 1.587401 ])
    'scales': np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]),
    # len(ratios) * len(scales)
    'num_anchors': 9,
}


class BackBoneSubmodel(nn.Module):
    def __init__(self, opts):
        super(BackBoneSubmodel, self).__init__()
        # pytorch MaxPool2D默认是ceil_mode=False(floor模式)
        # 需要注意的是 block4 的需要用 floor 模式(22/45),block6 需要用 ceil 模式(6/11)

        self.block1_conv1 = nn.Conv2d(5, 64, kernel_size=3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.block1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block1_rad_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2_conv1 = nn.Conv2d(66, 128, kernel_size=3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2_rad_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3_conv1 = nn.Conv2d(130, 256, kernel_size=3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.block3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3_rad_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4_conv1 = nn.Conv2d(258, 512, kernel_size=3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4_rad_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block5_conv1 = nn.Conv2d(514, 512, kernel_size=3, padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.block5_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block5_rad_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block6_rad_pool = nn.MaxPool2d(
            kernel_size=2, stride=2, ceil_mode=True)
        self.block7_rad_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x 是包含图像(3通道)雷达(2通道)在内的5通道数据
        # x ==> (None, 5, 360, 640)
        # pytorch 数据是channel first的
        channel_index = 1

        # block1
        # output_radar ==> (None, 2, 180, 320)
        # output_camera ==> (None, 66, 180, 320)
        output_camera = F.relu(self.block1_conv1(x))
        output_camera = F.relu(self.block1_conv2(output_camera))
        output_camera = self.block1_pool(output_camera)
        output_radar = self.block1_rad_pool(x[:, 3:])
        output_camera = torch.cat((output_camera, output_radar), channel_index)

        # block2
        # output_radar ==> (None, 2, 90, 160)
        # output_camera ==> (None, 130, 90, 160)
        output_camera = F.relu(self.block2_conv1(output_camera))
        output_camera = F.relu(self.block2_conv2(output_camera))
        output_camera = self.block2_pool(output_camera)
        output_radar = self.block2_rad_pool(output_radar)
        output_camera = torch.cat((output_camera, output_radar), channel_index)

        # block3
        # output_radar ==> (None, 2, 45, 80)
        # output_camera ==> (None, 258, 45, 80)
        output_camera = F.relu(self.block3_conv1(output_camera))
        output_camera = F.relu(self.block3_conv2(output_camera))
        output_camera = F.relu(self.block3_conv2(output_camera))
        output_camera = self.block3_pool(output_camera)
        output_radar = self.block3_rad_pool(output_radar)
        output_camera = torch.cat((output_camera, output_radar), channel_index)
        block3_radar = output_radar
        block3_feature = output_camera

        # block4
        # output_radar ==> (None, 2, 22, 40)
        # output_camera ==> (None, 514, 22, 40)
        output_camera = F.relu(self.block4_conv1(output_camera))
        output_camera = F.relu(self.block4_conv2(output_camera))
        output_camera = F.relu(self.block4_conv2(output_camera))
        output_camera = self.block4_pool(output_camera)
        output_radar = self.block4_rad_pool(output_radar)
        output_camera = torch.cat((output_camera, output_radar), channel_index)
        block4_radar = output_radar
        block4_feature = output_camera

        # block5
        # output_radar ==> (None, 2, 11, 20)
        # output_camera ==> (None, 514, 11, 20)
        output_camera = F.relu(self.block5_conv1(output_camera))
        output_camera = F.relu(self.block5_conv2(output_camera))
        output_camera = F.relu(self.block5_conv2(output_camera))
        output_camera = self.block5_pool(output_camera)
        output_radar = self.block5_rad_pool(output_radar)
        output_camera = torch.cat((output_camera, output_radar), channel_index)
        block5_radar = output_radar
        block5_feature = output_camera

        # block6_radar ==> (None, 2, 6, 10)
        # block7_radar ==> (None, 2, 3, 5)
        block6_radar = self.block6_rad_pool(output_radar)
        block7_radar = self.block7_rad_pool(block6_radar)

        return [block3_feature, block4_feature, block5_feature],\
            [block3_radar, block4_radar, block5_radar, block6_radar, block7_radar]

    def load_pretrained_layers(self):
        """加载预训练的VGG参数"""
        # https://pytorch.org/vision/stable/models.html#classification
        # BackBoneSubmodel state
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # VGG16 state
        pretrained_state_dict = torchvision.models.vgg16(
            pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        # 每个block的首个conv层形状不同,所以不能加载预训练参数
        param_index_to_load = [1, 2, 3, 5, 6, 7, 9, 10, 11,
                               12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25]
        # param_index_to_load = [2, 3, 6, 7, 10, 11,
        #                        12, 13, 16, 17, 18, 19,
        #                        22, 23, 24, 25]

        # 加载VGG中的参数(只加载特征提取卷积层,不要最后的全连接层)
        for i in param_index_to_load:
            param = param_names[i]
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)
        print('Load VGG16 pretrained weights')


class FPNSubmodel(nn.Module):
    def __init__(self, opts):
        super(FPNSubmodel, self).__init__()
        # PX定义源码:cfrnet/model/architectures/retinanet.py:144
        self.block5_C5_reduced = nn.Conv2d(
            514, 254, kernel_size=1, stride=1, padding=0)
        self.block5_P5_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.block5_P5 = nn.Conv2d(254, 254, kernel_size=3, padding=1)

        # 非整数倍上采样
        self.block4_C4_reduced = nn.Conv2d(
            514, 254, kernel_size=1, stride=1, padding=0)
        self.block4_P4_upsample = nn.Upsample(size=(45, 80), mode='bilinear')
        self.block4_P4 = nn.Conv2d(254, 254, kernel_size=3, padding=1)

        self.block3_C3_reduced = nn.Conv2d(
            258, 254, kernel_size=1, stride=1, padding=0)
        self.block3_P3 = nn.Conv2d(254, 254, kernel_size=3, padding=1)

        # (11,20,514)==>(6,10,254)
        self.block6_P6 = nn.Conv2d(
            514, 254, kernel_size=3, stride=2, padding=1)
        self.block6_P6_relu = nn.ReLU(inplace=True)

        # (6,10,254)==>(3,5,254)
        self.block7_P7 = nn.Conv2d(
            254, 254, kernel_size=3, stride=2, padding=1)

    def forward(self, features, radar):
        block3_feature, block4_feature, block5_feature = features
        block3_radar, block4_radar, block5_radar, block6_radar, block7_radar = radar
        channel_index = 1
        '''
        __________________________________________________________________________________________________
        concat_5 (Concatenate)          (None, 11, 20, 514)  0           block5_pool[0][0]                
                                                                        rad_block5_pool[0][0]            
        __________________________________________________________________________________________________
        C5_reduced (Conv2D)             (None, 11, 20, 254)  130810      concat_5[0][0]                   
        __________________________________________________________________________________________________
        P5_upsampled (UpsampleLike)     (None, 22, 40, 254)  0           C5_reduced[0][0]                 
                                                                        concat_4[0][0]                   
        __________________________________________________________________________________________________
        C4_reduced (Conv2D)             (None, 22, 40, 254)  130810      concat_4[0][0]                   
        __________________________________________________________________________________________________
        P4_merged (Add)                 (None, 22, 40, 254)  0           P5_upsampled[0][0]               
                                                                        C4_reduced[0][0]                 
        __________________________________________________________________________________________________
        P4_upsampled (UpsampleLike)     (None, 45, 80, 254)  0           P4_merged[0][0]                  
                                                                        concat_3[0][0]                   
        __________________________________________________________________________________________________
        C3_reduced (Conv2D)             (None, 45, 80, 254)  65786       concat_3[0][0]                   
        __________________________________________________________________________________________________
        P6 (Conv2D)                     (None, 6, 10, 254)   1175258     concat_5[0][0]                   
        __________________________________________________________________________________________________
        P3_merged (Add)                 (None, 45, 80, 254)  0           P4_upsampled[0][0]               
                                                                        C3_reduced[0][0]                 
        __________________________________________________________________________________________________
        rad_block6_pool (MaxPooling2D)  (None, 6, 10, 2)     0           rad_block5_pool[0][0]            
        __________________________________________________________________________________________________
        C6_relu (Activation)            (None, 6, 10, 254)   0           P6[0][0]                         
        __________________________________________________________________________________________________
        P3 (Conv2D)                     (None, 45, 80, 254)  580898      P3_merged[0][0]                  
        __________________________________________________________________________________________________
        P4 (Conv2D)                     (None, 22, 40, 254)  580898      P4_merged[0][0]                  
        __________________________________________________________________________________________________
        P5 (Conv2D)                     (None, 11, 20, 254)  580898      C5_reduced[0][0]                 
        __________________________________________________________________________________________________
        P7 (Conv2D)                     (None, 3, 5, 254)    580898      C6_relu[0][0]                    
        __________________________________________________________________________________________________
        rad_block7_pool (MaxPooling2D)  (None, 3, 5, 2)      0           rad_block6_pool[0][0]            
        __________________________________________________________________________________________________
        P3_rad (Concatenate)            (None, 45, 80, 256)  0           P3[0][0]                         
                                                                        rad_block3_pool[0][0]            
        __________________________________________________________________________________________________
        P4_rad (Concatenate)            (None, 22, 40, 256)  0           P4[0][0]                         
                                                                        rad_block4_pool[0][0]            
        __________________________________________________________________________________________________
        P5_rad (Concatenate)            (None, 11, 20, 256)  0           P5[0][0]                         
                                                                        rad_block5_pool[0][0]            
        __________________________________________________________________________________________________
        P6_rad (Concatenate)            (None, 6, 10, 256)   0           P6[0][0]                         
                                                                        rad_block6_pool[0][0]            
        __________________________________________________________________________________________________
        P7_rad (Concatenate)            (None, 3, 5, 256)    0           P7[0][0]                         
                                                                        rad_block7_pool[0][0]            
        __________________________________________________________________________________________________
        '''

        # P5
        # block5_feature ==> (None, 514, 11, 20)
        # P5_reduced ==> (None, 254, 11, 20)
        # P5_upsampled ==> (None, 254, 22, 40)
        # P5_output ==> (None, 256, 11, 20)
        P5_reduced = self.block5_C5_reduced(block5_feature)
        P5_upsampled = self.block5_P5_upsample(P5_reduced)
        P5_output = self.block5_P5(P5_reduced)
        P5_output = torch.cat((P5_output, block5_radar), channel_index)

        # P4
        # block4_feature ==> (None, 514, 22, 40)
        # P4_reduced ==> (None, 254, 22, 40)
        # P4_merged ==> (None, 254, 22, 40)
        # P4_upsampled ==> (None, 254, 45, 80)
        # P4_output ==> (None, 256, 11, 20)
        P4_reduced = self.block4_C4_reduced(block4_feature)
        P4_merged = P4_reduced + P5_upsampled
        P4_upsampled = self.block4_P4_upsample(P4_merged)
        P4_output = self.block4_P4(P4_merged)
        P4_output = torch.cat((P4_output, block4_radar), channel_index)

        # P3
        # block3_feature ==> (None, 258, 45, 80)
        # P3_reduced ==> (None, 254, 45, 80)
        # P3_merged ==> (None, 254, 45, 80)
        # P3_output ==> (None, 256, 45, 80)
        P3_reduced = self.block3_C3_reduced(block3_feature)
        P3_merged = P3_reduced + P4_upsampled
        P3_output = self.block3_P3(P3_merged)
        P3_output = torch.cat((P3_output, block3_radar), channel_index)

        # P6
        # block5_feature ==> (None, 514, 11, 20)
        # P6_output ==> (None, 256, 6, 10)
        # P6_activated ==> (None, 254, 6, 10)
        # 原项目中 P6 接的是 concat_5,跟 C5_reduced 是一样的
        P6_output = self.block6_P6(block5_feature)
        P6_activated = F.relu(P6_output)
        P6_output = torch.cat((P6_output, block6_radar), channel_index)

        # P7
        # P6_activated ==> (None, 254, 6, 10)
        # P7_output ==>(256, 3, 5)
        P7_output = self.block7_P7(P6_activated)
        P7_output = torch.cat((P7_output, block7_radar), channel_index)

        return [P3_output, P4_output, P5_output, P6_output, P7_output]


class ClassificationSubmodel(nn.Module):
    def __init__(self, opts):
        super(ClassificationSubmodel, self).__init__()
        # 类别预测
        # retinanet.py 56
        # (256, xx, xx)==>(256, xx, xx)
        self.cls_conv1 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.cls_conv2 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.cls_conv3 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.cls_conv4 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        # 每一层都有 relu 激活函数
        # (256, xx, xx)==>(num_anchors * cls_num, xx, xx)
        # cls_num: 类别数量, 可预测类别+背景
        self.cls_num = opts.cls_num
        self.cls_predict = nn.Conv2d(
            256,
            anchor_parameters['num_anchors'] * self.cls_num,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, features):
        batch_size = features.size()[0]

        output = F.relu(self.cls_conv1(features))
        output = F.relu(self.cls_conv2(features))
        output = F.relu(self.cls_conv3(features))
        output = F.relu(self.cls_conv4(features))

        # (batch_size, num_anchors * cls_num, xx, xx)
        output = F.sigmoid(self.cls_predict(output))
        # (batch_size, xx, xx, num_anchors * cls_num)
        output = output.permute(0, 2, 3, 1).contiguous()
        # (batch_size, num_anchors * feature.shape, cls_num)
        output = output.view(batch_size, -1, self.cls_num)

        return output


class RegressionSubmodel(nn.Module):
    def __init__(self, opts):
        super(RegressionSubmodel, self).__init__()
        # 位置回归
        # retinanet.py 112
        # (256, xx, xx)==>(256, xx, xx)
        self.loc_conv1 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.loc_conv2 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.loc_conv3 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        self.loc_conv4 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1)
        # 每一层都有 relu 激活函数
        # (256, xx, xx)==>( num_anchors * num_values, xx, xx)
        # num_values: 每个预测框需要的值(4个坐标), 即(CX,CY,W,H)
        self.loc_regress = nn.Conv2d(
            256,
            anchor_parameters['num_anchors'] * 4,
            kernel_size=3, stride=1, padding=1
        )
        self.num_anchors = anchor_parameters['num_anchors']

    def forward(self, features):
        batch_size = features.size()[0]

        output = F.relu(self.loc_conv1(features))
        output = F.relu(self.loc_conv2(features))
        output = F.relu(self.loc_conv3(features))
        output = F.relu(self.loc_conv4(features))

        # (batch_size, num_anchors * 4, xx, xx)
        output = self.loc_regress(output)
        # (batch_size, xx, xx, num_anchors * 4)
        output = output.permute(0, 2, 3, 1).contiguous()
        # (batch_size, num_anchors * feature.shape, 4)
        output = output.view(batch_size, -1, 4)

        return output


class CRFNet(nn.Module):
    def __init__(self, opts, load_pretrained_vgg=False):
        super(CRFNet, self).__init__()

        # Radar Image 特征提取部分
        self.backbone = BackBoneSubmodel(opts)

        # FPN 部分
        self.fpn = FPNSubmodel(opts)

        # Classification and Regression 部分
        # 所有feature层共用一个预测层(由多个卷积层构成)
        self.classification = ClassificationSubmodel(opts)
        self.regression = RegressionSubmodel(opts)

        # 创建Prior boxes(Anchors)()
        self.anchors_cxcy = self.create_anchors()

        # 加载预训练的VGG网络
        if load_pretrained_vgg:
            self.backbone.load_pretrained_layers()

    def forward(self, x):
        # 提取特征
        features, radar = self.backbone(x)
        fpn_output = self.fpn(features, radar)

        cls, loc = [], []
        for feature in fpn_output:
            cls.append(self.classification(feature))
            loc.append(self.regression(feature))

        # loc ==> (batch_size, num_anchors * feature.shape, 4)
        # cls ==> (batch_size, num_anchors * feature.shape, cls_num)

        loc = torch.cat(loc, dim=1)
        cls = torch.cat(cls, dim=1)
        return loc, cls

    def create_anchors(self):

        anchors = []
        # 不同层输出的特征图的大小
        fmap_dims = [(45, 80), (22, 40), (11, 20), (6, 10), (3, 5)]

        # 锚点的不同长宽比
        aspect_ratios = [0.5, 1, 2]
        aspect_ratios = [np.sqrt(ratio) for ratio in aspect_ratios]

        # 不同层锚点基础大小 = 1 / [(45,80),(22,40),(11,20),(6,10),(3,5)]
        obj_scales_base = np.array([(1/h, 1/w) for h, w in fmap_dims])

        # 不同大小相对基础大小的比例
        # array([1.       , 1.2599211, 1.587401 ])
        obj_scales_ratios = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

        # 不同层锚点大小
        obj_scales = np.tile(obj_scales_base, 3)
        obj_scales = obj_scales.reshape((5, 3, 2))
        obj_scales *= np.tile(obj_scales_ratios, (2, 1)).T
        # array([[[0.02222222, 0.0125    ],
        #         [0.02799825, 0.01574901],
        #         [0.03527558, 0.01984251]],

        #        [[0.04545455, 0.025     ],
        #         [0.05726914, 0.03149803],
        #         [0.07215459, 0.03968503]],

        #        [[0.09090909, 0.05      ],
        #         [0.11453828, 0.06299605],
        #         [0.14430919, 0.07937005]],

        #        [[0.16666667, 0.1       ],
        #         [0.20998684, 0.1259921 ],
        #         [0.26456684, 0.15874011]],

        #        [[0.33333333, 0.2       ],
        #         [0.41997368, 0.25198421],
        #         [0.52913368, 0.31748021]]])

        # 不同层输出的不同大小的特征图
        for k, (height, width) in enumerate(fmap_dims):
            # 遍历锚点格子
            for y in range(height):
                for x in range(width):
                    # 中心坐标
                    cx = (x + 0.5) / width
                    cy = (y + 0.5) / height

                    scales = obj_scales[k]
                    # 不同长宽比
                    for ratio in aspect_ratios:
                        # 不同
                        for scale in scales:
                            anchors.append(
                                (cx,
                                 cy,
                                 scale[1] * ratio,
                                 scale[0] / ratio)
                            )
            pass

        anchors = torch.FloatTensor(anchors).to(device)
        # 防止越界,调整到[0,1]区间
        anchors.clamp_(0, 1)
        # (42975, 4)
        # 对应到generator.py 303
        # 不过格式是相对比例的(cx,cy,x,y)
        return anchors


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from utils.config import get_config
    config_path = 'config/default.cfg'
    config = get_config(config_path)

    from data_processing.datasets.nuscenes_dataset import NuscenesDataset
    datasets = NuscenesDataset(opts=config)
    train_loader = torch.utils.data.DataLoader(datasets, batch_size=2,
                                               collate_fn=datasets.collate_fn,
                                               shuffle=True, pin_memory=True)

    model = CRFNet(opts=config).to(device)
    crf_loss = CRFLoss(anchors_cxcy=model.anchors_cxcy)

    # forward
    for i, (images, labels, bboxes, distances, visibilities) in enumerate(train_loader):
        images = images.to(device)
        bboxes = [b.to(device) for b in bboxes]
        labels = [l.to(device) for l in labels]
        predicted_loc, predicted_cls = model(images)

        loss = crf_loss(predicted_loc, predicted_cls, labels, bboxes)
        pass
    pass