import numpy as np
import torch

# Anchors Parameters
# 参考代码: anchor_parameter.py 46
# AnchorParameters.small
anchor_parameters = {
    # 不同层输出的特征图的大小
    'fmap_dims': [(45, 80), (22, 40), (11, 20), (6, 10), (3, 5)],    'stride': [8, 16, 32, 64, 128],
    'ratios': [0.5, 1, 2.0],
    # 不同大小相对基础大小的比例
    # array([1.       , 1.2599211, 1.587401 ])
    'obj_scales_ratios': np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]),
    # len(ratios) * len(scales)
    'num_anchors': 9,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_anchors_xyxy_relative():
    """创建(xyxy)格式的anchor(相对比例)"""

    anchors = []
    # 不同层输出的特征图的大小
    fmap_dims = anchor_parameters['fmap_dims']

    # 锚点的不同长宽比
    aspect_ratios = anchor_parameters['ratios']
    aspect_ratios = [np.sqrt(ratio) for ratio in aspect_ratios]

    # 不同层锚点基础大小 = 1 / [(45,80),(22,40),(11,20),(6,10),(3,5)]
    obj_scales_base = np.array([(1/h, 1/w) for h, w in fmap_dims])

    # 不同大小相对基础大小的比例
    # array([1.       , 1.2599211, 1.587401 ])
    obj_scales_ratios = anchor_parameters['obj_scales_ratios']

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
                        w_half = scale[1] * ratio / 2
                        h_half = scale[0] / ratio / 2
                        # cxcyxxyy = (
                        #     cx, cy,
                        #     scale[1] * ratio,
                        #     scale[0] / ratio
                        # )
                        anchors.append(
                            (cx - w_half,
                             cy - h_half,
                             cx + w_half,
                             cy + h_half)
                        )
        pass

    anchors = torch.FloatTensor(anchors).to(device)
    # 防止越界,调整到[0,1]区间
    anchors.clamp_(0, 1)
    # (42975, 4)
    # 对应到generator.py 303
    # 不过格式是相对比例的(x,y,x,y)
    return anchors

def create_anchors_xyxy_absolute(image_shape=None):
    """
    创建(xyxy)格式的anchor(绝对像素)
    image_shape :(4),如(360, 640, 360, 640)
    """

    if image_shape==None:
        image_shape = torch.tensor((360, 640, 360, 640), device=device)
    else:
        image_shape = image_shape.to(device)

    anchors = create_anchors_xyxy_relative()
    # image_shape ==> (4)
    return anchors * image_shape

