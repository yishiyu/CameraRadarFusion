import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import cxcy_to_xy, find_jaccard_overlap, xy_to_cxcy, gcxgcy_to_cxcy, cxcy_to_gcxgcy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class focal_loss(nn.Module):
    """
    原项目地址:https://github.com/yatengLG/Focal-Loss-Pytorch
    """

    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            assert len(alpha) == num_classes
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1-preds_softmax),
                          self.gamma), preds_logsoft)

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class CRFLoss(nn.Module):
    """
    由两部分组成:

    1. localization loss
    2. confidence loss
    """

    def __init__(self, anchors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1., cls_num=8):
        super(CRFLoss, self).__init__()

        # (anchors数量, 4)
        self.anchors_cxcy = anchors_cxcy
        # 修改成xyxy格式是为了方便计算IoU
        self.anchors_xy = cxcy_to_xy(self.anchors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.SmoothL1Loss()
        self.focal_loss = focal_loss(num_classes=cls_num)

    def forward(self, predicted_locs, predicted_scores, labels, bboxes):
        # predicted_locs ==> (batch_size, n_anchors , xx, xx, 4)
        # predicted_scores ==> (batch_size, num_anchors * cls_num, xx, xx)
        batch_size = predicted_locs.size(0)
        n_anchors = self.anchors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_anchors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros(
            (batch_size, n_anchors, 4), dtype=torch.float).to(device)
        true_classes = torch.zeros(
            (batch_size, n_anchors), dtype=torch.long).to(device)

        # annotations:
        #   labels: [目标标签]
        #   bboxes: [目标框位置]
        #   distances: [目标距离]
        #   visibilities: [目标可视等级]
        #   num_radar_pts: [目标含的雷达点数]

        # For each image
        for i in range(batch_size):
            n_objects = bboxes[i].shape[0]

            # 如果图中没有任何物体,则跳过当前图片
            if n_objects==0:
                label_for_each_prior = torch.zeros((n_anchors), dtype=torch.long).to(device)
                # true_locs[i] = torch.zeros((n_anchors), dtype=torch.long).to(device)
            else:
                # 计算真实目标框和所有预测框的IoU
                # (n_objects, n_anchors)
                overlap = find_jaccard_overlap(bboxes[i],
                                            self.anchors_cxcy)

                # 找到每个预测框最可能对应的物体(IoU, index)
                # (n_anchors)
                overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)

                # 存在两个问题:
                # 1. 没有一个预测框对应某个目标框
                # 2. 与目标框对应的所有预测框IoU可能都低于阈值(被赋值为bg类)

                # 找到每个物体最可能对应的预测框(index)
                # (N_objects)
                _, prior_for_each_object = overlap.max(dim=1)

                # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
                # 解决问题 1,将目标框对应最大概率的预测框,强制设置该物体为该预测框对应的物体
                object_for_each_prior[prior_for_each_object] = torch.LongTensor(
                    range(n_objects)).to(device)

                # 解决问题 2,将每个物体对应的预测框的IoU强制设为 1
                overlap_for_each_prior[prior_for_each_object] = 1.

                # Labels for each prior
                # (n_anchors)
                label_for_each_prior = labels[i][object_for_each_prior]
                # 将 IoU 低于阈值的预测框的label设为bg
                label_for_each_prior[overlap_for_each_prior <
                                    self.threshold] = 0  # (n_anchors)

                # Store
                true_classes[i] = label_for_each_prior

                # 将原本cxcyxy格式的预测框转换格式,成相对于对应锚点的便宜(中心点便宜,长宽比例)
                true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(
                    bboxes[i][object_for_each_prior]), self.anchors_cxcy)  # (n_anchors, 4)

            pass

        # 对应物体的锚点框
        # (batch_size, n_anchors)
        positive_anchors = (true_classes != 0)

        # LOCALIZATION LOSS
        # 仅考虑与物体对应的锚点框的 Location loss
        loc_loss = self.smooth_l1(
            predicted_locs[positive_anchors], true_locs[positive_anchors])

        # CONFIDENCE LOSS
        # TODO 添加 hard negative 相关代码
        n_positives = positive_anchors.sum(dim=1)

        # predicted_scores ==> (batch_size, n_anchors, n_classes)
        # true_classes ==> (batch_size, n_anchors)
        conf_loss = self.focal_loss(
            predicted_scores.view(-1, n_classes), true_classes.view(-1))

        # Total loss
        return conf_loss + self.alpha * loc_loss
