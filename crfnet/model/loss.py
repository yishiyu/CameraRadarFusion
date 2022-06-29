import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import cxcy_to_xy, find_jaccard_overlap, xy_to_cxcy, gcxgcy_to_cxcy, cxcy_to_gcxgcy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class focal_loss(nn.Module):
    """
    针对CRFNet设计的focal_loss
    参考losses.py中的focal函数
    """

    def __init__(self, num_classes, alpha=0.25, gamma=2, size_average=True):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        # self.cross_entropy = nn.CrossEntropyLoss()
        self.size_average = size_average

        assert alpha<1
        self.alpha = torch.zeros(num_classes)
        self.alpha[0] += alpha
        self.alpha[1:] += (1-alpha)
        self.alpha = self.alpha.to(device)
        # 0号下标为背景
        # self.alpha 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

    def forward(self, preds, labels):
        # 多一个anchor state: -1 for ignore, 0 for background, 1 for object
        # preds ==> (batch_size, 42975, cls_num + 1)
        # labels ==> (batch_size, 42975, cls_num)
        anchor_state = labels[:, :, -1]
        labels = labels[:, :, :-1]

        # filter out "ignore" anchors
        indices = (anchor_state != -1)
        labels = labels[indices]
        preds = preds[indices]
        labels = torch.argmax(labels, dim=1)
        # preds ==> (batch_size * 42975, cls_num)
        # labels ==> (batch_size * 42975)

        # 参考自 https://github.com/yatengLG/Focal-Loss-Pytorch/blob/master/Focal_Loss.py
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        # log_softmax, softmax
        preds_logsoft = torch.log(preds)
        preds_softmax = preds

        # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))

        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

        # # 计算focal loss
        # alpha_factor = torch.ones_like(labels) * self.alpha
        # alpha_factor = torch.where(labels == 1, alpha_factor, 1-alpha_factor)
        # focal_weight = torch.where(labels == 1, 1-preds, preds)
        # focal_weight = alpha_factor * focal_weight ** self.gamma

        # cls_loss = focal_weight * self.cross_entropy(labels, preds)

        # # compute the normalizer: the number of positive anchors
        # normalizer = (anchor_state == 1).shape[0]
        # normalizer = max(1.0, normalizer)

        # return torch.sum(cls_loss)/normalizer


class smooth_l1(nn.Module):
    def __init__(self, sigma=3.0, alpha=1.0):
        super(smooth_l1, self).__init__()
        self.sigma_squared = sigma**2
        self.alpha = alpha

    def forward(self, preds, labels):
        anchor_state = labels[:, :, -1]
        labels = labels[:, :, :-1]

        # filter out "ignore" anchors
        indices = (anchor_state == 1)
        labels = labels[indices]
        preds = preds[indices]

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = torch.abs(preds - labels)

        loc_loss = torch.where(
            torch.less(regression_diff, 1.0 / self.sigma_squared),
            0.5 * self.sigma_squared * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / self.sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        # normalizer = max(1.0, indices.shape[0])

        # 图像中没有任何目标
        if labels.shape[0] == 0:
            return 0

        # 直接根据目标数量取平均,就不再需要用batch_size再归一化了
        # return self.alpha * torch.mean(loc_loss) / normalizer
        return self.alpha * torch.mean(loc_loss)


class CRFLoss(nn.Module):
    """
    由两部分组成:

    1. localization loss
    2. confidence loss
    """

    def __init__(self, num_classes, focal_loss_alpha=0.25, focal_loss_gamma=2):
        super(CRFLoss, self).__init__()

        self.smooth_l1 = smooth_l1()
        self.focal_loss = focal_loss(num_classes,
                                     alpha=focal_loss_alpha,
                                     gamma=focal_loss_gamma)

    def forward(self, predicted_locs, predicted_scores, bboxes, labels):
        # predicted_locs ==> (batch_size, n_anchors , xx, xx, 4)
        # predicted_scores ==> (batch_size, num_anchors * cls_num, xx, xx)

        smooth_l1 = self.smooth_l1(predicted_locs, bboxes)
        focal_loss = self.focal_loss(predicted_scores, labels)

        return smooth_l1+focal_loss
