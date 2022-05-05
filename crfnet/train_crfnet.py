import os
import time
import argparse
import torch
from data_processing.datasets.nuscenes_dataset import NuscenesDataset
from torch.utils.data import DataLoader
from utils.config import get_config
from utils.utils import AverageMeter
from model.model import CRFNet
from model.loss import CRFLoss


def train(train_loader, model, loss_fn, optimizer, epoch, print_freq=10):
    """
    One epoch's training.
    """
    model.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    for i, (images, labels, bboxes, distances, visibilities) in enumerate(train_loader):
        # 数据加载时间
        data_time.update(time.time() - start)

        # move to default device
        images = images.to(device)
        bboxes = [b.to(device) for b in bboxes]
        labels = [l.to(device) for l in labels]

        # 正向传播
        predicted_loc, predicted_cls = model(images)

        # 计算loss
        loss = loss_fn(predicted_loc, predicted_cls, labels, bboxes)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新参数
        optimizer.step()

        losses.update(time.time() - start)
        start = time.time()
        # 输出训练状态

        start = time.time()
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))


if __name__ == '__main__':
    FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

    # 读取配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default=os.path.join(FILE_DIRECTORY, "config/default.cfg"))
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(
            "ERROR: Config file \"%s\" not found" % (args.config))
    else:
        config = get_config(args.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 训练数据
    train_dataset = NuscenesDataset(data_version='v1.0-mini', opts=config)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batchsize,
                                               collate_fn=train_dataset.collate_fn,
                                               shuffle=True, pin_memory=True)

    # 创建模型
    model = CRFNet(opts=config, load_pretrained_vgg=True).to(device)
    crf_loss = CRFLoss(anchors_cxcy=model.anchors_cxcy,cls_num=config.cls_num).to(device)
    # TODO 加载 checkpoint

    # 训练参数
    lr = 1e-3
    # TODO 在 ecay_lr_at 个epoch后调整学习率
    # decay_lr_at = [5, 8]
    momentum = 0.9
    weight_decay = 5e-4

    parameters = []
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            parameters.append(param)
    optimizer = torch.optim.SGD(parameters,
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # 训练模型
    # TODO 从加载的checkpoint开始继续训练
    for epoch in range(0, config.epochs):
        # 调整lr

        # 训练一个epoch
        train(train_loader, model, crf_loss, optimizer, epoch)
        # save checkpoint
        # TODO
