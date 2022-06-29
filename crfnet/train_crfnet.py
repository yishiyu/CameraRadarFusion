import os
import cv2
import numpy as np
import time
import argparse
import torch
from utils.visualization import draw_detections
from data_processing.fusion.fusion_projection_lines import create_imagep_visualization
from data_processing.datasets.nuscenes_dataset import NuscenesDataset
from torch.utils.data import DataLoader
from utils.config import get_config
from utils.utils import AverageMeter, save_checkpoint
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

    for i, (images, bboxes, labels) in enumerate(train_loader):
        # 数据加载时间
        data_time.update(time.time() - start)

        # move to default device
        images = images.to(device)
        bboxes = bboxes.to(device)
        labels = labels.to(device)

        # 正向传播
        predicted_loc, predicted_cls = model(images)

        # 计算loss
        loss = loss_fn(predicted_loc, predicted_cls, bboxes, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新参数
        optimizer.step()

        losses.update(loss)
        batch_time.update(time.time() - start)
        start = time.time()

        # 输出训练状态
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    # 清除变量,节省内存
    del predicted_loc, predicted_cls, images, bboxes, labels

def evaluate(val_loader, model, save_path=None, render=False):
    # 用当前模型在一张图片上预测
    images, bboxes_gt, labels_gt = next(iter(val_loader))
    model.eval()

    # filtered = [
    #   [boxes, scores, labels] * batch_size
    # ]
    filtered_result = model.predict(images.to(device))

    # eval.py 65
    for i, (boxes, scores, labels) in enumerate(filtered_result):
        boxes = boxes.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        # select indices which have a score above the threshold
        indices = np.where(scores>config.score_threshold)[0]

        # select those scores
        scores = scores[indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:config.max_detections]

        # select detections
        image_boxes      = boxes[indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[indices[scores_sort]]

        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        # Create Visualization
        if save_path or render:
            viz_image = np.ascontiguousarray(images[i][:3].permute((1,2,0)).numpy()*255, dtype=np.uint8)
            # viz_image = create_imagep_visualization(images, cfg=config)
            #draw_annotations(viz_image, generator.load_annotations(i), label_to_name=generator.label_to_name) # Draw annotations
            draw_detections(viz_image, image_boxes, image_scores, image_labels,score_threshold=config.score_threshold, label_to_name=None) # Draw detections
        
        if render:
            # Show 
            try:
                cv2.imshow("debug", viz_image)
                cv2.waitKey(1)
            except Exception as e:
                print("Render error:")
                print(e)

        if save_path is not None:
            # Store
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            result = cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(i)), viz_image)
            pass
    pass


def get_data_loader(config):
    train_dataset = NuscenesDataset(data_version=config.nusc_version, opts=config)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batchsize,
                                               collate_fn=train_dataset.collate_fn(
                                                   image_dropout=config.image_dropout),
                                               shuffle=True, pin_memory=True, 
                                               num_workers=config.num_workders)
    test_loader = None
    test_dataset = NuscenesDataset(data_version=config.test_version, opts=config)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batchsize,
    #                                            collate_fn=train_dataset.collate_fn(
    #                                                image_dropout=config.image_dropout),
    #                                            shuffle=True, pin_memory=True, 
    #                                            num_workers=config.num_workders)
    val_dataset = NuscenesDataset(data_version=config.val_version, opts=config)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batchsize,
                                               collate_fn=train_dataset.collate_fn(
                                                   image_dropout=config.image_dropout),
                                               shuffle=True, pin_memory=True, 
                                               num_workers=config.num_workders)
    return train_loader,test_loader,val_loader
    # return train_loader, train_loader, train_loader

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

    # 训练数据,测试数据集,验证数据集
    train_loader, test_loader, val_loader = get_data_loader(config)

    # 训练模型参数
    checkpoint_dir = config.checkpoints_dir
    start_epoch = config.start_epoch
    epochs = config.epochs

    if start_epoch != 0:
        # 从checkpoint中恢复训练
        filename = os.path.join(
            checkpoint_dir, 'checkpoint_crfnet_{0:{1}3d}.pth.tar'.format(start_epoch-1, '0'))
        checkpoint = torch.load(filename)
        print('Load checkpoint from epoch {}'.format(start_epoch-1))
        model = checkpoint['model']
        model.classification.load_activation_layer()
        model.regression.load_activation_layer()
        optimizer = checkpoint['optimizer']

        evaluate(val_loader, model, save_path='log/epoch{}'.format(start_epoch))

    else:
        # 创建模型
        model = CRFNet(opts=config, load_pretrained_vgg=True).to(device)
        evaluate(val_loader, model, save_path='log/epoch{}'.format(start_epoch))

        # 训练参数
        lr = config.learning_rate
        # TODO 在 ecay_lr_at 个epoch后调整学习率
        # decay_lr_at = [5, 8]
        momentum = 0.3
        weight_decay = 5e-4

        parameters = []
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                parameters.append(param)
        optimizer = torch.optim.SGD(parameters,
                                    lr=lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    model = model.to(device)
    crf_loss = CRFLoss(config.cls_num).to(device)

    for epoch in range(start_epoch, epochs):
        # 调整lr

        # 训练一个epoch
        train(train_loader, model, crf_loss, optimizer, epoch)

        # save checkpoint
        save_checkpoint(checkpoint_dir, epoch, model, optimizer)

        # 使用验证集验证
        evaluate(val_loader, model, save_path='./log/epoch{}'.format(epoch))
