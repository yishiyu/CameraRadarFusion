import sys
import os
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', '..', '..'))
# import crfnet.data_processing  # noqa: F401
__package__ = "crfnet.data_processing.data_loader"

from ...utils.compute_overlap import compute_overlap
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
import torch
from torch.utils import data
from ..fusion.fusion_projection_lines import imageplus_creation
from ...utils.nuscenes_helper import get_sensor_sample_data
from nuscenes.utils.data_classes import RadarPointCloud
from ...utils import radar
import numpy as np
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image, points_in_box
from ...utils.anchors import create_anchors_xyxy_absolute



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NuscenesDataset(data.Dataset):

    def __init__(
        self,
        data_version='v1.0-mini',
        opts=None,
    ):
        super(NuscenesDataset, self).__init__()
        self.opts = opts

        # NuScenes数据集
        self.radar_channel = 'RADAR_FRONT'
        self.camera_channel = 'CAM_FRONT'
        self.channels = opts.channels
        self.only_radar_annotated = opts.only_radar_annotated

        self.n_sweeps = opts.n_sweeps
        self.nusc = NuScenes(version=data_version,
                             dataroot=opts.data_dir, verbose=True)

        # 创建 name<==>label 的双向映射
        self.classes, self.labels = self._get_class_label_mapping(
            [c['name'] for c in self.nusc.category], opts.category_mapping)
        self.cls_num = len(self.labels)

        # 融合参数
        self.image_target_shape = opts.image_size
        self.height = (0, opts.radar_projection_height)

        # 获取数据集中的所有 sample
        self.samples = self.nusc.sample

        # 创建 anchors
        self.anchors = create_anchors_xyxy_absolute().cpu().numpy()

    @staticmethod
    def _get_class_label_mapping(category_names, category_mapping):
        """
        :param category_mapping: [dict] Map from original name to target name. Subsets of names are supported. 
            e.g. {'pedestrian' : 'pedestrian'} will map all pedestrian types to the same label

        :returns: 
            [0]: [dict of (str, int)] mapping from category name to the corresponding index-number
            [1]: [dict of (int, str)] mapping from index number to category name
        """

        # Initialize local variables
        original_name_to_label = {}
        original_category_names = category_names.copy()
        # yishiyu 将bg放在第一个(下标为0)
        original_category_names.insert(0, 'bg')
        # original_category_names.append('bg')
        if category_mapping is None:
            # Create identity mapping and ignore no class
            category_mapping = dict()
            for cat_name in category_names:
                category_mapping[cat_name] = cat_name

        # List of unique class_names
        selected_category_names = set(category_mapping.values())  # unordered
        selected_category_names = list(selected_category_names)
        selected_category_names.sort()  # ordered
        # yishiyu 将bg放在第一个(下标为0)
        selected_category_names.insert(0, 'bg')

        # Create the label to class_name mapping
        label_to_name = {label: name for label,
                         name in enumerate(selected_category_names)}
        # label_to_name[len(label_to_name)] = 'bg'  # Add the background class

        # Create original class name to label mapping
        for label, label_name in label_to_name.items():

            # Looking for all the original names that are adressed by label name
            targets = [
                original_name for original_name in original_category_names if label_name in original_name]

            # Assigning the same label for all adressed targets
            for target in targets:

                # Check for ambiguity
                assert target not in original_name_to_label.keys(
                ), 'ambigous mapping found for (%s->%s)' % (target, label_name)

                # Assign label to original name
                # Some label_names will have the same label, which is totally fine
                original_name_to_label[target] = label

        # Check for correctness
        actual_labels = original_name_to_label.values()
        # we want to start labels at 0
        expected_labels = range(0, max(actual_labels)+1)
        assert all([label in actual_labels for label in expected_labels]
                   ), 'Expected labels do not match actual labels'

        return original_name_to_label, label_to_name

    def __len__(self):
        return len(self.samples)

    def load_image(self, index):
        """加载图像数据
        """
        sample = self.samples[index]
        camera_token = sample['data'][self.camera_channel]
        camera_sample = get_sensor_sample_data(
            self.nusc, sample, self.camera_channel,
            dtype=np.float32, size=None
        )

        # TODO Add noise to the image if enabled
        # nuscenes_generator.py 354行

        return camera_token, camera_sample

    def load_radar(self, index):
        """加载雷达数据
        """
        sample = self.samples[index]
        radar_token = sample['data'][self.radar_channel]

        # TODO noise_filter
        # noise_filter没有开源出来
        # nuscenes_generator.py 374,385,392

        pcs, times = RadarPointCloud.from_file_multisweep(self.nusc, sample, self.radar_channel,
                                                          self.radar_channel, nsweeps=self.n_sweeps, min_distance=0.0, merge=False)

        radar_sample = [radar.enrich_radar_data(pc.points) for pc in pcs]

        # radar_sample = [radar_data * frame]
        # radar_data.shape = (21, count)
        # 21: 每个雷达点有21个属性,radar.py:194行
        ## count: 雷达点数
        # 如果没有雷达点数据就创建空向量代替
        # 多个雷达帧合并
        if len(radar_sample) == 0:
            radar_sample = np.zeros(shape=(len(radar.channel_map), 0))
        else:
            radar_sample = np.concatenate(radar_sample, axis=-1)

        radar_sample = radar_sample.astype(dtype=np.float32)

        # TODO perfect_noise_filter
        # 降噪处理,后面再添加
        # TODO normalize_radar
        # 雷达点正则化处理
        # 用跟处理图像一样的方法处理
        # 具体参考nuscenes_generator.py 418

        return radar_token, radar_sample

    def load_annotations(self, index):
        """加载标注数据
        """
        sample = self.samples[index]
        annotations = {
            'labels': [],       # <list of n int>
            'bboxes': [],       # <list of n x 4 float> [xmin, ymin, xmax, ymax]
            # <list of n float>  Center of box given as x, y, z.
            'distances': [],
            'visibilities': [],  # <list of n enum> nuscenes.utils.geometry_utils.BoxVisibility
            'num_radar_pts': []  # <list of n int>  number of radar points that cover that annotation
        }
        # 读取目标框和摄像机参数
        camera_data = self.nusc.get(
            'sample_data', sample['data'][self.camera_channel])
        _, boxes, camera_intrinsic = self.nusc.get_sample_data(
            camera_data['token'], box_vis_level=BoxVisibility.ANY)
        # 原始图像数据以及缩放比例
        imsize_src = (camera_data['width'], camera_data['height'])
        bbox_resize = [self.image_target_shape[0] / camera_data['height'],
                       self.image_target_shape[1] / camera_data['width']]
        for box in boxes:
            # 只处理一部分目标的box
            if box.name in self.classes:
                box.label = self.classes[box.name]
                # BoxVisibility.ANY: box至少有一部分在图像中
                if box_in_image(box=box, intrinsic=camera_intrinsic, imsize=imsize_src, vis_level=BoxVisibility.ANY):

                    # 筛选出至少有 2 个雷达点的box
                    if self.only_radar_annotated == 2:
                        pcs, times = RadarPointCloud.from_file_multisweep(self.nusc, sample, self.radar_sensors[0],
                                                                          self.camera_channel, nsweeps=self.n_sweeps, min_distance=0.0, merge=False)

                        for pc in pcs:
                            pc.points = radar.enrich_radar_data(pc.points)

                        if len(pcs) > 0:
                            radar_sample = np.concatenate(
                                [pc.points for pc in pcs], axis=-1)
                        else:
                            print(
                                "[WARNING] only_radar_annotated=2 and sweeps=0 removes all annotations")
                            radar_sample = np.zeros(
                                shape=(len(radar.channel_map), 0))
                        radar_sample = radar_sample.astype(dtype=np.float32)

                        mask = points_in_box(box, radar_sample[0:3, :])
                        if True not in mask:
                            # 没有任何一个box满足条件
                            continue

                    # 把box映射到2d
                    # [xmin, ymin, xmax, ymax]
                    box2d = box.box2d(camera_intrinsic)
                    box2d[0] *= bbox_resize[1]
                    box2d[1] *= bbox_resize[0]
                    box2d[2] *= bbox_resize[1]
                    box2d[3] *= bbox_resize[0]

                    annotations['bboxes'].append(box2d)
                    annotations['labels'].append(box.label)
                    annotations['num_radar_pts'].append(
                        self.nusc.get('sample_annotation',
                                      box.token)['num_radar_pts']
                    )
                    distance = (box.center[0]**2 +
                                box.center[1]**2 +
                                box.center[2]**2)**0.5
                    annotations['distances'].append(distance)
                    annotations['visibilities'].append(
                        int(self.nusc.get('sample_annotation', box.token)['visibility_token']))

        annotations['labels'] = np.array(annotations['labels'])
        annotations['bboxes'] = np.array(annotations['bboxes'])
        annotations['distances'] = np.array(annotations['distances'])
        annotations['num_radar_pts'] = np.array(annotations['num_radar_pts'])
        annotations['visibilities'] = np.array(annotations['visibilities'])

        # 筛选出至少有 1 个雷达点的box
        # 感觉上面那个筛选也可以这样写,不知道为什么不这么写,怪怪的
        if self.only_radar_annotated == 1:

            anns_to_keep = np.where(annotations['num_radar_pts'])[0]

            for key in annotations:
                annotations[key] = annotations[key][anns_to_keep]

        # TODO filter_annotations_enabled
        # 额外的筛选

        return annotations

    def compute_gt_annotations(
        self,
        anchors,
        annotations,
        negative_overlap=0.4,
        positive_overlap=0.5
    ):
        """ Obtain indices of gt annotations with the greatest overlap.

        Args
            anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
            annotations: np.array of shape (N, 5) for (x1, y1, x2, y2, label).
            negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
            positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

        Returns
            positive_indices: indices of positive anchors
            ignore_indices: indices of ignored anchors
            argmax_overlaps_inds: ordered overlaps indices
        """

        # overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
        overlaps = compute_overlap(anchors, annotations)
        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        max_overlaps = overlaps[np.arange(
            overlaps.shape[0]), argmax_overlaps_inds]

        # assign "dont care" labels
        positive_indices = max_overlaps >= positive_overlap
        ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

        return positive_indices, ignore_indices, argmax_overlaps_inds

    def bbox_transform(self, anchors, gt_boxes, mean=None, std=None):
        """Compute bounding-box regression targets for an image."""

        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError(
                'Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError(
                'Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]

        targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
        targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
        targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
        targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

        targets = np.stack(
            (targets_dx1, targets_dy1, targets_dx2, targets_dy2))
        targets = targets.T

        targets = (targets - mean) / std

        return targets

    def compute_targets(self, anchors, images, annotations,
                        distance=False,
                        negative_overlap=0.4,
                        positive_overlap=0.5,
                        distance_scaling=100):
        """根据anchors和annotations生成targets

        Args:
            anchors (np.array): (N,4)==>(x1,y1,x2,y2)
            images (np.array): 图像数据
            annotations (dict): 标注数据
            distance (bool, optional): 是否生成距离标签. Defaults to False.
            negative_overlap (float, optional): IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative). Defaults to 0.4.
            positive_overlap (float, optional): IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive). Defaults to 0.5.
            distance_scaling (int, optional): 距离上限-. Defaults to 100.
        Return:
            regression_targets: 目标框
            labels_targets: 分类标签
        """
        # 对应anchor_calc.py 48
        assert('bboxes' in annotations), "Annotations should contain bboxes."
        assert('labels' in annotations), "Annotations should contain labels."

        # regression_targets ==> (N, x1, y1, x2, y2, states)
        # states:-1 for ignore, 0 for bg, 1 for fg
        # labels_targets ==> (N, cls_num+1)
        regression_targets = torch.zeros(
            (anchors.shape[0], 4+1), dtype=torch.float)
        labels_targets = torch.zeros(
            (anchors.shape[0], self.cls_num+1), dtype=torch.float)
        # 将默认类别设为bg
        labels_targets[:,self.classes['bg']] = 1
        # distance_targets = torch.zeros((anchors.shape[0], 1+1), dtype=torch.float, device=device)

        # 该场景中存在目标
        if annotations['bboxes'].shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            positive_indices, ignore_indices, argmax_overlaps_inds = \
                self.compute_gt_annotations(
                    anchors, annotations['bboxes'], negative_overlap, positive_overlap)

            labels_targets[ignore_indices, -1] = -1
            labels_targets[positive_indices, -1] = 1

            regression_targets[ignore_indices, -1] = -1
            regression_targets[positive_indices, -1] = 1

            # distance_batch[index, ignore_indices, -1]   = -1
            # distance_batch[index, positive_indices, -1] = 1

            # compute target class labels
            pos_overlap_inds = [argmax_overlaps_inds[positive_indices]]
            label_indices = annotations['labels'][tuple(
                pos_overlap_inds)].astype(int)

            labels_targets[positive_indices, self.classes['bg']] = 0
            labels_targets[positive_indices, label_indices] = 1

            regression_targets[:, :-1] = torch.tensor(self.bbox_transform(
                anchors, annotations['bboxes'][argmax_overlaps_inds, :]))

        return regression_targets, labels_targets

    def __getitem__(self, index):
        # 传感器名
        # self.radar_channel = 'RADAR_FRONT'
        # self.camera_channel = 'CAM_FRONT'
        # 融合后图像大小
        # self.image_target_shape
        sample = self.samples[index]

        # 1. 加载图像数据
        camera_token, camera_sample = self.load_image(index)

        # 2. 加载雷达数据
        radar_token, radar_sample = self.load_radar(index)

        # 3. 数据融合
        # 雷达投影高度
        # self.height = (0, opts.radar_projection_height)
        # 融合后图像大小
        # self.image_target_shape = (opts.image_min_side, opts.image_max_side)
        kwargs = {
            'pointsensor_token': radar_token,
            'camera_token': sample['data'][self.camera_channel],
            'height': self.height,
            'image_target_shape': self.image_target_shape,
            'clear_radar': False,
            'clear_image': False,
        }
        image_full = imageplus_creation(
            self.nusc, image_data=camera_sample, radar_data=radar_sample, **kwargs)
        # 只取需要的通道(R,G,B,rcs,distance)
        image_full = np.array(image_full[:, :, self.channels])

        # 4. 加载标注数据
        annotations = self.load_annotations(index)

        # TODO 数据增强
        # generator.py 332
        # TODO 根据网络调整

        # anchor相关
        # generator.py 298
        regression_targets, labels_targets = self.compute_targets(self.anchors, image_full, annotations)

        image_full = image_full.transpose(2, 0, 1)
        # image_full ==> (5, 360, 640)
        # channel first格式,fusion_projection_lines.py中的一些函数用的时候需要调整一下格式
        # 调整一下通道顺序就行了
        return image_full, regression_targets, labels_targets

    @staticmethod
    def collate_fn(image_dropout):
        """
        指导dataloader如何把不同数据中不同数量的目标(框)合并成一个batch
        调用后生成一个合并函数
        image_dropout: 图片清空的概率
        """

        def collecter(batch):
            images = []
            bboxes = []
            labels = []

            for b in batch:
                if np.random.rand() < image_dropout:
                    images.append(torch.zeros(b[0].shape))
                else:
                    images.append(torch.tensor(b[0]))
                bboxes.append(b[1])
                labels.append(b[2])

            images = torch.stack(images, dim=0)
            bboxes = torch.stack(bboxes, dim=0)
            labels = torch.stack(labels, dim=0)

            return images, bboxes, labels

        return collecter


if __name__ == '__main__':

    from ...utils.config import get_config

    config_path = 'config/default.cfg'
    config = get_config(config_path)

    datasets = NuscenesDataset(opts=config)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=config.batchsize,
                                               collate_fn=datasets.collate_fn(
                                                   image_dropout=config.image_dropout),
                                               shuffle=True, pin_memory=True, 
                                               num_workers=config.num_workders)

    for i, data in enumerate(dataloader):

        pass

    data = datasets.__getitem__(0)

    import cv2
    from ..fusion.fusion_projection_lines import create_imagep_visualization
    imgp_viz = create_imagep_visualization(data[0].transpose(1, 2, 0))
    cv2.imshow('image', imgp_viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
