import os
from random import sample
from turtle import distance
from matplotlib.text import Annotation
from torch.utils import data
import torch
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

# if __name__ == '__main__':
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', '..', '..'))
# import crfnet.data_processing  # noqa: F401
__package__ = "crfnet.data_processing.data_loader"

from ..fusion.fusion_projection_lines import imageplus_creation
from ...utils.nuscenes_helper import get_sensor_sample_data
from nuscenes.utils.data_classes import RadarPointCloud
from ...utils import radar
import numpy as np
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image, points_in_box


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

        # 融合参数
        self.image_target_shape = opts.image_size
        self.height = (0, opts.radar_projection_height)

        # 获取数据集中的所有 sample
        self.samples = self.nusc.sample

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
        original_category_names.insert(0,'bg')
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

        # TODO anchor相关
        # generator.py 298

        image_full = image_full.transpose(2,0,1)
        # image_full ==> (5, 360, 640)
        # channel first格式,fusion_projection_lines.py中的一些函数用的时候需要调整一下格式
        # 调整一下通道顺序就行了
        return image_full, annotations

    def collate_fn(self, batch):
        """
        指导dataloader如何把不同数据中不同数量的目标(框)合并成一个batch
        """
        images = []
        labels = []
        bboxes = []
        distances = []
        visibilities = []

        for b in batch:
            images.append(torch.tensor(b[0]))
            labels.append(torch.tensor(b[1]['labels']))
            bboxes.append(torch.tensor(b[1]['bboxes']))
            distances.append(torch.tensor(b[1]['distances']))
            visibilities.append(torch.tensor(b[1]['visibilities']))
        
        images = torch.stack(images, dim=0)
        

        return images, labels, bboxes, distances, visibilities

if __name__ == '__main__':

    from ...utils.config import get_config

    config_path = 'config/default.cfg'
    config = get_config(config_path)

    datasets = NuscenesDataset(opts=config)

    data = datasets.__getitem__(0)

    import cv2
    from ..fusion.fusion_projection_lines import create_imagep_visualization
    imgp_viz = create_imagep_visualization(data[0].transpose(1,2,0))
    cv2.imshow('image', imgp_viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

