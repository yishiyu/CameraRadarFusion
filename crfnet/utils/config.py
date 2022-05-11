import configparser
import ast


def get_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    class Configuration():
        def __init__(self):
            # 路径参数
            self.data_dir = config['PATH']['data_dir']
            self.checkpoints_dir = config['PATH']['checkpoints_dir']

            # 数据集参数
            self.nusc_version = config['DATASET']['nusc_version']
            self.n_sweeps = config.getint('DATASET', 'n_sweeps')

            try:
                self.category_mapping = dict(config['CATEGORY_MAPPING'])
            except:
                self.category_mapping = {
                    "vehicle.car" : "vehicle.car",
                    "vehicle.motorcycle" : "vehicle.motorcycle",
                    "vehicle.bicycle" : "vehicle.bicycle",
                    "vehicle.bus" : "vehicle.bus",
                    "vehicle.truck" : "vehicle.truck",
                    "vehicle.emergency" : "vehicle.truck",
                    "vehicle.trailer" : "vehicle.trailer",
                    "human" : "human", }
            
            # 可预测类别+背景
            self.cls_num = 8

            # 融合参数
            self.image_size = (config.getint('DATAFUSION', 'image_height'),
                               config.getint('DATAFUSION', 'image_width'))
            self.radar_projection_height = \
                config.getfloat('DATAFUSION', 'radar_projection_height')

            self.channels = ast.literal_eval(config.get('DATAFUSION', 'channels'))

            try:
                self.only_radar_annotated = config.getint('PREPROCESSING', 'only_radar_annotated')
            except:
                self.only_radar_annotated = 0

            # 训练超参数
            self.learning_rate = config.getfloat('HYPERPARAMETERS', 'learning_rate')
            self.batchsize = config.getint('HYPERPARAMETERS', 'batchsize')
            self.epochs = config.getint('HYPERPARAMETERS', 'epochs')
            self.start_epoch = config.getint('HYPERPARAMETERS','start_epoch')

    cfg = Configuration()
    return cfg
