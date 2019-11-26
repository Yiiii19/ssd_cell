# config.py
import os.path

# gets home dir cross platform
# HOME = os.path.expanduser("~")
HOME = "/work/scratch/zhou"

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

# SSD 512
# voc = {
#     'num_classes': 21,
#     'lr_steps': (80000, 100000, 120000),
#     'max_iter': 120000,
#     'feature_maps': [64, 32, 16, 8, 4, 2, 1],
#     'min_dim': 512,
#     'steps': [8, 16, 32, 64, 128, 256, 512],
#     'min_sizes': [36, 77, 154, 230, 307, 384, 460],
#     'max_sizes': [77, 154, 230, 307, 384, 460, 537],
#     'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
#     'variance': [0.1, 0.2],
#     'clip': True,
#     'name': 'VOC',
# }

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

# cell = {
#     'num_classes': 2,
#     'lr_steps': (1000, 5000, 10000, 15000),
#     'max_iter': 20000,
#     'feature_maps': [38, 19, 10, 5, 3, 1],
#     'min_dim': 300,
#     'steps': [8, 16, 32, 64, 100, 300],
#     # 'min_sizes': [10, 30, 50, 80, 110, 160],
#     # 'max_sizes': [30, 50, 80, 110, 140, 160],
#     'min_sizes': [5, 10, 20, 40, 60, 100],
#     'max_sizes': [10, 20, 40, 60, 100, 120],
#     'aspect_ratios': [[1], [1, 1], [1, 1], [1, 1], [1], [1]],
#     'variance': [0.1, 0.2],
#     'clip': True,
#     'name': 'CELL',
# }

# SSD512 CONFIGS
cell = {
    'num_classes': 2,
    'lr_steps': (1000, 10000, 15000, 25000),
    'max_iter': 5000,
    'min_dim': 512,
    'min_dim_res': 513,
    # 'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [20, 40, 60, 80, 100, 120, 140],
    'max_sizes': [40, 60, 80, 100, 120, 140, 160],
    # 'min_sizes': [30, 60, 111, 162, 213, 264],
    # 'max_sizes': [60, 111, 162, 213, 264, 315],
    # 'max_sizes': [30, 50, 80, 110, 140, 160],

    # 'aspect_ratios': [[1.3], [1.3, 1.5], [1.3, 1.5], [1.3, 1.5], [1.3, 1.5], [1.3], [1.3]],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3]],
    # 'aspect_ratios': [[1.5], [1.5, 2], [1.5, 2], [1.5, 2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'CELL',

    'feature_maps': [64, 32, 16, 8],
    # 'steps': [88, 16, 32, 64],
    # 'min_sizes': [20, 40, 60, 80],
    # 'max_sizes': [40, 60, 80, 100],
}
