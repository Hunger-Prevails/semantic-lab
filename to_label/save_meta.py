import os
import json

meta = dict()

mean = [0.485, 0.456, 0.406]
meta['mean'] = mean

stddvn = [0.229, 0.224, 0.225]
meta['stddvn'] = stddvn

root = dict(
    smile_view = '/home/yinglun.liu/Datasets/smile_view',
    smile_architect = '/home/yinglun.liu/Datasets/smile_architect',
    senezh_align = '/home/yinglun.liu/Datasets/senezh_align',
    senezh_exocad = '/home/yinglun.liu/Datasets/senezh_exocad'
)
meta['root'] = root

smile_view_anno = [
    [255, 255, 255],
    [0, 255, 0],
    [0, 0, 0],
    [80, 80, 80],
    [255, 0, 0],
    [5, 5, 5],
    [10, 10, 10],
    [15, 15, 15],
    [20, 20, 20],
    [25, 25, 25],
    [30, 30, 30],
    [35, 35, 35],
    [40, 40, 40],
    [45, 45, 45],
    [50, 50, 50],
    [55, 55, 55],
    [60, 60, 60],
    [65, 65, 65],
    [70, 70, 70],
    [75, 75, 75],
    [160, 160, 160],
    [155, 155, 155],
    [150, 150, 150],
    [145, 145, 145],
    [140, 140, 140],
    [135, 135, 135],
    [130, 130, 130],
    [125, 125, 125],
    [120, 120, 120],
    [115, 115, 115],
    [110, 110, 110],
    [105, 105, 105],
    [100, 100, 100],
    [95, 95, 95],
    [90, 90, 90],
    [85, 85, 85]
]
senezh_anno = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [128, 128, 0],
    [0, 64, 128],
    [128, 64, 128],
    [0, 192, 128],
    [128, 192, 128],
    [64, 64, 0],
    [192, 64, 0],
    [64, 192, 0],
    [192, 192, 0],
    [64, 64, 128],
    [194, 64, 128],
    [64, 192, 128],
    [192, 192, 128],
    [0, 0, 64],
    [128, 0, 64],
    [0, 128, 64],
    [128, 128, 64]
]
annotation = dict(
    smile_view = smile_view_anno,
    smile_architect = smile_view_anno,
    senezh_align = smile_view_anno,
    senezh_exocad = senezh_anno
)
meta['annotation'] = annotation

smile_view_mirror = [
    [ 5,  3],
    [ 6, 19],
    [ 7, 18],
    [ 8, 17],
    [ 9, 16],
    [10, 15],
    [11, 14],
    [12, 13],
    [20, 35],
    [21, 34],
    [22, 33],
    [23, 32],
    [24, 31],
    [25, 30],
    [26, 29],
    [27, 28]
]
mirror = dict(
    smile_view = smile_view_mirror,
    smile_architect = smile_view_mirror,
    senezh_align = smile_view_mirror,
    senezh_exocad = smile_view_mirror
)
meta['mirror'] = mirror

with open('/home/yinglun.liu/Datasets/metadata.json', 'w') as file:
    file.write(json.dumps(meta))
