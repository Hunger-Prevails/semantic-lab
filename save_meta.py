import os
import json

path = 'metadata.json'

meta = dict()

mean = [0.485, 0.456, 0.406]
meta['mean'] = mean

stddvn = [0.229, 0.224, 0.225]
meta['stddvn'] = stddvn

root = dict(smile_view = '/home/yinglun.liu/Datasets/smile_view')
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
    [85, 85, 85],
]
annotation = dict(smile_view = smile_view_anno)
meta['annotation'] = annotation

with open(path, 'w') as file:
    file.write(json.dumps(meta))